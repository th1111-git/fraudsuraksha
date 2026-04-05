"""
FraudLens — Databricks-hosted fraud detection app.
Triggers notebooks via the Databricks Jobs API and surfaces results.

Environment variables (set in Databricks Apps UI → Environment):
  DATABRICKS_HOST   — e.g. https://adb-xxxx.azuredatabricks.net
  DATABRICKS_TOKEN  — Personal Access Token (PAT)
  RF_JOB_ID         — Job ID wrapping "Supervised Model- Random Forest"
  ISO_JOB_ID        — Job ID wrapping "Unsupervised model- Isolation forest"
"""

import os, io, json, time, logging
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB

# ── Databricks config ────────────────────────────────────────────────────────
DATABRICKS_HOST  = os.environ.get("DATABRICKS_HOST",  "https://<your-workspace>.azuredatabricks.net")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "<your-personal-access-token>")
RF_JOB_ID        = 956652827269953
ISO_JOB_ID       = os.environ.get("ISO_JOB_ID",       "")

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type":  "application/json",
}

# ── Databricks helpers ────────────────────────────────────────────────────────

def _trigger_job(job_id: str, params: dict) -> str:
    """Run a Databricks job and return the run_id."""
    url  = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    body = {"job_id": int(job_id), "notebook_params": params}
    r    = requests.post(url, headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return str(r.json()["run_id"])


def _poll_run(run_id: str, timeout_s: int = 600) -> None:
    """Poll until the run finishes; raise on failure or timeout."""
    url      = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get"
    deadline = time.time() + timeout_s
    terminal = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}

    while time.time() < deadline:
        r     = requests.get(url, headers=HEADERS, params={"run_id": run_id}, timeout=20)
        r.raise_for_status()
        state = r.json()["state"]
        if state["life_cycle_state"] in terminal:
            if state.get("result_state") != "SUCCESS":
                raise RuntimeError(
                    f"Run {run_id} ended: {state.get('result_state')} — "
                    f"{state.get('state_message', '')}"
                )
            return
        log.info("Run %s — %s", run_id, state["life_cycle_state"])
        time.sleep(6)

    raise TimeoutError(f"Run {run_id} did not complete within {timeout_s}s")


def _get_output(run_id: str) -> dict:
    """Retrieve notebook output JSON from a completed run."""
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output"
    r   = requests.get(url, headers=HEADERS, params={"run_id": run_id}, timeout=20)
    r.raise_for_status()
    raw = r.json().get("notebook_output", {}).get("result", "null")
    try:
        return json.loads(raw) if raw and raw != "null" else {}
    except (json.JSONDecodeError, TypeError):
        return {"raw_output": raw}


def run_notebook(job_id: str, csv_text: str, model_label: str) -> dict:
    """Trigger → poll → return parsed notebook output."""
    run_id = _trigger_job(job_id, {"csv_data": csv_text, "model_type": model_label})
    log.info("Triggered %s job — run_id=%s", model_label, run_id)
    _poll_run(run_id)
    return _get_output(run_id)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "fraudlens"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    models = request.form.getlist("models") or ["rf", "iso"]

    try:
        raw    = f.read().decode("utf-8", errors="replace")
        df     = pd.read_csv(io.StringIO(raw))
        n_rows = len(df)
        cols   = list(df.columns)
    except Exception as exc:
        return jsonify({"error": f"CSV parse error: {exc}"}), 400

    if n_rows == 0:
        return jsonify({"error": "Uploaded CSV is empty"}), 400

    results = {}

    if "rf" in models:
        if not RF_JOB_ID:
            results["random_forest"] = {"error": "RF_JOB_ID environment variable is not set"}
        else:
            try:
                results["random_forest"] = run_notebook(RF_JOB_ID, raw, "random_forest")
            except Exception as exc:
                log.exception("RF job failed")
                results["random_forest"] = {"error": str(exc)}

    if "iso" in models:
        if not ISO_JOB_ID:
            results["isolation_forest"] = {"error": "ISO_JOB_ID environment variable is not set"}
        else:
            try:
                results["isolation_forest"] = run_notebook(ISO_JOB_ID, raw, "isolation_forest")
            except Exception as exc:
                log.exception("ISO job failed")
                results["isolation_forest"] = {"error": str(exc)}

    return jsonify({"total_rows": n_rows, "columns": cols, "results": results})


# ── Inline HTML ───────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FraudLens · Transaction Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Syne+Mono&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#060810;--panel:#0b0d1a;--panel2:#0f1224;
  --border:#181d38;--border2:#21274a;
  --red:#ff2d55;--amber:#ffb800;--green:#00e5a0;--blue:#4da6ff;
  --text:#d8dff5;--sub:#5a6080;--sub2:#8890b8;
  --head:'Syne',sans-serif;--mono:'IBM Plex Mono',monospace;--logo:'Syne Mono',monospace;
  --r:10px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:var(--mono);background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden;
}
/* ── grid bg ── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    radial-gradient(ellipse 70% 50% at 50% -5%,rgba(77,166,255,.06) 0%,transparent 70%),
    linear-gradient(var(--border) 1px,transparent 1px),
    linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:100%,52px 52px,52px 52px;
}
/* ── scanlines ── */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px);
}
.wrap{position:relative;z-index:1}

/* ─── HEADER ─── */
header{
  border-bottom:1px solid var(--border2);
  background:rgba(6,8,16,.9);backdrop-filter:blur(20px);
  position:sticky;top:0;z-index:200;
}
.h-inner{
  max-width:1160px;margin:0 auto;padding:15px 28px;
  display:flex;align-items:center;gap:20px;
}
.logo{font-family:var(--logo);font-size:1.3rem;letter-spacing:.06em;display:flex;align-items:center;gap:10px}
.pulse{
  width:9px;height:9px;border-radius:50%;background:var(--red);
  box-shadow:0 0 0 0 rgba(255,45,85,.5);
  animation:beat 2s ease infinite;
}
@keyframes beat{
  0%{box-shadow:0 0 0 0 rgba(255,45,85,.5)}
  60%{box-shadow:0 0 0 8px rgba(255,45,85,0)}
  100%{box-shadow:0 0 0 0 rgba(255,45,85,0)}
}
.h-divider{flex:1}
.h-tag{
  font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--sub2);border:1px solid var(--border2);padding:5px 11px;border-radius:4px;
}
.h-live{display:flex;align-items:center;gap:7px;font-size:.6rem;letter-spacing:.1em;color:var(--green)}
.h-live::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}

/* ─── MAIN ─── */
main{max-width:1160px;margin:0 auto;padding:52px 28px 100px}

/* ─── HERO ─── */
.hero{margin-bottom:60px;max-width:640px}
.hero-eyebrow{
  font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;
  color:var(--sub2);margin-bottom:18px;display:flex;align-items:center;gap:10px;
}
.hero-eyebrow::before{content:'';width:24px;height:1px;background:var(--red)}
.hero h1{
  font-family:var(--head);font-size:clamp(2.4rem,5vw,4rem);
  font-weight:800;line-height:1.02;letter-spacing:-.035em;margin-bottom:18px;
}
.hero h1 em{font-style:normal;color:var(--red);text-shadow:0 0 50px rgba(255,45,85,.3)}
.hero p{color:var(--sub2);font-size:.8rem;line-height:1.8;max-width:460px}

/* ─── CARDS ─── */
.card{
  background:var(--panel);border:1px solid var(--border2);
  border-radius:var(--r);padding:28px 32px;margin-bottom:20px;
}
.section-label{
  font-size:.6rem;letter-spacing:.2em;text-transform:uppercase;color:var(--sub);
  margin-bottom:18px;display:flex;align-items:center;gap:10px;
}
.section-label span{
  background:var(--red);color:#fff;font-size:.55rem;
  padding:2px 7px;border-radius:3px;letter-spacing:.1em;
}

/* ─── DROP ZONE ─── */
#dz{
  border:2px dashed var(--border2);border-radius:8px;
  padding:48px 20px;text-align:center;cursor:pointer;
  transition:border-color .2s,background .2s;position:relative;overflow:hidden;
}
#dz.over,#dz:hover{border-color:var(--blue);background:rgba(77,166,255,.03)}
#dz svg{margin:0 auto 14px;display:block;color:var(--sub);transition:color .2s}
#dz:hover svg{color:var(--blue)}
#dz p{color:var(--sub2);font-size:.75rem;line-height:1.9}
#dz b{color:var(--text)}
#dz small{font-size:.62rem;color:var(--sub)}
#fi{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
#fn{
  display:none;margin-top:14px;
  font-size:.7rem;color:var(--green);
  background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.2);
  border-radius:6px;padding:9px 16px;text-align:left;letter-spacing:.02em;
}

/* ─── MODEL TOGGLES ─── */
.models{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px}
@media(max-width:600px){.models{grid-template-columns:1fr}}
.mtog{
  border:1px solid var(--border2);border-radius:8px;padding:16px 18px;
  cursor:pointer;transition:border-color .2s,background .2s;
  display:flex;align-items:flex-start;gap:13px;position:relative;
}
.mtog.on{border-color:rgba(255,45,85,.5);background:rgba(255,45,85,.04)}
.mtog input[type=checkbox]{
  appearance:none;width:15px;height:15px;min-width:15px;margin-top:3px;
  border:1.5px solid var(--border2);border-radius:3px;
  cursor:pointer;transition:.15s;display:flex;align-items:center;justify-content:center;
}
.mtog input[type=checkbox]:checked{background:var(--red);border-color:var(--red)}
.mtog input[type=checkbox]:checked::after{content:'✓';color:#fff;font-size:.6rem;display:block;text-align:center;line-height:1;margin-top:1px}
.mt-name{font-size:.75rem;font-weight:500;margin-bottom:5px;letter-spacing:.02em}
.mt-desc{font-size:.63rem;color:var(--sub2);line-height:1.55}
.mt-pill{
  position:absolute;top:10px;right:12px;
  font-size:.52rem;letter-spacing:.1em;text-transform:uppercase;padding:3px 8px;border-radius:3px;
}
.sup{background:rgba(77,166,255,.14);color:var(--blue)}
.unsup{background:rgba(255,184,0,.12);color:var(--amber)}

/* ─── RUN BUTTON ─── */
#run{
  margin-top:22px;width:100%;padding:15px 24px;
  border:none;border-radius:8px;
  background:var(--red);color:#fff;
  font-family:var(--mono);font-size:.75rem;font-weight:500;
  letter-spacing:.14em;text-transform:uppercase;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:10px;
  transition:opacity .2s,transform .15s,box-shadow .25s;
  box-shadow:0 0 28px rgba(255,45,85,.22);
}
#run:hover:not(:disabled){opacity:.9;transform:translateY(-1px);box-shadow:0 6px 32px rgba(255,45,85,.38)}
#run:disabled{opacity:.35;cursor:not-allowed;transform:none;box-shadow:none}
.spin{width:13px;height:13px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}

/* ─── PROGRESS ─── */
#prog{display:none;margin-bottom:20px}
.ps{
  display:flex;align-items:center;gap:14px;
  padding:11px 18px;border-radius:8px;
  background:var(--panel);border:1px solid var(--border);
  margin-bottom:8px;font-size:.7rem;
  transition:border-color .3s;
}
.ps.run{border-color:var(--amber)}
.ps.ok{border-color:var(--green)}
.ps.err{border-color:var(--red)}
.pi{font-size:1rem;width:20px;text-align:center}
.pt{flex:1;letter-spacing:.04em}
.ptag{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;padding:3px 9px;border-radius:3px}
.t-q{background:rgba(255,255,255,.05);color:var(--sub)}
.t-r{background:rgba(255,184,0,.13);color:var(--amber)}
.t-ok{background:rgba(0,229,160,.1);color:var(--green)}
.t-e{background:rgba(255,45,85,.14);color:var(--red)}

/* ─── RESULTS ─── */
#res{display:none}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:24px}
.sc{
  background:var(--panel);border:1px solid var(--border2);border-radius:var(--r);
  padding:20px 22px;position:relative;overflow:hidden;
}
.sc::after{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.sc.r::after{background:var(--red)}  .sc.g::after{background:var(--green)}
.sc.a::after{background:var(--amber)}.sc.b::after{background:var(--blue)}
.sl{font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:var(--sub);margin-bottom:8px}
.sv{font-family:var(--head);font-size:2.1rem;font-weight:800;line-height:1;letter-spacing:-.02em}
.sv.r{color:var(--red)}.sv.g{color:var(--green)}.sv.a{color:var(--amber)}.sv.b{color:var(--blue)}
.ss{font-size:.62rem;color:var(--sub);margin-top:5px}

/* ─── COMPARE ─── */
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}
@media(max-width:700px){.cmp{grid-template-columns:1fr}}
.cp{background:var(--panel);border:1px solid var(--border2);border-radius:var(--r);overflow:hidden}
.cph{padding:13px 18px;border-bottom:1px solid var(--border2);background:var(--panel2);display:flex;align-items:center;justify-content:space-between}
.cpn{font-family:var(--head);font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.cpb{padding:16px 18px}
.ms{display:flex;justify-content:space-between;align-items:center;font-size:.7rem;padding:8px 0;border-bottom:1px solid var(--border)}
.ms:last-child{border:none}
.msl{color:var(--sub2)}.msv{font-weight:500}
.err-txt{color:var(--red);font-size:.7rem;padding:8px 0}

/* ─── TABLE ─── */
.tbl-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:10px}
.tbl-hdr h3{font-family:var(--head);font-size:.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.tc{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.fb{
  font-family:var(--mono);font-size:.6rem;padding:6px 12px;border-radius:5px;
  border:1px solid var(--border2);background:transparent;color:var(--sub);
  cursor:pointer;letter-spacing:.06em;transition:.15s;
}
.fb.on{border-color:var(--red);color:var(--red)}
.fb:hover:not(.on){border-color:var(--sub);color:var(--text)}
#dl{
  font-family:var(--mono);font-size:.6rem;padding:6px 12px;border-radius:5px;
  border:1px solid var(--green);background:transparent;color:var(--green);
  cursor:pointer;letter-spacing:.06em;transition:.15s;
}
#dl:hover{background:rgba(0,229,160,.08)}
.twrap{overflow-x:auto;border-radius:8px;border:1px solid var(--border2)}
table{width:100%;border-collapse:collapse;font-size:.68rem}
thead th{
  background:var(--panel2);padding:10px 14px;text-align:left;
  color:var(--sub);font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;
  border-bottom:1px solid var(--border2);white-space:nowrap;
}
tbody tr{border-bottom:1px solid var(--border);transition:background .1s}
tbody tr:hover{background:rgba(255,255,255,.025)}
tbody td{padding:9px 14px;white-space:nowrap}
.rh{color:var(--red);font-weight:500;display:inline-flex;align-items:center;gap:5px}
.rh::before{content:'●';font-size:.45rem}
.rm{color:var(--amber)}.rl{color:var(--sub2)}
.bar{display:inline-block;height:3px;border-radius:2px;vertical-align:middle;margin-left:8px;min-width:3px}

footer{
  border-top:1px solid var(--border);padding:20px 28px;text-align:center;
  font-size:.6rem;color:var(--sub);letter-spacing:.06em;
}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="h-inner">
    <div class="logo"><div class="pulse"></div>FRAUDLENS</div>
    <div style="font-size:.6rem;color:var(--sub);letter-spacing:.06em">Transaction Intelligence Platform</div>
    <div class="h-divider"></div>
    <span class="h-tag">Databricks · Spark</span>
    <span class="h-live">ALL SYSTEMS LIVE</span>
  </div>
</header>

<main>
  <div class="hero">
    <div class="hero-eyebrow">Fraud Detection System</div>
    <h1>Flag <em>Suspicious</em><br/>Transactions.</h1>
    <p>Upload a CSV of transaction data. Two models — a supervised Random Forest and an unsupervised Isolation Forest — score every row for fraud risk in parallel on Databricks.</p>
  </div>

  <!-- Upload -->
  <div class="card">
    <div class="section-label"><span>01</span> Upload Transaction Data</div>
    <div id="dz">
      <input type="file" id="fi" accept=".csv"/>
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
      <p>Drop your <b>.csv file</b> here &nbsp;·&nbsp; or <b>click to browse</b></p>
      <small>Max 100 MB &nbsp;·&nbsp; UTF-8 encoded</small>
    </div>
    <div id="fn"></div>

    <div class="section-label" style="margin-top:26px"><span>02</span> Select Models to Run</div>
    <div class="models">
      <label class="mtog on" id="t-rf">
        <input type="checkbox" id="c-rf" checked/>
        <div>
          <div class="mt-name">Random Forest</div>
          <div class="mt-desc">Supervised ensemble trained on labelled fraud data. High precision on known fraud patterns. Outputs per-row fraud probability.</div>
        </div>
        <span class="mt-pill sup">Supervised</span>
      </label>
      <label class="mtog on" id="t-iso">
        <input type="checkbox" id="c-iso" checked/>
        <div>
          <div class="mt-name">Isolation Forest</div>
          <div class="mt-desc">Unsupervised anomaly detector. No labels needed — flags statistically unusual transactions based on feature isolation depth.</div>
        </div>
        <span class="mt-pill unsup">Unsupervised</span>
      </label>
    </div>

    <button id="run" disabled>
      <div class="spin" id="spin"></div>
      <span id="rtext">▶ &nbsp; Run Analysis</span>
    </button>
  </div>

  <!-- Progress -->
  <div id="prog">
    <div class="ps" id="s-up"><span class="pi">📤</span><span class="pt">Uploading CSV to Databricks</span><span class="ptag t-q" id="st-up">QUEUED</span></div>
    <div class="ps" id="s-rf"><span class="pi">🌲</span><span class="pt">Supervised Model &mdash; Random Forest</span><span class="ptag t-q" id="st-rf">QUEUED</span></div>
    <div class="ps" id="s-iso"><span class="pi">🔍</span><span class="pt">Unsupervised Model &mdash; Isolation Forest</span><span class="ptag t-q" id="st-iso">QUEUED</span></div>
  </div>

  <!-- Results -->
  <div id="res">
    <div class="stat-grid" id="sg"></div>
    <div class="section-label"><span>MODEL</span> Side-by-Side Comparison</div>
    <div class="cmp" id="cmp" style="margin-bottom:24px"></div>
    <div class="card">
      <div class="tbl-hdr">
        <h3>Scored Transactions</h3>
        <div class="tc">
          <button class="fb on" id="fa" onclick="filt('all')">All</button>
          <button class="fb" id="fh" onclick="filt('high')">High Risk</button>
          <button class="fb" id="fm" onclick="filt('med')">Medium</button>
          <button class="fb" id="fl" onclick="filt('low')">Low Risk</button>
          <button id="dl" onclick="dlcsv()">↓ Export CSV</button>
        </div>
      </div>
      <div class="twrap"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div>
    </div>
  </div>
</main>

<footer>FraudLens &nbsp;·&nbsp; Databricks Jobs API &nbsp;·&nbsp; Random Forest &amp; Isolation Forest &nbsp;·&nbsp; 2024</footer>
</div>

<script>
let tdata=[], cfilter='all';

// ── File ────────────────────────────────────────────────────────────────────
const dz=document.getElementById('dz'), fi=document.getElementById('fi');
const fn=document.getElementById('fn'), run=document.getElementById('run');
let selFile=null;

function setFile(f){
  if(!f||!f.name.toLowerCase().endsWith('.csv')){alert('Please select a .csv file');return}
  selFile=f;
  fn.textContent=`✓  ${f.name}  (${(f.size/1024).toFixed(1)} KB)`;
  fn.style.display='block'; run.disabled=false;
}
fi.addEventListener('change',e=>setFile(e.target.files[0]));
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');setFile(e.dataTransfer.files[0])});

['rf','iso'].forEach(id=>{
  const cb=document.getElementById(`c-${id}`), tg=document.getElementById(`t-${id}`);
  cb.addEventListener('change',()=>tg.classList.toggle('on',cb.checked));
});

// ── Step state ──────────────────────────────────────────────────────────────
function step(id,s){
  const el=document.getElementById(`s-${id}`), tg=document.getElementById(`st-${id}`);
  el.className='ps '+(s==='run'?'run':s==='ok'?'ok':s==='err'?'err':'');
  const m={q:['t-q','QUEUED'],run:['t-r','RUNNING'],ok:['t-ok','DONE'],err:['t-e','ERROR']};
  const [cls,txt]=m[s]||m.q;
  tg.className=`ptag ${cls}`; tg.textContent=txt;
}

// ── Run ─────────────────────────────────────────────────────────────────────
run.addEventListener('click',async()=>{
  if(!selFile)return;
  const mods=[];
  if(document.getElementById('c-rf').checked) mods.push('rf');
  if(document.getElementById('c-iso').checked) mods.push('iso');
  if(!mods.length){alert('Select at least one model');return}

  run.disabled=true;
  document.getElementById('spin').style.display='block';
  document.getElementById('rtext').textContent='Analysing…';
  document.getElementById('prog').style.display='block';
  document.getElementById('res').style.display='none';

  step('up','run'); step('rf',mods.includes('rf')?'run':'q'); step('iso',mods.includes('iso')?'run':'q');

  const fd=new FormData(); fd.append('file',selFile);
  mods.forEach(m=>fd.append('models',m));

  try{
    step('up','ok');
    const r=await fetch('/analyze',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error)throw new Error(d.error);
    step('rf', mods.includes('rf')  ? (d.results?.random_forest?.error  ?'err':'ok') : 'q');
    step('iso',mods.includes('iso') ? (d.results?.isolation_forest?.error?'err':'ok') : 'q');
    render(d);
  }catch(e){
    step('up','err');
    if(mods.includes('rf'))  step('rf','err');
    if(mods.includes('iso')) step('iso','err');
    alert('Error: '+e.message);
  }finally{
    run.disabled=false;
    document.getElementById('spin').style.display='none';
    document.getElementById('rtext').textContent='▶  Run Analysis';
  }
});

// ── Render ───────────────────────────────────────────────────────────────────
function render(d){
  const rf=d.results?.random_forest||{}, iso=d.results?.isolation_forest||{};
  const rff=rf.flagged_count??rf.n_fraud??'—';
  const isf=iso.flagged_count??iso.n_fraud??'—';
  const rfa=rf.accuracy!=null?`${(rf.accuracy*100).toFixed(1)}%`:'—';
  const rfr=rf.recall!=null?`${(rf.recall*100).toFixed(1)}%`:'—';

  document.getElementById('sg').innerHTML=`
    <div class="sc r"><div class="sl">Flagged · RF</div><div class="sv r">${rff}</div><div class="ss">of ${d.total_rows} rows</div></div>
    <div class="sc a"><div class="sl">Flagged · ISO</div><div class="sv a">${isf}</div><div class="ss">of ${d.total_rows} rows</div></div>
    <div class="sc b"><div class="sl">Total Scanned</div><div class="sv b">${d.total_rows.toLocaleString()}</div><div class="ss">transactions</div></div>
    <div class="sc g"><div class="sl">RF Accuracy</div><div class="sv g">${rfa}</div><div class="ss">validation set</div></div>`;

  function panel(title,res,pill,pillClass){
    if(res.error)return`<div class="cp"><div class="cph"><span class="cpn" style="color:var(--red)">${title}</span><span class="mt-pill ${pillClass}">${pill}</span></div><div class="cpb"><div class="err-txt">⚠ ${res.error}</div></div></div>`;
    const rows=[
      ['Flagged transactions',res.flagged_count??res.n_fraud??'—'],
      ['Fraud rate',res.fraud_rate!=null?`${(res.fraud_rate*100).toFixed(2)}%`:'—'],
      ['Accuracy',res.accuracy!=null?`${(res.accuracy*100).toFixed(1)}%`:'—'],
      ['Precision',res.precision!=null?`${(res.precision*100).toFixed(1)}%`:'—'],
      ['Recall',res.recall!=null?`${(res.recall*100).toFixed(1)}%`:'—'],
      ['F1 Score',res.f1!=null?res.f1.toFixed(3):'—'],
      ['Avg fraud prob',res.avg_fraud_prob!=null?`${(res.avg_fraud_prob*100).toFixed(1)}%`:'—'],
    ];
    return`<div class="cp"><div class="cph"><span class="cpn">${title}</span><span class="mt-pill ${pillClass}">${pill}</span></div><div class="cpb">${rows.map(([l,v])=>`<div class="ms"><span class="msl">${l}</span><span class="msv">${v}</span></div>`).join('')}</div></div>`;
  }
  document.getElementById('cmp').innerHTML=panel('Random Forest',rf,'Supervised','sup')+panel('Isolation Forest',iso,'Unsupervised','unsup');

  tdata=rf.predictions??iso.predictions??[];
  buildTable(tdata,'all');

  document.getElementById('res').style.display='block';
  document.getElementById('res').scrollIntoView({behavior:'smooth',block:'start'});
}

function risk(p){
  if(p==null)return{lbl:'—',cls:'',lvl:'?'};
  if(p>=.7)return{lbl:`HIGH ${(p*100).toFixed(0)}%`,cls:'rh',lvl:'high'};
  if(p>=.4)return{lbl:`MED ${(p*100).toFixed(0)}%`,cls:'rm',lvl:'med'};
  return{lbl:`LOW ${(p*100).toFixed(0)}%`,cls:'rl',lvl:'low'};
}

function buildTable(rows,filter){
  if(!rows.length){
    document.getElementById('th').innerHTML='<tr><th>Column</th><th>RF Score</th><th>ISO Score</th><th>Decision</th></tr>';
    document.getElementById('tb').innerHTML='<tr><td colspan="4" style="text-align:center;color:var(--sub);padding:28px 14px">No row-level predictions returned.<br/><span style="font-size:.62rem">Ensure notebooks output a JSON field <code>"predictions"</code> with per-row scores.</span></td></tr>';
    return;
  }
  const skip=new Set(['rf_fraud_prob','iso_score','iso_anomaly','rf_prediction','iso_prediction','fraud_prob','anomaly_score']);
  const dcols=Object.keys(rows[0]).filter(k=>!skip.has(k)).slice(0,7);
  document.getElementById('th').innerHTML='<tr>'+dcols.map(c=>`<th>${c}</th>`).join('')+'<th>RF Score</th><th>ISO Score</th><th>Decision</th></tr>';

  const filtered=filter==='all'?rows:rows.filter(r=>{
    const p=r.rf_fraud_prob??r.fraud_prob??r.iso_score??null;
    return risk(p).lvl===filter;
  });

  document.getElementById('tb').innerHTML=filtered.map(r=>{
    const rp=r.rf_fraud_prob??r.fraud_prob??null;
    const ip=r.iso_score??r.anomaly_score??null;
    const rr=risk(rp), ir=risk(ip);
    const bw=rp!=null?Math.round(rp*72):0;
    const bc=rp>=.7?'var(--red)':rp>=.4?'var(--amber)':'var(--green)';
    const dec=rp>=.7?'<span style="color:var(--red);font-weight:500">⚑ FRAUD</span>':rp>=.4?'<span style="color:var(--amber)">? REVIEW</span>':'<span style="color:var(--sub)">✓ CLEAR</span>';
    return'<tr>'+dcols.map(c=>`<td>${r[c]??'—'}</td>`).join('')+
      `<td><span class="${rr.cls}">${rr.lbl}</span><span class="bar" style="width:${bw}px;background:${bc}"></span></td>`+
      `<td><span class="${ir.cls}">${ir.lbl}</span></td>`+
      `<td>${dec}</td></tr>`;
  }).join('')||'<tr><td colspan="20" style="text-align:center;color:var(--sub);padding:24px">No transactions match this filter.</td></tr>';
}

function filt(f){
  cfilter=f;
  ['all','high','med','low'].forEach(id=>document.getElementById('f'+id[0]).classList.toggle('on',id===f));
  buildTable(tdata,f);
}

function dlcsv(){
  if(!tdata.length)return;
  const h=Object.keys(tdata[0]);
  const csv=[h.join(','),...tdata.map(r=>h.map(k=>JSON.stringify(r[k]??'')).join(','))].join('\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='fraudlens_results.csv'; a.click();
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
