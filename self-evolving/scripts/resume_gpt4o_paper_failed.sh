#!/usr/bin/env bash
# Resume failed paper+harness after AIHubMix quota outage:
#   1) continue static from START_ID (default 390)
#   2) recompute static Table1 / metrics over all logs
#   3) re-run evolve + full harness eval (SKIP_STATIC=1)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/raid/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/raid/czk/EhrAgent/ehrsql-ehragent}"

AIHUB_ENV="${EHRAGENT_ROOT}/ehragent/.aihubmix_env"
# shellcheck disable=SC1090
source "$AIHUB_ENV"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
unset EHRAGENT_TIGHT_CONTEXT EHRAGENT_MAX_TOKENS EHRAGENT_MAX_TOOL_CHARS EHRAGENT_MAX_INIT_CHARS || true

LLM="${LLM:-gpt-4o}"
START_ID="${START_ID:-390}"
ART_STATIC="${ART_STATIC:-$ROOT/artifacts/ehr_gpt4o_paper_static}"
ART="${ART:-$ROOT/artifacts/ehr_gpt4o_paper_harness}"
STATIC_RUN="${STATIC_RUN:-$ART_STATIC/runs/gpt4o_paper_static_581}"
RESUME_LOG="${RESUME_LOG:-$ART/resume_failed.log}"
PYTHON_BIN="${PYTHON_BIN:-$EHRAGENT_ROOT/.venv/bin/python}"
DATA_PATH="${EHRAGENT_DATA_ROOT}/mimic_iii/valid_preprocessed.json"

mkdir -p "$ART" "$STATIC_RUN"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$RESUME_LOG") 2>&1

echo "======== $(date -Is) RESUME failed parts ========"
echo "LLM=$LLM START_ID=$START_ID PROXY=$HTTP_PROXY"

# Smoke
OPENAI_MODEL="$LLM" python3 - <<'PY'
import os
from openai import OpenAI
c=OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=30)
r=c.chat.completions.create(model=os.environ.get("OPENAI_MODEL","gpt-4o"),
  messages=[{"role":"user","content":"Reply OK"}], max_tokens=5)
print("smoke:", (r.choices[0].message.content or "").strip())
PY

# Ensure static harness knobs
python -m ehr_harness.cli --artifacts "$ART_STATIC" init --llm "$LLM" >/dev/null 2>&1 || true
RETRY_FLAG=0 ART_DIR="$ART_STATIC" HNAME=ehr_gpt4o_paper_static LLM_NAME="$LLM" python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["ART_DIR"]) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
spec.update({
    "name": os.environ["HNAME"], "llm": os.environ["LLM_NAME"], "num_shots": 4,
    "planner_heuristic_only": True, "compress_prompt": True, "memory_agent": True,
    "no_worldmm_context": False, "ltm_disable": False, "ltm_code_max_lines": 28,
    "max_consecutive_auto_reply": 10, "retry_on_fail": False,
    "retrieval_budget": {"task_pitfalls":5,"skills":3,"executable_traces":2,"recent_states":3},
    "constraint_overlays": [], "family_overlays": {},
    "notes": "paper Memory Agent static resume after quota",
})
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False)+"\n")
print("static harness ok", {k:spec[k] for k in ("llm","planner_heuristic_only","retry_on_fail")})
PY

echo "-------- $(date -Is) static resume start_id=$START_ID --------"
# Direct EhrAgent resume into same logs dir (keeps 0..START_ID-1 txt files)
"$PYTHON_BIN" "$EHRAGENT_ROOT/ehragent/main.py" \
  --llm "$LLM" \
  --num_shots 4 \
  --memory_agent \
  --planner_heuristic_only \
  --compress_prompt \
  --dataset mimic_iii \
  --data_path "$DATA_PATH" \
  --logs_path "$STATIC_RUN" \
  --num_questions -1 \
  --start_id "$START_ID" \
  --seed 42 \
  --no_shuffle \
  --memory_trace \
  --quiet \
  2>&1 | tee -a "$STATIC_RUN/resume_from_${START_ID}.log"

echo "-------- $(date -Is) recompute static table1/metrics --------"
"$PYTHON_BIN" "$EHRAGENT_ROOT/ehragent/report_table1_mimic.py" \
  --data_path "$DATA_PATH" \
  --logs_dir "$STATIC_RUN/4" \
  | tee "$ART_STATIC/table1_resume.txt"

python3 - <<PY
import json, re
from pathlib import Path
run = Path(${STATIC_RUN@Q})
logs = run / "4"
# Aggregate from per-id logs via report already printed; also write metrics.json compatible summary
# Prefer scanning run.log + resume log for judge lines if present
texts = []
for p in [run/"run.log", run/f"resume_from_${START_ID}.log"]:
    if p.is_file():
        texts.append(p.read_text(errors="replace"))
joined = "\n".join(texts)
# Fallback: count TERMINATE/judge from id files is hard; use report_table1 numbers from file if needed
# Parse last SR line from table1
t1 = Path(${ART_STATIC@Q}) / "table1_resume.txt"
sr = cr = None
correct = total = finished = None
if t1.is_file():
    for line in t1.read_text().splitlines():
        m = re.search(r"All:\s+SR=([0-9.]+)%\s+CR=([0-9.]+)%\s+\(correct=(\d+)/(\d+), finished=(\d+)/(\d+)\)", line)
        if m:
            sr, cr = float(m.group(1)), float(m.group(2))
            correct, total, finished = int(m.group(3)), int(m.group(4)), int(m.group(5))
metrics = {
    "total": total or 581,
    "correct": correct or 0,
    "incorrect": max(0, (finished or 0) - (correct or 0)),
    "unfinished": max(0, (total or 581) - (finished or 0)),
    "sr": sr if sr is not None else 0.0,
    "cr": cr if cr is not None else 0.0,
    "tag": "gpt4o_paper_static_581_resumed",
    "llm": ${LLM@Q},
    "start_id_resumed_from": int(${START_ID@Q}),
    "run_dir": str(run),
    "knobs": {"num_shots": 4, "compress_prompt": True, "planner_heuristic_only": True, "ltm_code_max_lines": 28},
}
Path(${ART_STATIC@Q}, "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
print("static metrics:", metrics)
PY

echo "-------- $(date -Is) rewrite harness + evolve + full eval --------"
rm -rf "$ART/runs/gpt4o_paper_harness_581" \
       "$ART/runs"/gen00_* "$ART/runs"/gen01_* "$ART/runs"/baseline_hold_* 2>/dev/null || true

RETRY_FLAG=1 ART_DIR="$ART" HNAME=ehr_gpt4o_paper_harness LLM_NAME="$LLM" python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["ART_DIR"]) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
spec.update({
    "name": os.environ["HNAME"], "llm": os.environ["LLM_NAME"], "version": 0,
    "num_shots": 4, "planner_heuristic_only": True, "compress_prompt": True,
    "memory_agent": True, "no_worldmm_context": False, "ltm_disable": False,
    "ltm_code_max_lines": 28, "max_consecutive_auto_reply": 10, "retry_on_fail": True,
    "retrieval_budget": {"task_pitfalls":5,"skills":3,"executable_traces":2,"recent_states":3},
    "constraint_overlays": [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ],
    "family_overlays": {},
    "notes": "paper Memory Agent + failure harness (resume after quota)",
})
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False)+"\n")
print("harness:", {k:spec[k] for k in ("llm","retry_on_fail","planner_heuristic_only","no_worldmm_context")})
PY

GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
TRAIN_START="${TRAIN_START:-0}"
HOLD_START="${HOLD_START:-300}"
ACCEPT_DELTA="${ACCEPT_DELTA:-1}"

echo "-------- $(date -Is) evolve --------"
python -m ehr_harness.cli --artifacts "$ART" evolve \
  --generations "$GENS" \
  --train_questions "$TRAIN_N" \
  --holdout_questions "$HOLD_N" \
  --train_start "$TRAIN_START" \
  --holdout_start "$HOLD_START" \
  --llm "$LLM" \
  --meta_llm "$LLM" \
  --no_inner_llm_planner \
  --accept_min_holdout_delta "$ACCEPT_DELTA"

echo "-------- $(date -Is) full harness eval --------"
python -m ehr_harness.cli --artifacts "$ART" eval \
  --num_questions -1 \
  --start_id 0 \
  --tag "gpt4o_paper_harness_581" \
  --llm "$LLM"

echo "======== $(date -Is) RESUME finished ========"
echo "static metrics: $ART_STATIC/metrics.json"
echo "static table1:  $ART_STATIC/table1_resume.txt"
echo "harness metrics:$ART/metrics.json"
