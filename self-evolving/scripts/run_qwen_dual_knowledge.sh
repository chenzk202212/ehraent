#!/usr/bin/env bash
# Qwen bilateral knowledge harness: PASS→skills + FAIL→structured diagnostics.
# Seeds from ehr_qwen_memory_static + ehr_qwen_fail_v3 traces, then evolve + full 581.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/raid/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/raid/czk/EhrAgent/ehrsql-ehragent}"
# Force local vLLM — do not inherit aihubmix/proxy BASE_URL from parent shell.
PORT="${QWEN_PORT:-8012}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export EHRAGENT_API_TYPE="openai"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

LLM="${LLM:-qwen-local}"
ART="${ART:-$ROOT/artifacts/ehr_qwen_dual_knowledge}"
SRC_STATIC="${SRC_STATIC:-$ROOT/artifacts/ehr_qwen_memory_static/runs/memory_static_581}"
SRC_V3="${SRC_V3:-$ROOT/artifacts/ehr_qwen_fail_v3}"
SRC_HARNESS="${SRC_HARNESS:-$SRC_V3/runs/qwen_fail_v3_eval}"
DATA_PATH="${EHRAGENT_DATA_ROOT}/mimic_iii/valid_preprocessed.json"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
ACCEPT_DELTA="${ACCEPT_DELTA:-1}"

# Local 8k budgets
export EHRAGENT_TIGHT_CONTEXT="${EHRAGENT_TIGHT_CONTEXT:-1}"
export EHRAGENT_MAX_TOKENS="${EHRAGENT_MAX_TOKENS:-512}"
export EHRAGENT_MAX_TOOL_CHARS="${EHRAGENT_MAX_TOOL_CHARS:-1200}"
export EHRAGENT_MAX_INIT_CHARS="${EHRAGENT_MAX_INIT_CHARS:-9000}"
export EHRAGENT_MAX_AUTO_REPLY="${EHRAGENT_MAX_AUTO_REPLY:-7}"

if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}" >&2
  exit 2
fi

mkdir -p "$ART"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG="$ART/full_run.log"

log() { echo "$@" | tee -a "$LOG"; }

log "======== $(date -Is) qwen dual-knowledge start ========"
log "ART=$ART LLM=$LLM BASE=$OPENAI_BASE_URL"
log "SRC_STATIC=$SRC_STATIC"
log "SRC_HARNESS=$SRC_HARNESS"

python -m ehr_harness.cli --artifacts "$ART" init --llm "$LLM" >/dev/null 2>&1 || true

LLM_NAME="$LLM" ART_DIR="$ART" python3 - <<'PY' | tee -a "$LOG"
import json, os
from pathlib import Path
p = Path(os.environ["ART_DIR"]) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
spec.update({
    "name": "ehr_qwen_dual_knowledge",
    "version": 0,
    "llm": os.environ["LLM_NAME"],
    "num_shots": 4,
    "planner_heuristic_only": False,
    "compress_prompt": True,
    "memory_agent": True,
    "no_worldmm_context": True,
    "ltm_disable": False,
    "ltm_code_max_lines": 16,
    "max_consecutive_auto_reply": 7,
    "retry_on_fail": True,
    "retrieval_budget": {"task_pitfalls": 3, "skills": 2, "executable_traces": 2, "recent_states": 2},
    "constraint_overlays": [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ],
    "family_overlays": {},
    "notes": "qwen bilateral knowledge: PASS how-to + FAIL diagnostics",
})
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print("harness:", {k: spec[k] for k in ("llm", "retry_on_fail", "no_worldmm_context", "num_shots")})
PY

SRC_STATIC="$SRC_STATIC" SRC_HARNESS="$SRC_HARNESS" DATA_PATH="$DATA_PATH" ART_DIR="$ART" python3 - <<'PY' | tee -a "$LOG"
import json, os
from pathlib import Path
from ehr_harness.traces import collect_traces
from ehr_harness.mutate import distill_knowledge_from_traces

art = Path(os.environ["ART_DIR"])
tm_path = art / "task_memory.json"
tm = json.loads(tm_path.read_text()) if tm_path.is_file() else {
    "skills": {}, "experiences": {}, "pitfalls": {}, "fail_knowledge": {}
}
data = os.environ["DATA_PATH"]
for label, root in [("static", os.environ.get("SRC_STATIC") or ""), ("v3", os.environ.get("SRC_HARNESS") or "")]:
    if not root:
        print("skip %s: empty path" % label)
        continue
    root_p = Path(root)
    logs = root_p / "4"
    run_log = root_p / "run.log"
    if not logs.is_dir():
        print("skip %s: no %s" % (label, logs))
        continue
    traces = collect_traces(logs, data_path=data, run_log=run_log if run_log.is_file() else None)
    tm, notes = distill_knowledge_from_traces(tm, traces, max_new_skills=48, max_new_fails=48)
    print(
        "seeded %s: traces=%d notes=%d skills=%d fail_kb=%d"
        % (label, len(traces), len(notes), len(tm.get("skills") or {}), len(tm.get("fail_knowledge") or {}))
    )
tm_path.write_text(json.dumps(tm, indent=2, ensure_ascii=False) + "\n")
PY

log "-------- $(date -Is) evolve --------"
python -m ehr_harness.cli --artifacts "$ART" evolve \
  --generations "$GENS" \
  --train_questions "$TRAIN_N" \
  --holdout_questions "$HOLD_N" \
  --train_start 0 \
  --holdout_start 300 \
  --llm "$LLM" \
  --meta_llm "$LLM" \
  --accept_min_holdout_delta "$ACCEPT_DELTA" \
  2>&1 | tee -a "$LOG"

log "-------- $(date -Is) full eval --------"
python -m ehr_harness.cli --artifacts "$ART" eval \
  --num_questions -1 \
  --start_id 0 \
  --tag "qwen_dual_knowledge_581" \
  --llm "$LLM" \
  2>&1 | tee -a "$LOG"

log "======== $(date -Is) qwen dual-knowledge finished ========"
