#!/usr/bin/env bash
# Failure-triggered harness: first try normal; on FAIL/unfinished, arm memory fixes and retry once.
# Goal: raise SR (accuracy), not just CR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8012/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"

LLM="${LLM:-qwen-local}"
ART="${ART:-$ROOT/artifacts/ehr_qwen_fail_harness}"
PORT="${QWEN_PORT:-8012}"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
TRAIN_START="${TRAIN_START:-0}"
HOLD_START="${HOLD_START:-300}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
START_ID="${START_ID:-0}"
EVAL_TAG="${EVAL_TAG:-qwen_fail_harness_eval}"
SKIP_EVOLVE="${SKIP_EVOLVE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# Tight budgets for local 8k Qwen
export EHRAGENT_TIGHT_CONTEXT="${EHRAGENT_TIGHT_CONTEXT:-1}"
export EHRAGENT_MAX_TOKENS="${EHRAGENT_MAX_TOKENS:-512}"
export EHRAGENT_MAX_TOOL_CHARS="${EHRAGENT_MAX_TOOL_CHARS:-1200}"
export EHRAGENT_MAX_INIT_CHARS="${EHRAGENT_MAX_INIT_CHARS:-11000}"
export EHRAGENT_MAX_AUTO_REPLY="${EHRAGENT_MAX_AUTO_REPLY:-7}"

if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}" >&2
  exit 2
fi

mkdir -p "$ART"
cd "$ROOT"
LOG="$ART/full_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) failure-triggered harness run ========"
echo "ART=$ART LLM=$LLM retry_on_fail=1"

if [[ ! -f "$ART/harness.json" ]]; then
  python -m ehr_harness.cli --artifacts "$ART" init --llm "$LLM"
fi

python - <<PY
import json
from pathlib import Path
p = Path(${ART@Q}) / "harness.json"
spec = json.loads(p.read_text())
spec["llm"] = ${LLM@Q}
spec["compress_prompt"] = True
spec["planner_heuristic_only"] = False
spec["no_worldmm_context"] = True
spec["retry_on_fail"] = True
# Keep enough shots for accuracy; do not start by starving few-shot.
spec["num_shots"] = max(3, int(spec.get("num_shots") or 3))
# Shorter LTM snippets for 8k local context.
spec["ltm_code_max_lines"] = min(int(spec.get("ltm_code_max_lines") or 16), 16)
rb = dict(spec.get("retrieval_budget") or {})
rb.setdefault("task_pitfalls", 3)
rb.setdefault("skills", 2)
rb.setdefault("executable_traces", 2)
rb.setdefault("recent_states", 2)
# Cap retrieval so task-memory growth cannot blow context.
spec["retrieval_budget"] = {
    "task_pitfalls": min(int(rb.get("task_pitfalls", 3) or 3), 4),
    "skills": min(int(rb.get("skills", 2) or 2), 3),
    "executable_traces": min(int(rb.get("executable_traces", 2) or 2), 2),
    "recent_states": min(int(rb.get("recent_states", 2) or 2), 2),
}
if not spec.get("constraint_overlays"):
    spec["constraint_overlays"] = [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ]
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print("harness:", {k: spec[k] for k in ("llm","num_shots","retry_on_fail","version","compress_prompt")})
PY

if [[ "$SKIP_EVOLVE" != "1" ]]; then
  echo "-------- $(date -Is) evolve start --------"
  python -m ehr_harness.cli --artifacts "$ART" evolve \
    --generations "$GENS" \
    --train_questions "$TRAIN_N" \
    --holdout_questions "$HOLD_N" \
    --train_start "$TRAIN_START" \
    --holdout_start "$HOLD_START" \
    --llm "$LLM" \
    --meta_llm "$LLM" \
    --accept_min_holdout_delta 0
  echo "-------- $(date -Is) evolve done --------"
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "-------- $(date -Is) full eval start --------"
  python -m ehr_harness.cli --artifacts "$ART" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "$EVAL_TAG" \
    --llm "$LLM"
  echo "-------- $(date -Is) full eval done --------"
fi

echo "======== $(date -Is) finished ========"
