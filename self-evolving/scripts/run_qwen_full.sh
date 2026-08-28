#!/usr/bin/env bash
# Full local-Qwen pipeline: evolve harness → evaluate all 581 MIMIC-III valid questions.
# Does not update model weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8012/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"

LLM="${LLM:-qwen-local}"
ART="${ART:-$ROOT/artifacts/ehr_qwen_full}"
PORT="${QWEN_PORT:-8012}"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-32}"
HOLD_N="${HOLD_N:-16}"
TRAIN_START="${TRAIN_START:-0}"
HOLD_START="${HOLD_START:-300}"
SKIP_EVOLVE="${SKIP_EVOLVE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"   # -1 = full valid set
# NOTE: harness adapter converts (start_id, count) → EhrAgent end index:
#   holdout 300 + 16 → --start_id 300 --num_questions 316

if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}." >&2
  echo "  bash ${EHRAGENT_ROOT}/ehragent/scripts/start_local_qwen.sh" >&2
  exit 2
fi

mkdir -p "$ART"
cd "$ROOT"
LOG="$ART/full_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) full Qwen harness run ========"
echo "ART=$ART LLM=$LLM GENS=$GENS TRAIN_N=$TRAIN_N HOLD_N=$HOLD_N NUM_QUESTIONS=$NUM_QUESTIONS"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"

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
spec["num_shots"] = max(2, int(spec.get("num_shots") or 2))
if not spec.get("constraint_overlays"):
    spec["constraint_overlays"] = [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ]
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print("harness knobs:", {k: spec[k] for k in ("llm","num_shots","compress_prompt","planner_heuristic_only","no_worldmm_context","version")})
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
  python -m ehr_harness.cli --artifacts "$ART" show || true
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "-------- $(date -Is) full eval start (num_questions=$NUM_QUESTIONS) --------"
  python -m ehr_harness.cli --artifacts "$ART" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id 0 \
    --tag "qwen_full_eval" \
    --llm "$LLM"
  echo "-------- $(date -Is) full eval done --------"
fi

echo "======== $(date -Is) finished ========"
echo "Log: $LOG"
echo "Metrics: $ART/metrics.json"
echo "Best snapshot under: $ART/runs/"
