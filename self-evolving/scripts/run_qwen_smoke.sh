#!/usr/bin/env bash
# Smoke / trial run of self-evolving harness against local vLLM Qwen.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"

# Local Qwen (vLLM) — must override any cloud keys in ehragent/.env
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8012/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"

LLM="${LLM:-qwen-local}"
ART="${ART:-$ROOT/artifacts/ehr_qwen}"
NUM_QUESTIONS="${NUM_QUESTIONS:-3}"
START_ID="${START_ID:-0}"
MODE="${MODE:-eval}"   # eval | evolve
GENS="${GENS:-1}"
TRAIN_N="${TRAIN_N:-4}"
HOLD_N="${HOLD_N:-2}"

PORT="${QWEN_PORT:-8012}"
if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}. Start with:" >&2
  echo "  bash ${EHRAGENT_ROOT}/ehragent/scripts/start_local_qwen.sh" >&2
  exit 2
fi

cd "$ROOT"
python -m ehr_harness.cli --artifacts "$ART" init --llm "$LLM"

# Prefer lighter smoke: compress prompt + keep inner LLM planner; skip WorldMM context
python - <<PY
import json
from pathlib import Path
p = Path(${ART@Q}) / "harness.json"
spec = json.loads(p.read_text())
spec["llm"] = ${LLM@Q}
spec["compress_prompt"] = True
spec["planner_heuristic_only"] = False
spec["no_worldmm_context"] = True
spec["num_shots"] = 2
p.write_text(json.dumps(spec, indent=2) + "\n")
print("harness ready:", p)
PY

echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "LLM=${LLM}  MODE=${MODE}  ART=${ART}"

if [[ "${MODE}" == "evolve" ]]; then
  python -m ehr_harness.cli --artifacts "$ART" evolve \
    --generations "$GENS" \
    --train_questions "$TRAIN_N" \
    --holdout_questions "$HOLD_N" \
    --train_start "$START_ID" \
    --holdout_start "$((START_ID + 50))" \
    --llm "$LLM" \
    --meta_llm "$LLM"
else
  python -m ehr_harness.cli --artifacts "$ART" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "qwen_smoke" \
    --llm "$LLM"
fi
