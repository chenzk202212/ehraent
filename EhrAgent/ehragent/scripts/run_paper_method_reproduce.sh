#!/usr/bin/env bash
# Reproduce paper \method (Memory Agent) on full MIMIC-III valid (581).
# Canonical flags from run_full_memory_agent.sh / run_ehr_experiments dynamic_memory:
#   --memory_agent --planner_heuristic_only --memory_trace --num_shots 4
#   WorldMM ON (do NOT pass --no_worldmm_context)
# Default LLM: gpt-4o (paper-level; override with LLM=gpt-4o-mini).
set -euo pipefail

EHRAGENT_ROOT="${EHRAGENT_ROOT:-/raid/czk/EhrAgent}"
EHR_DIR="${EHRAGENT_ROOT}/ehragent"
cd "$EHR_DIR"

if [[ -f "${EHR_DIR}/.aihubmix_env" ]]; then
  # shellcheck disable=SC1091
  source "${EHR_DIR}/.aihubmix_env"
fi

export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/raid/czk/EhrAgent/ehrsql-ehragent}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
# Do not inherit local-8k harness clamps
unset EHRAGENT_TIGHT_CONTEXT EHRAGENT_MAX_TOKENS EHRAGENT_MAX_TOOL_CHARS EHRAGENT_MAX_INIT_CHARS || true

LLM="${LLM:-gpt-4o}"
NUM_SHOTS="${NUM_SHOTS:-4}"
START_ID="${START_ID:-0}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
COMPRESS_PROMPT="${COMPRESS_PROMPT:-1}"
LOGS_PATH="${LOGS_PATH:-${EHRAGENT_ROOT}/runs_paper_method_gpt4o}"
DATA_PATH="${EHRAGENT_DATA_ROOT}/mimic_iii/valid_preprocessed.json"
PYTHON_BIN="${PYTHON_BIN:-${EHRAGENT_ROOT}/.venv/bin/python}"

mkdir -p "$LOGS_PATH"
RUN_LOG="${LOGS_PATH}/run.log"

echo "======== $(date -Is) paper \\method reproduce ========" | tee -a "$RUN_LOG"
echo "LLM=$LLM PROXY=${HTTP_PROXY:-} BASE=${OPENAI_BASE_URL:-}" | tee -a "$RUN_LOG"
echo "LOGS_PATH=$LOGS_PATH DATA=$DATA_PATH" | tee -a "$RUN_LOG"

# Smoke
"$PYTHON_BIN" - <<PY | tee -a "$RUN_LOG"
import os
from openai import OpenAI
c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=30)
r = c.chat.completions.create(model="${LLM}", messages=[{"role":"user","content":"Reply OK"}], max_tokens=5)
print("smoke:", (r.choices[0].message.content or "").strip())
PY

compress_args=()
if [[ "$COMPRESS_PROMPT" != "0" ]]; then
  compress_args=(--compress_prompt)
fi

"$PYTHON_BIN" main.py \
  --memory_agent \
  --planner_heuristic_only \
  --memory_trace \
  --dataset mimic_iii \
  --data_path "$DATA_PATH" \
  --logs_path "$LOGS_PATH" \
  --num_questions "$NUM_QUESTIONS" \
  --start_id "$START_ID" \
  --num_shots "$NUM_SHOTS" \
  --no_shuffle \
  --llm "$LLM" \
  --quiet \
  "${compress_args[@]}" \
  2>&1 | tee -a "$RUN_LOG"

echo "" | tee -a "$RUN_LOG"
echo "======== $(date -Is) finished; table1 ========" | tee -a "$RUN_LOG"
"$PYTHON_BIN" report_table1_mimic.py \
  --data_path "$DATA_PATH" \
  --logs_dir "${LOGS_PATH}/${NUM_SHOTS}" \
  2>&1 | tee -a "$RUN_LOG"
