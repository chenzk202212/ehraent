#!/usr/bin/env bash
# Full MIMIC-III valid set (581 questions) with Memory Agent + heuristic planner.
# Run from a shell that already has OPENAI_API_KEY / OPENAI_BASE_URL (or .env in ehragent/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EHRAGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EHRAGENT_DIR}"

if [[ -z "${OPENAI_BASE_URL:-}" && -f "${EHRAGENT_DIR}/.aihubmix_env" ]]; then
  # Includes the proxy settings required by fox to reach AIHubMix.
  source "${EHRAGENT_DIR}/.aihubmix_env"
fi

PYTHON_BIN="${PYTHON_BIN:-${EHRAGENT_DIR}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
DATA_PATH="${EHRAGENT_DATA_ROOT}/mimic_iii/valid_preprocessed.json"
LOGS_PATH="${LOGS_PATH:-./logs_full_memory}"
NUM_SHOTS="${NUM_SHOTS:-4}"
LLM="${LLM:-${OPENAI_MODEL:-gpt-4o-mini}}"
START_ID="${START_ID:-0}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing ${DATA_PATH}; set EHRAGENT_DATA_ROOT." >&2
  exit 2
fi

mkdir -p "${LOGS_PATH}"
RUN_LOG="${LOGS_PATH}/run.log"

echo "EHRAGENT_DIR=${EHRAGENT_DIR}"
echo "DATA_PATH=${DATA_PATH}"
echo "LOGS_PATH=${LOGS_PATH}"
echo "LLM=${LLM}  NUM_QUESTIONS=${NUM_QUESTIONS}  START_ID=${START_ID}"
echo "Log: ${RUN_LOG}"

"${PYTHON_BIN}" main.py \
  --memory_agent \
  --planner_heuristic_only \
  --dataset mimic_iii \
  --data_path "${DATA_PATH}" \
  --logs_path "${LOGS_PATH}" \
  --num_questions "${NUM_QUESTIONS}" \
  --start_id "${START_ID}" \
  --num_shots "${NUM_SHOTS}" \
  --no_shuffle \
  --llm "${LLM}" \
  --memory_trace \
  2>&1 | tee -a "${RUN_LOG}"

echo ""
echo "Done. Per-question logs: ${LOGS_PATH}/${NUM_SHOTS}/<id>.txt"
echo "Aggregate metrics:"
echo "  ${PYTHON_BIN} report_table1_mimic.py --data_path \"${DATA_PATH}\" --logs_dir ${LOGS_PATH}/${NUM_SHOTS}"
