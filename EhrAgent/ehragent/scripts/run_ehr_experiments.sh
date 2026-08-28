#!/usr/bin/env bash
# EHR-style experiment runner for EHRAgent variants.
# Runs baseline, dynamic memory agent, and memory ablations on MIMIC-III/eICU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EHRAGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EHRAGENT_DIR}"

export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-${EHRAGENT_DIR}/../ehrsql-ehragent}"

DATASET="${DATASET:-mimic_iii}"        # mimic_iii | eicu | both
LLM="${LLM:-gpt-4o-mini}"
NUM_QUESTIONS="${NUM_QUESTIONS:-20}"   # use -1 for full validation
START_ID="${START_ID:-0}"
NUM_SHOTS="${NUM_SHOTS:-4}"
RUN_ROOT="${RUN_ROOT:-./runs_ehr}"
COMPRESS_PROMPT="${COMPRESS_PROMPT:-1}"

run_one() {
  local dataset="$1"
  local variant="$2"
  local extra_args="$3"
  local data_path="${EHRAGENT_DATA_ROOT}/${dataset}/valid_preprocessed.json"
  local logs_path="${RUN_ROOT}/${dataset}/${variant}"
  local run_log="${logs_path}/run.log"

  if [[ ! -f "${data_path}" ]]; then
    echo "Missing ${data_path}; set EHRAGENT_DATA_ROOT." >&2
    exit 2
  fi

  mkdir -p "${logs_path}"
  echo ""
  echo "== ${dataset} / ${variant} =="
  echo "DATA_PATH=${data_path}"
  echo "LOGS_PATH=${logs_path}"
  echo "LLM=${LLM} NUM_QUESTIONS=${NUM_QUESTIONS} START_ID=${START_ID} NUM_SHOTS=${NUM_SHOTS}"

  local compress_args=""
  if [[ "${COMPRESS_PROMPT}" != "0" ]]; then
    compress_args="--compress_prompt"
  fi

  # shellcheck disable=SC2086
  python main.py \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --logs_path "${logs_path}" \
    --num_questions "${NUM_QUESTIONS}" \
    --start_id "${START_ID}" \
    --num_shots "${NUM_SHOTS}" \
    --no_shuffle \
    --llm "${LLM}" \
    --quiet \
    ${compress_args} \
    ${extra_args} \
    2>&1 | tee -a "${run_log}"

  python report_ehr_experiment.py \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --logs_dir "${logs_path}/${NUM_SHOTS}" \
    | tee -a "${run_log}"
}

run_dataset() {
  local dataset="$1"
  run_one "${dataset}" "baseline_ltm" ""
  run_one "${dataset}" "dynamic_memory" "--memory_agent --planner_heuristic_only --memory_trace"
  run_one "${dataset}" "dynamic_memory_no_growth" "--memory_agent --planner_heuristic_only --memory_trace --ltm_disable"
  run_one "${dataset}" "dynamic_memory_no_worldmm" "--memory_agent --planner_heuristic_only --memory_trace --no_worldmm_context"
}

python main.py --show_llm_endpoint --llm "${LLM}" >/dev/null

case "${DATASET}" in
  mimic_iii|eicu)
    run_dataset "${DATASET}"
    ;;
  both)
    run_dataset mimic_iii
    run_dataset eicu
    ;;
  *)
    echo "DATASET must be mimic_iii, eicu, or both; got ${DATASET}" >&2
    exit 2
    ;;
esac

echo ""
echo "Done. Results under ${RUN_ROOT}/"
