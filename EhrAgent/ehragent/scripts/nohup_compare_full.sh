#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/czk/EhrAgent"
EHR_DIR="${ROOT}/ehragent"
OUT="${ROOT}/runs_ehr_compare_full_v2.nohup.out"
PID_FILE="${ROOT}/runs_ehr_compare_full_v2.pid"

if [[ "${RUN_COMPARE_FULL_WORKER:-0}" != "1" ]]; then
  mkdir -p "${ROOT}"
  RUN_COMPARE_FULL_WORKER=1 nohup bash "$0" >"${OUT}" 2>&1 </dev/null &
  pid="$!"
  echo "${pid}" >"${PID_FILE}"
  echo "started pid=${pid}"
  echo "pid_file=${PID_FILE}"
  echo "log=${OUT}"
  exit 0
fi

source /home/czk/miniforge3/etc/profile.d/conda.sh
conda activate Ehr-czk
unset ALL_PROXY all_proxy

cd "${EHR_DIR}"
export EHRAGENT_DATA_ROOT="${ROOT}/ehrsql-ehragent"

DATA="${ROOT}/ehrsql-ehragent/mimic_iii/valid_preprocessed.json"
RUN_ROOT="./runs_ehr_compare_full_v2/mimic_iii"
COMMON_ARGS=(
  --dataset mimic_iii
  --data_path "${DATA}"
  --num_questions -1
  --start_id 0
  --num_shots 4
  --no_shuffle
  --llm gpt-4o-mini
  --quiet
)

report_one() {
  local variant="$1"
  python report_table1_mimic.py \
    --data_path "${DATA}" \
    --logs_dir "${RUN_ROOT}/${variant}/4"
}

echo "START $(date '+%F %T %Z')"

echo "ehragent_original full"
python main.py \
  "${COMMON_ARGS[@]}" \
  --logs_path "${RUN_ROOT}/ehragent_original"
report_one "ehragent_original"

echo "dynamic_memory full"
python main.py \
  "${COMMON_ARGS[@]}" \
  --logs_path "${RUN_ROOT}/dynamic_memory" \
  --compress_prompt \
  --memory_agent \
  --planner_heuristic_only \
  --memory_trace
report_one "dynamic_memory"

echo "dynamic_memory_no_growth full"
python main.py \
  "${COMMON_ARGS[@]}" \
  --logs_path "${RUN_ROOT}/dynamic_memory_no_growth" \
  --compress_prompt \
  --memory_agent \
  --planner_heuristic_only \
  --memory_trace \
  --ltm_disable
report_one "dynamic_memory_no_growth"

echo "dynamic_memory_no_worldmm full"
python main.py \
  "${COMMON_ARGS[@]}" \
  --logs_path "${RUN_ROOT}/dynamic_memory_no_worldmm" \
  --compress_prompt \
  --memory_agent \
  --planner_heuristic_only \
  --memory_trace \
  --no_worldmm_context
report_one "dynamic_memory_no_worldmm"

echo "DONE $(date '+%F %T %Z')"
