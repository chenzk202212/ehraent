#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${QWEN_MODEL_PATH:-/home/czk/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
VLLM="${VLLM_BIN:-/raid/czk/miniforge3/envs/cot_com/bin/vllm}"
PORT="${QWEN_PORT:-8012}"
GPU="${QWEN_GPU:-2}"
LOG_DIR="${ROOT}/logs_local_qwen"
mkdir -p "${LOG_DIR}"

if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "Local Qwen is already ready on port ${PORT}."
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" nohup "${VLLM}" serve "${MODEL}" \
  --served-model-name qwen-local \
  --host 127.0.0.1 --port "${PORT}" \
  --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.60 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  >"${LOG_DIR}/vllm.log" 2>&1 </dev/null &
echo "$!" >"${LOG_DIR}/vllm.pid"
echo "Started local Qwen PID $! on GPU ${GPU}; log: ${LOG_DIR}/vllm.log"
