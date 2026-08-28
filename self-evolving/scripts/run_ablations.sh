#!/usr/bin/env bash
# Sequential 80-question ablations on indices [400, 480).
# Shared vLLM: one condition at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
# Must override EhrAgent .env (AIHubMix). Same as run_qwen_fail_v2.sh.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8012/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$ROOT"

START_ID="${START_ID:-400}"
NUM_QUESTIONS="${NUM_QUESTIONS:-80}"
LLM="${LLM:-qwen-local}"
PORT="${QWEN_PORT:-8012}"
LOG="$ROOT/artifacts/ablation_80/run_ablations.log"

if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}" >&2
  exit 2
fi
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"

CONDS=(a0_baseline a1_overlays a2_experience a3_retry a4_full)

python3 "$ROOT/scripts/prepare_ablations.py"

{
  echo "=== ablation start $(date -Is) start_id=${START_ID} n=${NUM_QUESTIONS} llm=${LLM} ==="
  for c in "${CONDS[@]}"; do
    ART="$ROOT/artifacts/ablation_80/$c"
    MET="$ART/runs/ablation80/metrics.json"
    if [[ -f "$MET" ]]; then
      echo "[skip] $c already has $MET"
      continue
    fi
    echo "=== $c $(date -Is) ==="
    python3 -m ehr_harness.cli --artifacts "$ART" eval \
      --num_questions "$NUM_QUESTIONS" \
      --start_id "$START_ID" \
      --tag "ablation80" \
      --llm "$LLM"
    echo "=== $c done $(date -Is) ==="
  done
  python3 "$ROOT/scripts/summarize_ablations.py" || true
  echo "=== ablation all done $(date -Is) ==="
} 2>&1 | tee -a "$LOG"
