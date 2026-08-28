#!/usr/bin/env bash
# Full 581-q eval of PASS-only harness (A2 winner: experience, no overlays, no retry).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8012/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

LLM="${LLM:-qwen-local}"
ART="${ART:-$ROOT/artifacts/ehr_qwen_pass_only}"
PORT="${QWEN_PORT:-8012}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
START_ID="${START_ID:-0}"
EVAL_TAG="${EVAL_TAG:-pass_only_581}"

export EHRAGENT_TIGHT_CONTEXT="${EHRAGENT_TIGHT_CONTEXT:-1}"
export EHRAGENT_MAX_TOKENS="${EHRAGENT_MAX_TOKENS:-512}"
export EHRAGENT_MAX_TOOL_CHARS="${EHRAGENT_MAX_TOOL_CHARS:-1200}"
export EHRAGENT_MAX_INIT_CHARS="${EHRAGENT_MAX_INIT_CHARS:-9000}"
export EHRAGENT_MAX_AUTO_REPLY="${EHRAGENT_MAX_AUTO_REPLY:-7}"

if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}" >&2
  exit 2
fi

cd "$ROOT"
LOG="$ART/full_run.log"
{
  echo "======== $(date -Is) PASS-only 581 eval ========"
  echo "ART=$ART LLM=$LLM OPENAI_BASE_URL=$OPENAI_BASE_URL retry=off overlays=0"
  python3 -m ehr_harness.cli --artifacts "$ART" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "$EVAL_TAG" \
    --llm "$LLM"
  echo "======== $(date -Is) finished ========"
} 2>&1 | tee -a "$LOG"
