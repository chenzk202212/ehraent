#!/usr/bin/env bash
# Evaluate current harness on a MIMIC slice (no weight updates).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
ART="${ART:-$ROOT/artifacts/ehr_default}"
NUM_QUESTIONS="${NUM_QUESTIONS:-8}"
START_ID="${START_ID:-0}"
LLM="${LLM:-gpt-4o-mini}"

cd "$ROOT"
python -m ehr_harness.cli --artifacts "$ART" init --llm "$LLM" >/dev/null 2>&1 || true
python -m ehr_harness.cli --artifacts "$ART" eval \
  --num_questions "$NUM_QUESTIONS" \
  --start_id "$START_ID" \
  --tag "smoke_eval" \
  --llm "$LLM"
