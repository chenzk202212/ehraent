#!/usr/bin/env bash
# Outer harness evolution loop (scaffold only — never writes model weights).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
ART="${ART:-$ROOT/artifacts/ehr_default}"
LLM="${LLM:-gpt-4o-mini}"
GENS="${GENS:-3}"
TRAIN_N="${TRAIN_N:-12}"
HOLD_N="${HOLD_N:-8}"

META_LLM="${META_LLM:-$LLM}"

cd "$ROOT"
python -m ehr_harness.cli --artifacts "$ART" init --llm "$LLM"
python -m ehr_harness.cli --artifacts "$ART" evolve \
  --generations "$GENS" \
  --train_questions "$TRAIN_N" \
  --holdout_questions "$HOLD_N" \
  --train_start 0 \
  --holdout_start 200 \
  --llm "$LLM" \
  --meta_llm "$META_LLM"
