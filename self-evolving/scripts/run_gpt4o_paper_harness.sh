#!/usr/bin/env bash
# Paper \method (Memory Agent + WorldMM + heuristic planner) + failure-triggered harness.
# Base (paper): gpt-4o, memory_agent, planner_heuristic_only, WorldMM ON, shots=4
# Harness: retry_on_fail, outer evolve (PASS distill + meta), then full 581 eval
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/raid/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/raid/czk/EhrAgent/ehrsql-ehragent}"

AIHUB_ENV="${EHRAGENT_ROOT}/ehragent/.aihubmix_env"
if [[ -f "$AIHUB_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$AIHUB_ENV"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://aihubmix.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?need OPENAI_API_KEY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7890}"

# No local-8k clamps
unset EHRAGENT_TIGHT_CONTEXT EHRAGENT_MAX_TOKENS EHRAGENT_MAX_TOOL_CHARS EHRAGENT_MAX_INIT_CHARS || true
export EHRAGENT_MAX_AUTO_REPLY="${EHRAGENT_MAX_AUTO_REPLY:-10}"

LLM="${LLM:-gpt-4o}"
ART="${ART:-$ROOT/artifacts/ehr_gpt4o_paper_harness}"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
TRAIN_START="${TRAIN_START:-0}"
HOLD_START="${HOLD_START:-300}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
START_ID="${START_ID:-0}"
ACCEPT_DELTA="${ACCEPT_DELTA:-1}"
SKIP_EVOLVE="${SKIP_EVOLVE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_STATIC="${SKIP_STATIC:-0}"
ART_STATIC="${ART_STATIC:-$ROOT/artifacts/ehr_gpt4o_paper_static}"

mkdir -p "$ART"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG="$ART/full_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) paper+harness start ========"
echo "ART=$ART LLM=$LLM PROXY=$HTTP_PROXY WorldMM=ON heuristic=ON retry=ON"

smoke() {
  python3 - <<'PY'
import os
from openai import OpenAI
c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=30)
r = c.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
    messages=[{"role": "user", "content": "Reply OK"}],
    max_tokens=5,
)
print("smoke:", (r.choices[0].message.content or "").strip())
PY
}
OPENAI_MODEL="$LLM" smoke

write_paper_base() {
  local art="$1"
  local retry="$2"   # 0|1
  local name="$3"
  mkdir -p "$art"
  python -m ehr_harness.cli --artifacts "$art" init --llm "$LLM" >/dev/null 2>&1 || true
  RETRY_FLAG="$retry" ART_DIR="$art" HNAME="$name" LLM_NAME="$LLM" python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["ART_DIR"]) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
retry = os.environ["RETRY_FLAG"].strip() in ("1", "true", "True")
spec.update({
    "name": os.environ["HNAME"],
    "llm": os.environ["LLM_NAME"],
    "num_shots": 4,
    "planner_heuristic_only": True,   # paper Memory Agent
    "compress_prompt": True,
    "memory_agent": True,
    "no_worldmm_context": False,      # WorldMM ON
    "ltm_disable": False,
    "ltm_code_max_lines": 28,
    "max_consecutive_auto_reply": 10,
    "retry_on_fail": retry,
    "retrieval_budget": {
        "task_pitfalls": 5,
        "skills": 3,
        "executable_traces": 2,
        "recent_states": 3,
    },
    "family_overlays": {},
    "notes": (
        "paper Memory Agent + failure harness (WorldMM+heuristic+retry)"
        if retry else
        "paper Memory Agent static (WorldMM+heuristic, no retry)"
    ),
})
spec["constraint_overlays"] = (
    [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ]
    if retry
    else []
)
# hard assert paper knobs
assert spec["planner_heuristic_only"] is True
assert spec["no_worldmm_context"] is False
assert spec["retry_on_fail"] is retry
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
got = json.loads(p.read_text())
print("harness:", {k: got[k] for k in (
    "llm", "planner_heuristic_only", "no_worldmm_context", "retry_on_fail", "num_shots"
)})
assert got["planner_heuristic_only"] is True, got
PY
}

# Optional: static paper baseline (no retry / no evolve) for clean delta
if [[ "$SKIP_STATIC" != "1" ]]; then
  echo "-------- $(date -Is) paper STATIC (no harness retry) --------"
  write_paper_base "$ART_STATIC" "0" "ehr_gpt4o_paper_static"
  python -m ehr_harness.cli --artifacts "$ART_STATIC" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "gpt4o_paper_static_581" \
    --llm "$LLM"
  echo "-------- $(date -Is) paper STATIC done --------"
fi

write_paper_base "$ART" "1" "ehr_gpt4o_paper_harness"

if [[ "$SKIP_EVOLVE" != "1" ]]; then
  echo "-------- $(date -Is) evolve (keep heuristic planner) --------"
  python -m ehr_harness.cli --artifacts "$ART" evolve \
    --generations "$GENS" \
    --train_questions "$TRAIN_N" \
    --holdout_questions "$HOLD_N" \
    --train_start "$TRAIN_START" \
    --holdout_start "$HOLD_START" \
    --llm "$LLM" \
    --meta_llm "$LLM" \
    --no_inner_llm_planner \
    --accept_min_holdout_delta "$ACCEPT_DELTA"
  echo "-------- $(date -Is) evolve done --------"
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "-------- $(date -Is) full eval paper+harness --------"
  python -m ehr_harness.cli --artifacts "$ART" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "gpt4o_paper_harness_581" \
    --llm "$LLM"
  echo "-------- $(date -Is) full eval done --------"
fi

echo "======== $(date -Is) finished ========"
echo "static:  $ART_STATIC/metrics.json"
echo "harness: $ART/metrics.json"
