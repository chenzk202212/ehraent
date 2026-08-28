#!/usr/bin/env bash
# GPT-4o-mini contrast vs Qwen main table:
#   1) static Memory Agent (no evolve, no fail-retry) — full 581
#   2) fail-auto (targeted repair + outer evolve) — same knobs as qwen fail_v3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/raid/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/raid/czk/EhrAgent/ehrsql-ehragent}"

# Load AIHubMix + working proxy (7890)
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

LLM="${LLM:-gpt-4o-mini}"
PROXY_PORT="${PROXY_PORT:-7890}"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
TRAIN_START="${TRAIN_START:-0}"
HOLD_START="${HOLD_START:-300}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
START_ID="${START_ID:-0}"
ACCEPT_DELTA="${ACCEPT_DELTA:-1}"
SKIP_STATIC="${SKIP_STATIC:-0}"
SKIP_AUTO="${SKIP_AUTO:-0}"
SKIP_EVOLVE="${SKIP_EVOLVE:-0}"

ART_STATIC="${ART_STATIC:-$ROOT/artifacts/ehr_gpt_memory_static}"
ART_AUTO="${ART_AUTO:-$ROOT/artifacts/ehr_gpt_fail_v3}"
MASTER_LOG="${MASTER_LOG:-$ROOT/artifacts/ehr_gpt_contrast/master.log}"
mkdir -p "$(dirname "$MASTER_LOG")" "$ART_STATIC" "$ART_AUTO"

# GPT has larger context; keep mild budgets (not the local-8k clamps)
export EHRAGENT_MAX_TOKENS="${EHRAGENT_MAX_TOKENS:-1024}"
export EHRAGENT_MAX_AUTO_REPLY="${EHRAGENT_MAX_AUTO_REPLY:-10}"
unset EHRAGENT_TIGHT_CONTEXT || true

smoke_api() {
  python3 - <<'PY'
import os, sys
from openai import OpenAI
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.setdefault(k, "http://127.0.0.1:7890")
c = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1"),
    timeout=30,
)
r = c.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "Reply OK"}],
    max_tokens=5,
)
print("smoke:", (r.choices[0].message.content or "").strip())
PY
}

echo "======== $(date -Is) GPT contrast start ========" | tee -a "$MASTER_LOG"
echo "LLM=$LLM PROXY=$HTTP_PROXY BASE=$OPENAI_BASE_URL" | tee -a "$MASTER_LOG"

if ! smoke_api 2>&1 | tee -a "$MASTER_LOG"; then
  echo "API smoke failed; abort" | tee -a "$MASTER_LOG"
  exit 2
fi

cd "$ROOT"

run_static() {
  local art="$ART_STATIC"
  local log="$art/full_run.log"
  echo "-------- $(date -Is) STATIC Memory Agent --------" | tee -a "$MASTER_LOG" "$log"
  python -m ehr_harness.cli --artifacts "$art" init --llm "$LLM" >/dev/null 2>&1 || true
  python - <<PY
import json
from pathlib import Path
p = Path(${art@Q}) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
spec.update({
    "name": "ehr_gpt_memory_static",
    "version": 0,
    "llm": ${LLM@Q},
    "num_shots": 4,
    "planner_heuristic_only": False,
    "compress_prompt": True,
    "memory_agent": True,
    "no_worldmm_context": True,
    "ltm_disable": False,
    "ltm_code_max_lines": 24,
    "max_consecutive_auto_reply": 10,
    "retry_on_fail": False,
    "retrieval_budget": {
        "task_pitfalls": 3,
        "skills": 2,
        "executable_traces": 2,
        "recent_states": 2,
    },
    "constraint_overlays": [],
    "family_overlays": {},
    "notes": "GPT static Memory Agent baseline; compare to ehr_qwen_memory_static",
})
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print("static harness:", {k: spec[k] for k in ("llm", "retry_on_fail", "num_shots")})
PY
  python -m ehr_harness.cli --artifacts "$art" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "gpt_memory_static_581" \
    --llm "$LLM" 2>&1 | tee -a "$log" "$MASTER_LOG"
  echo "-------- $(date -Is) STATIC done --------" | tee -a "$MASTER_LOG" "$log"
}

run_auto() {
  local art="$ART_AUTO"
  local log="$art/full_run.log"
  echo "-------- $(date -Is) FAIL-AUTO (v3-style) --------" | tee -a "$MASTER_LOG" "$log"
  python -m ehr_harness.cli --artifacts "$art" init --llm "$LLM" >/dev/null 2>&1 || true
  python - <<PY
import json
from pathlib import Path
p = Path(${art@Q}) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
spec.update({
    "name": "ehr_gpt_fail_v3",
    "llm": ${LLM@Q},
    "num_shots": max(3, int(spec.get("num_shots") or 4)),
    "planner_heuristic_only": False,
    "compress_prompt": True,
    "memory_agent": True,
    "no_worldmm_context": True,
    "ltm_disable": False,
    "ltm_code_max_lines": min(int(spec.get("ltm_code_max_lines") or 24), 24),
    "max_consecutive_auto_reply": 10,
    "retry_on_fail": True,
    "retrieval_budget": {
        "task_pitfalls": 3,
        "skills": 2,
        "executable_traces": 2,
        "recent_states": 2,
    },
    "notes": "GPT fail-auto contrast; same outer knobs as ehr_qwen_fail_v3",
})
if not spec.get("constraint_overlays"):
    spec["constraint_overlays"] = [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ]
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print("auto harness:", {k: spec[k] for k in ("llm", "retry_on_fail", "num_shots", "version")})
PY
  if [[ "$SKIP_EVOLVE" != "1" ]]; then
    python -m ehr_harness.cli --artifacts "$art" evolve \
      --generations "$GENS" \
      --train_questions "$TRAIN_N" \
      --holdout_questions "$HOLD_N" \
      --train_start "$TRAIN_START" \
      --holdout_start "$HOLD_START" \
      --llm "$LLM" \
      --meta_llm "$LLM" \
      --accept_min_holdout_delta "$ACCEPT_DELTA" 2>&1 | tee -a "$log" "$MASTER_LOG"
  fi
  python -m ehr_harness.cli --artifacts "$art" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "gpt_fail_v3_eval" \
    --llm "$LLM" 2>&1 | tee -a "$log" "$MASTER_LOG"
  echo "-------- $(date -Is) FAIL-AUTO done --------" | tee -a "$MASTER_LOG" "$log"
}

if [[ "$SKIP_STATIC" != "1" ]]; then
  run_static
fi
if [[ "$SKIP_AUTO" != "1" ]]; then
  run_auto
fi

echo "======== $(date -Is) GPT contrast finished ========" | tee -a "$MASTER_LOG"
echo "static metrics: $ART_STATIC/metrics.json"
echo "auto metrics:   $ART_AUTO/metrics.json"
