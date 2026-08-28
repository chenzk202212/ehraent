#!/usr/bin/env bash
# Bilateral knowledge harness: PASS→skills + FAIL→structured diagnostics.
# Seeds from existing paper_static / paper_harness traces, then evolve + full eval.
# Runs in parallel with the ongoing paper_harness_581 eval (separate ART).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/raid/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/raid/czk/EhrAgent/ehrsql-ehragent}"
# shellcheck disable=SC1090
source "${EHRAGENT_ROOT}/ehragent/.aihubmix_env"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
unset EHRAGENT_TIGHT_CONTEXT EHRAGENT_MAX_TOKENS EHRAGENT_MAX_TOOL_CHARS EHRAGENT_MAX_INIT_CHARS || true

LLM="${LLM:-gpt-4o}"
ART="${ART:-$ROOT/artifacts/ehr_gpt4o_dual_knowledge}"
SRC_STATIC="${SRC_STATIC:-$ROOT/artifacts/ehr_gpt4o_paper_static/runs/gpt4o_paper_static_581}"
SRC_HARNESS="${SRC_HARNESS:-$ROOT/artifacts/ehr_gpt4o_paper_harness/runs/gpt4o_paper_harness_581}"
DATA_PATH="${EHRAGENT_DATA_ROOT}/mimic_iii/valid_preprocessed.json"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
ACCEPT_DELTA="${ACCEPT_DELTA:-1}"

mkdir -p "$ART"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG="$ART/full_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) dual-knowledge harness start ========"
echo "ART=$ART LLM=$LLM PASS+FAIL distill ON"

python3 - <<'PY'
import os
from openai import OpenAI
c=OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=30)
r=c.chat.completions.create(model=os.environ.get("LLM","gpt-4o"), messages=[{"role":"user","content":"OK"}], max_tokens=3)
print("smoke:", (r.choices[0].message.content or "").strip())
PY

# Init harness (paper base + retry)
python -m ehr_harness.cli --artifacts "$ART" init --llm "$LLM" >/dev/null 2>&1 || true
LLM_NAME="$LLM" ART_DIR="$ART" python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["ART_DIR"]) / "harness.json"
spec = json.loads(p.read_text()) if p.is_file() else {}
spec.update({
    "name": "ehr_gpt4o_dual_knowledge",
    "version": 0,
    "llm": os.environ["LLM_NAME"],
    "num_shots": 4,
    "planner_heuristic_only": True,
    "compress_prompt": True,
    "memory_agent": True,
    "no_worldmm_context": False,
    "ltm_disable": False,
    "ltm_code_max_lines": 28,
    "max_consecutive_auto_reply": 10,
    "retry_on_fail": True,
    "retrieval_budget": {"task_pitfalls": 5, "skills": 3, "executable_traces": 2, "recent_states": 3},
    "constraint_overlays": [
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
    ],
    "family_overlays": {},
    "notes": "bilateral knowledge: PASS how-to + FAIL diagnostics",
})
p.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print("harness:", {k: spec[k] for k in ("llm", "retry_on_fail", "planner_heuristic_only")})
PY

# Seed task_memory from existing runs (static + partial harness)
SRC_STATIC="$SRC_STATIC" SRC_HARNESS="$SRC_HARNESS" DATA_PATH="$DATA_PATH" ART_DIR="$ART" python3 - <<'PY'
import json, os
from pathlib import Path
from ehr_harness.traces import collect_traces
from ehr_harness.mutate import distill_knowledge_from_traces

art = Path(os.environ["ART_DIR"])
tm_path = art / "task_memory.json"
tm = json.loads(tm_path.read_text()) if tm_path.is_file() else {"skills": {}, "experiences": {}, "pitfalls": {}, "fail_knowledge": {}}
data = os.environ["DATA_PATH"]
all_notes = []
for label, root in [("static", os.environ["SRC_STATIC"]), ("harness", os.environ["SRC_HARNESS"])]:
    root_p = Path(root)
    logs = root_p / "4"
    run_log = root_p / "run.log"
    if not logs.is_dir():
        print(f"skip {label}: no {logs}")
        continue
    traces = collect_traces(logs, data_path=data, run_log=run_log if run_log.is_file() else None)
    tm, notes = distill_knowledge_from_traces(tm, traces, max_new_skills=48, max_new_fails=48)
    all_notes.extend([f"{label}:{n}" for n in notes[:12]])
    print(f"seeded from {label}: traces={len(traces)} notes={len(notes)} "
          f"skills={len(tm.get('skills') or {})} fail_kb={len(tm.get('fail_knowledge') or {})} "
          f"pitfalls={len(tm.get('pitfalls') or {})}")
tm_path.write_text(json.dumps(tm, indent=2, ensure_ascii=False) + "\n")
print("sample notes:", all_notes[:10])
PY

echo "-------- $(date -Is) evolve --------"
python -m ehr_harness.cli --artifacts "$ART" evolve \
  --generations "$GENS" \
  --train_questions "$TRAIN_N" \
  --holdout_questions "$HOLD_N" \
  --train_start 0 \
  --holdout_start 300 \
  --llm "$LLM" \
  --meta_llm "$LLM" \
  --no_inner_llm_planner \
  --accept_min_holdout_delta "$ACCEPT_DELTA"

echo "-------- $(date -Is) full eval --------"
python -m ehr_harness.cli --artifacts "$ART" eval \
  --num_questions -1 \
  --start_id 0 \
  --tag "gpt4o_dual_knowledge_581" \
  --llm "$LLM"

echo "======== $(date -Is) dual-knowledge finished ========"
