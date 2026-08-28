#!/usr/bin/env bash
# Outer-loop ablations + multi-seed probe on 80-q slice [400, 480).
# Conditions:
#   o0_no_evolve     : frozen harness (static), eval only
#   o1_heur_delta1   : evolve --heuristic_only, accept +1pp, then eval
#   o2_meta_delta0   : evolve with meta-LLM, accept +0pp, then eval
#   o3_meta_delta1   : evolve with meta-LLM, accept +1pp, then eval  (full outer)
# Multi-seed: eval v3 harness + static on seeds 42/43/44 (80-q).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EHRAGENT_ROOT="${EHRAGENT_ROOT:-/home/czk/EhrAgent}"
export EHRAGENT_DATA_ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8012/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EHRAGENT_API_TYPE="${EHRAGENT_API_TYPE:-openai}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

LLM="${LLM:-qwen-local}"
PORT="${QWEN_PORT:-8012}"
START_ID="${START_ID:-400}"
NUM_QUESTIONS="${NUM_QUESTIONS:-80}"
GENS="${GENS:-2}"
TRAIN_N="${TRAIN_N:-24}"
HOLD_N="${HOLD_N:-16}"
TRAIN_START="${TRAIN_START:-0}"
HOLD_START="${HOLD_START:-300}"
OUT="${OUT:-$ROOT/artifacts/outer_ablation_80}"
SRC_STATIC="${SRC_STATIC:-$ROOT/artifacts/ehr_qwen_memory_static}"
SRC_V3="${SRC_V3:-$ROOT/artifacts/ehr_qwen_fail_v3}"

export EHRAGENT_TIGHT_CONTEXT="${EHRAGENT_TIGHT_CONTEXT:-1}"
export EHRAGENT_MAX_TOKENS="${EHRAGENT_MAX_TOKENS:-512}"
export EHRAGENT_MAX_TOOL_CHARS="${EHRAGENT_MAX_TOOL_CHARS:-1200}"
export EHRAGENT_MAX_INIT_CHARS="${EHRAGENT_MAX_INIT_CHARS:-9000}"
export EHRAGENT_MAX_AUTO_REPLY="${EHRAGENT_MAX_AUTO_REPLY:-7}"

if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Local Qwen not ready on :${PORT}" >&2
  exit 2
fi

mkdir -p "$OUT"
cd "$ROOT"
LOG="$OUT/run_outer.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) outer ablation start ========"
echo "OUT=$OUT LLM=$LLM slice=${START_ID}+${NUM_QUESTIONS}"

seed_art() {
  local name="$1"
  local src="$2"
  local dest="$OUT/$name"
  mkdir -p "$dest/runs"
  cp "$src/harness.json" "$dest/harness.json"
  if [[ -f "$src/task_memory.json" ]]; then
    cp "$src/task_memory.json" "$dest/task_memory.json"
  else
    python3 - <<PY
import json
from pathlib import Path
p=Path("$dest")/"task_memory.json"
p.write_text(json.dumps({
  "version":0,"pitfalls":{},"table_hints":{},"experiences":{},"skills":{},"active_states":[]
}, indent=2)+"\n")
PY
  fi
  python3 - <<PY
import json
from pathlib import Path
p=Path("$dest")/"harness.json"
h=json.loads(p.read_text())
h["llm"]="$LLM"
h["memory_agent"]=True
h["compress_prompt"]=True
h["no_worldmm_context"]=True
h["planner_heuristic_only"]=False
h["notes"]=(h.get("notes") or "") + "\n[outer_ablation] $name"
p.write_text(json.dumps(h, indent=2, ensure_ascii=False)+"\n")
print("seeded", "$name", "v", h.get("version"), "retry", h.get("retry_on_fail"))
PY
}

run_eval() {
  local art="$1"
  local tag="$2"
  local seed="${3:-42}"
  local met="$art/runs/$tag/metrics.json"
  if [[ -f "$met" ]]; then
    echo "[skip eval] $tag exists"
    return 0
  fi
  echo "-------- $(date -Is) eval $tag seed=$seed --------"
  # Adapter uses seed via evolve/evaluate_only; pass through env for reproducibility of shuffle=false path
  python3 -m ehr_harness.cli --artifacts "$art" eval \
    --num_questions "$NUM_QUESTIONS" \
    --start_id "$START_ID" \
    --tag "$tag" \
    --llm "$LLM"
}

run_evolve() {
  local art="$1"
  local delta="$2"
  local heur="$3"  # 0 or 1
  local hist="$art/history.jsonl"
  if [[ -f "$hist" ]] && [[ -f "$art/metrics.json" ]]; then
    echo "[skip evolve] $art already has history"
    return 0
  fi
  echo "-------- $(date -Is) evolve art=$art delta=$delta heuristic=$heur --------"
  extra=()
  if [[ "$heur" == "1" ]]; then
    extra+=(--heuristic_only)
  fi
  python3 -m ehr_harness.cli --artifacts "$art" evolve \
    --generations "$GENS" \
    --train_questions "$TRAIN_N" \
    --holdout_questions "$HOLD_N" \
    --train_start "$TRAIN_START" \
    --holdout_start "$HOLD_START" \
    --llm "$LLM" \
    --meta_llm "$LLM" \
    --accept_min_holdout_delta "$delta" \
    "${extra[@]}"
}

# --- Outer conditions: start from static Memory Agent (clean) ---
# o0: no evolve
seed_art o0_no_evolve "$SRC_STATIC"
# force no retry for pure static outer-off
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/raid/czk/EnerVerse-AC/self-evolving/artifacts/outer_ablation_80/o0_no_evolve/harness.json")
h=json.loads(p.read_text()); h["retry_on_fail"]=False; h["constraint_overlays"]=[]
p.write_text(json.dumps(h, indent=2, ensure_ascii=False)+"\n")
PY
run_eval "$OUT/o0_no_evolve" "outer80_seed42"

# o1: heuristic outer only
seed_art o1_heur_delta1 "$SRC_STATIC"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/raid/czk/EnerVerse-AC/self-evolving/artifacts/outer_ablation_80/o1_heur_delta1/harness.json")
h=json.loads(p.read_text()); h["retry_on_fail"]=True
p.write_text(json.dumps(h, indent=2, ensure_ascii=False)+"\n")
PY
run_evolve "$OUT/o1_heur_delta1" 1 1
run_eval "$OUT/o1_heur_delta1" "outer80_seed42"

# o2: meta, weak accept
seed_art o2_meta_delta0 "$SRC_STATIC"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/raid/czk/EnerVerse-AC/self-evolving/artifacts/outer_ablation_80/o2_meta_delta0/harness.json")
h=json.loads(p.read_text()); h["retry_on_fail"]=True
p.write_text(json.dumps(h, indent=2, ensure_ascii=False)+"\n")
PY
run_evolve "$OUT/o2_meta_delta0" 0 0
run_eval "$OUT/o2_meta_delta0" "outer80_seed42"

# o3: meta, strict accept
seed_art o3_meta_delta1 "$SRC_STATIC"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/raid/czk/EnerVerse-AC/self-evolving/artifacts/outer_ablation_80/o3_meta_delta1/harness.json")
h=json.loads(p.read_text()); h["retry_on_fail"]=True
p.write_text(json.dumps(h, indent=2, ensure_ascii=False)+"\n")
PY
run_evolve "$OUT/o3_meta_delta1" 1 0
run_eval "$OUT/o3_meta_delta1" "outer80_seed42"

# Note: multi-seed with --no_shuffle + temperature=0 is nearly identical; skip for now.

python3 - <<'PY'
import json
from pathlib import Path
root=Path("/raid/czk/EnerVerse-AC/self-evolving/artifacts/outer_ablation_80")
rows=[]
for d in sorted(root.iterdir()):
  if not d.is_dir(): continue
  for m in d.glob("runs/*/metrics.json"):
    x=json.load(open(m))
    rows.append({
      "cond": d.name,
      "tag": m.parent.name,
      "sr": x.get("sr"),
      "cr": x.get("cr"),
      "correct": x.get("correct"),
      "total": x.get("total"),
      "harness_version": x.get("harness_version"),
    })
  hist=d/"history.jsonl"
  if hist.is_file():
    gens=[]
    for line in hist.read_text().splitlines():
      r=json.loads(line)
      if "generation" in r:
        gens.append({"g":r["generation"],"acc":r["accepted"],
                     "train":r["train"].get("sr"),"hold":r["holdout"].get("sr")})
    if gens:
      rows.append({"cond": d.name, "tag": "evolve_history", "gens": gens})
summary={"rows": rows}
(root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY

echo "======== $(date -Is) outer ablation finished ========"
