# Review status of the self-evolving EHR harness (updated after wiring fixes).

## Verdict
- Direction: **reasonable** (frozen LLM + mutable harness + outer LLM meta-planner + inner Memory Planner).
- Completeness: **usable MVP**, not research-complete. Critical wiring gaps below were partially fixed.

## Architecture (intended)
1. Inner loop: EhrAgent `--memory_agent` → LLM Memory Planner → code executor
2. Outer loop: train eval → LLM Meta-Planner → harness JSON/task_memory mutate → holdout → accept/reject
3. Never write model weights / LoRA / checkpoints

## What is solid
- Clear separation of frozen backbone vs mutable artifacts (`HarnessSpec`, `task_memory.json`)
- Outer LLM meta-planner (`meta_planner.py`) with heuristic fallback
- Inner LLM planner kept ON by default (`planner_heuristic_only=false`)
- Subprocess adapter reuses EhrAgent without copying the whole agent
- Accept/reject with task-memory rollback on reject
- Metrics prefer `judge=` lines from `run.log` when available

## Gaps fixed in this pass
1. **Overlays were dead**: injected only as pitfalls, which are retrieval-gated by token overlap → often never entered the Memory Plan.  
   Fix: EhrAgent `MedAgentMemoryAgent` loads `logs/harness.json` and **always** appends `constraint_overlays` to plan constraints; family overlays matched by keyword.
2. **`ltm_code_max_lines` / retrieval_budget** were written but ignored.  
   Fix: harness sidecar applies `ltm_code_max_lines` + retrieval budget into the agent/plan.
3. **SR/CR unreliable**: `"TERMINATE" in text` false-positives from few-shot prompts; soft judge on log tails.  
   Fix: parse `judge=PASS/FAIL` + `TERMINATE in log:` from `run.log` when present.
4. **Reject did not rollback** task memory after meta-plan.  
   Fix: snapshot parent TM/spec; restore on reject.
5. **accept_min_holdout_delta=-1** accepted regressions.  
   Fix: default `0.0`.

## Remaining incompleteness (known)
| Item | Severity | Notes |
|------|----------|-------|
| `configs/ehr_harness.yaml` not auto-loaded by CLI | low | docs-only; CLI uses `artifacts/` |
| `max_consecutive_auto_reply` not exposed in EhrAgent CLI | medium | harness field unused |
| WorldMM timeline dir not passed by adapter | medium | relies on EhrAgent defaults/cache |
| No complexity-stratified / Table-1 report in evolve | medium | use `report_table1_mimic.py` offline |
| Meta-planner does not see full chat / q_tag / gold SQL | medium | only short failure/success briefs |
| Online TM writes during holdout still happen before reject rollback | low | rolled back after holdout |
| No multi-seed / statistical significance | low | research polish |
| No unit/integration CI in this folder | low | smoke scripts only |

## Reasonableness checklist
- [x] Self-evolving without weight updates
- [x] LLM planning on outer loop
- [x] LLM planning on inner loop after evolve
- [x] Artifacts versioned under `artifacts/`
- [x] Overlays actually reach the executor prompt
- [~] Evaluation metrics aligned with EhrAgent judge (better; still prefer full table1 report for papers)
- [ ] Production-ready / paper-ready ablation suite

## Suggested next hardening (if continuing)
1. Adapter pass `--worldmm_timeline_dir` / ensure CSV timeline cache
2. Load YAML config in CLI `init/evolve`
3. Expose `max_consecutive_auto_reply` in EhrAgent `main.py`
4. Meta-planner input: attach q_tag + complexity + last tool error only (structured)
5. Holdout should optionally disable online TM writes (`ltm_disable` / read-only task memory) for cleaner accept tests
