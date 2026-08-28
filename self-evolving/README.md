# Self-Evolving EHR Agent Harness

Optimize the **agent harness** (memory, planner overlays, retrieval budgets, prompt compression knobs) around a **frozen LLM**. No model weight updates, no fine-tuning, no gradient steps.

Built on top of `/home/czk/EhrAgent` Memory Agent (`--memory_agent`).

## Two LLM planners (weights frozen)

| Planner | When | Role |
|---------|------|------|
| **Inner Memory Planner** | every question | WorldMM + task memory → tool/table plan → coding agent |
| **Outer Meta-Planner** | every evolve generation | read FAIL/PASS traces → propose harness JSON edits |

```text
                    ┌─────────────────────────┐
   traces ────────►│ LLM Meta-Planner         │──► harness deltas
                    │ (knobs / overlays /      │    (JSON artifacts only)
                    │  pitfalls / skills)      │
                    └─────────────────────────┘
                                 │
                                 ▼
   question ──────► LLM Memory Planner ──► code executor ──► logs
                    (inner, still ON after evolve)
```

## What is the harness?

| Layer | Mutable? | Examples |
|-------|----------|----------|
| LLM backbone | **No** | `gpt-4o-mini` / local Qwen via API |
| Inner planner / executor scaffolding | **Yes** | overlays, TERMINATE / schema constraints |
| Task / skill memory | **Yes** | pitfalls, reusable skills, executable traces |
| Retrieval & compression knobs | **Yes** | `num_shots`, `compress_prompt`, skill budget |

Outer loop:

```text
harness_v → train eval → LLM meta-plan → apply artifacts
         → holdout → accept / reject → harness_v+1
```

## Layout

```text
self-evolving/
  ehr_harness/          # package
    spec.py             # HarnessSpec (JSON knobs)
    artifacts.py        # versioned artifact store
    adapter.py          # calls EhrAgent without editing weights
    traces.py           # parse per-question logs
    mutate.py           # heuristic fallback mutations
    meta_planner.py     # LLM meta-planner (outer)
    llm_client.py       # OpenAI-compatible chat
    evolve.py           # outer loop
    cli.py              # entrypoint
  configs/ehr_harness.yaml
  artifacts/            # generated harness state
  scripts/run_eval.sh
  scripts/run_evolve.sh
```

## Setup

Uses EhrAgent's env / venv (API key in `EhrAgent/ehragent/.env`).

```bash
export EHRAGENT_ROOT=/home/czk/EhrAgent
export EHRAGENT_DATA_ROOT=/home/czk/EhrAgent/ehrsql-ehragent
cd /home/czk/EnerVerse-AC/self-evolving
```

## Commands

```bash
# create artifact dir
python -m ehr_harness.cli --artifacts ./artifacts/ehr_default init --llm gpt-4o-mini

# smoke eval (8 questions)
bash scripts/run_eval.sh
# or:
NUM_QUESTIONS=8 bash scripts/run_eval.sh

# evolve harness for a few generations (LLM meta-planner ON by default)
GENS=3 TRAIN_N=12 HOLD_N=8 bash scripts/run_evolve.sh

# optional: separate model for meta-planning
python -m ehr_harness.cli --artifacts ./artifacts/ehr_default evolve \
  --llm gpt-4o-mini --meta_llm gpt-4o-mini --generations 3

# ablation: rule-based outer mutate only
python -m ehr_harness.cli ... evolve --heuristic_only

# inspect
python -m ehr_harness.cli --artifacts ./artifacts/ehr_default show
```

## Design constraints

1. **Never write model checkpoints / LoRA / optimizer state.**
2. Only JSON artifacts under `artifacts/` may change across generations.
3. EhrAgent remains the inner executor; this package is the outer harness controller.

## Relation to EhrAgent

This does **not** replace EhrAgent. It wraps it:

- seeds `logs/.task_memory.json` from harness artifacts
- injects constraint overlays into task-memory pitfalls (consumed by Memory Planner)
- tunes CLI knobs (`--num_shots`, `--compress_prompt`, `--planner_heuristic_only`, …)
- mines FAIL/PASS logs to grow skills & pitfalls, then keeps changes if holdout SR holds
