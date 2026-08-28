# EHR Agent + Self-Evolving Harness

Bundled repo for running EHRAgent benchmarks and the self-evolving harness experiments.

## Contents

| Path | Description |
|------|-------------|
| `EhrAgent/` | Memory Agent, WorldMM bridge, MIMIC-III tooling |
| `EhrAgent/ehrsql-ehragent/` | MIMIC-III CSVs + `valid_preprocessed.json` (581 val questions) |
| `self-evolving/` | Outer harness loop (`ehr_harness/`, run scripts) |

**Not included:** `.venv`, run logs, `artifacts/`, API keys (see `env.example`).

## Quick start

```bash
# 1. Clone and enter repo
git clone https://github.com/chenzk202212/ehraent.git
cd ehraent

# 2. Python env (EhrAgent)
python3 -m venv EhrAgent/.venv
source EhrAgent/.venv/bin/activate
pip install -r EhrAgent/requirements.txt
pip install -r self-evolving/requirements.txt

# 3. API / paths
cp env.example EhrAgent/ehragent/.env   # edit with your keys
source env.example                    # sets EHRAGENT_ROOT / DATA_ROOT

# 4. Smoke test (local Qwen on :8012, or set GPT in .env)
cd self-evolving
bash scripts/run_qwen_smoke.sh

# 5. Full eval (example)
bash scripts/run_eval.sh
```

## Environment variables

```bash
export EHRAGENT_ROOT="$(pwd)/EhrAgent"
export EHRAGENT_DATA_ROOT="$(pwd)/EhrAgent/ehrsql-ehragent"
```

Benchmark file used by harness:  
`$EHRAGENT_DATA_ROOT/mimic_iii/valid_preprocessed.json`

## Key scripts (`self-evolving/scripts/`)

| Script | Purpose |
|--------|---------|
| `run_eval.sh` | Smoke / short eval |
| `run_qwen_smoke.sh` | Local Qwen sanity check |
| `run_qwen_dual_knowledge.sh` | Qwen bilateral knowledge harness |
| `run_gpt4o_paper_harness.sh` | Paper-style GPT-4o + harness |
| `run_gpt4o_dual_knowledge.sh` | GPT dual-knowledge harness |
| `resume_gpt4o_paper_failed.sh` | Resume after API failures |

## Dataset

`EhrAgent/ehrsql-ehragent/mimic_iii/` contains MIMIC-III tables (CSV), SQLite DB, and validation JSON.  
Derived from the EHRSQL / EhrAgent benchmark layout.

## Upstream

- EhrAgent base: [wshi83/EhrAgent](https://github.com/wshi83/EhrAgent)
- Local modifications: memory agent, failure harness, WorldMM integration

## License

Respect MIMIC-III PhysioNet terms for the clinical data; code follows upstream licenses where applicable.
