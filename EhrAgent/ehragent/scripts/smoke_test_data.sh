#!/usr/bin/env bash
# Quick data-path check before running main.py (run from repo: bash ehragent/scripts/smoke_test_data.sh)
set -euo pipefail
ROOT="${EHRAGENT_DATA_ROOT:-/home/czk/EhrAgent/ehrsql-ehragent}"
export EHRAGENT_DATA_ROOT="$ROOT"
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
echo "EHRAGENT_DATA_ROOT=$EHRAGENT_DATA_ROOT"
"${PYTHON_BIN}" verify_tabtools.py
PYTHONPATH=.. "${PYTHON_BIN}" -c "
from tools import tabtools as t
r = t.sql_interpreter(\"SELECT COUNT(*) FROM D_ICD_DIAGNOSES\")
print('SQLInterpreter OK:', r)
"
