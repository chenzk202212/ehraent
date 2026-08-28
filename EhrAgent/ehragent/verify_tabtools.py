#!/usr/bin/env python3
"""Smoke test: EhrAgent ``tools.tabtools`` + ``EHRAGENT_DATA_ROOT`` (no PyPI ``tools`` package)."""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools import tabtools as t  # noqa: E402

if __name__ == "__main__":
    root = os.environ.get("EHRAGENT_DATA_ROOT", "").strip()
    if not root:
        print("Set EHRAGENT_DATA_ROOT to your ehrsql-ehragent root, then re-run.", file=sys.stderr)
        sys.exit(2)
    d = t.db_loader("d_icd_diagnoses")
    print("LoadDB OK:", len(d), "rows;", "first columns:", d.columns[:3].tolist())
    r = t.sql_interpreter("SELECT COUNT(*) FROM D_ICD_DIAGNOSES")
    print("SQLInterpreter OK:", r)
