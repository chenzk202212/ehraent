"""Resolve WorldMM MIMIC timeline JSON paths for EHRAgent benchmark rows."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def ensure_worldmm_on_path(worldmm_root: str) -> str:
    """Insert ``<WorldMM>/src`` on ``sys.path``. Returns the src path."""
    import sys

    root = os.path.abspath(worldmm_root)
    src = os.path.join(root, "src")
    if not os.path.isdir(src):
        raise FileNotFoundError(f"WorldMM src not found: {src}")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def resolve_timeline_path(row: Dict[str, Any], timeline_dir: Optional[str]) -> Optional[str]:
    """
    Find a patient timeline JSON for WorldMM ``EHRMWorldMemory.load_mimic_json``.

    Resolution order:
      1. ``row["timeline"]`` — absolute path, or basename under ``timeline_dir``
      2. ``{timeline_dir}/{id}.json``
      3. ``{timeline_dir}/_mimic_hadm_{hadm}.json`` using ``row["hadm_id"]`` or ``row["value"]["hadm_id"]``
    """
    tdir = (timeline_dir or "").strip()
    raw = row.get("timeline")
    if isinstance(raw, str) and raw.strip():
        p = raw.strip()
        if os.path.isabs(p) and os.path.isfile(p):
            return p
        if tdir:
            cand = os.path.join(tdir, os.path.basename(p))
            if os.path.isfile(cand):
                return cand
    if not tdir or not os.path.isdir(tdir):
        return None
    qid = row.get("id")
    if qid is not None:
        for name in (f"{qid}.json", f"_mimic_question_{qid}.json"):
            cand = os.path.join(tdir, name)
            if os.path.isfile(cand):
                return cand
    hadm = row.get("hadm_id")
    if hadm is None:
        v = row.get("value")
        if isinstance(v, dict):
            hadm = v.get("hadm_id") or v.get("HADM_ID")
    if hadm is not None:
        try:
            cand = os.path.join(tdir, f"_mimic_hadm_{int(hadm)}.json")
            if os.path.isfile(cand):
                return cand
        except (TypeError, ValueError):
            pass
    return None
