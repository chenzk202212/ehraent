#!/usr/bin/env python3
"""Build isolated harness dirs for overlay / experience / retry ablations."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "ehr_qwen_fail_v3"
OUT = ROOT / "artifacts" / "ablation_80"

EMPTY_TM = {
    "version": 0,
    "pitfalls": {},
    "table_hints": {},
    "experiences": {},
    "skills": {},
    "active_states": [],
}


def _base_spec() -> dict:
    spec = json.loads((SRC / "harness.json").read_text(encoding="utf-8"))
    spec["llm"] = "qwen-local"
    spec["planner_heuristic_only"] = False
    spec["compress_prompt"] = True
    spec["memory_agent"] = True
    spec["no_worldmm_context"] = True
    spec["ltm_disable"] = False
    spec["num_shots"] = 4
    spec["ltm_code_max_lines"] = 24
    spec["family_overlays"] = {}
    return spec


def _experience_tm() -> dict:
    tm = json.loads((SRC / "task_memory.json").read_text(encoding="utf-8"))
    pitfalls = {}
    for k, v in (tm.get("pitfalls") or {}).items():
        if not isinstance(v, dict):
            continue
        err = str(v.get("error") or "")
        if err.startswith("[harness overlay]") or err.startswith("[family:"):
            continue
        pitfalls[k] = v
    tm["pitfalls"] = pitfalls
    for k in list(tm):
        if str(k).startswith("_harness"):
            tm.pop(k, None)
    tm["active_states"] = []
    return tm


def _write(name: str, spec: dict, tm: dict, notes: str) -> Path:
    dest = OUT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "runs").mkdir()
    spec = copy.deepcopy(spec)
    spec["notes"] = notes
    spec["name"] = f"ablation_{name}"
    spec["version"] = 0
    (dest / "harness.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (dest / "task_memory.json").write_text(
        json.dumps(tm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dest


def main() -> int:
    if not (SRC / "harness.json").is_file():
        print(f"missing source harness: {SRC}", file=sys.stderr)
        return 2
    spec = _base_spec()
    overlays = list(spec.get("constraint_overlays") or [])
    exp_tm = _experience_tm()

    a0 = copy.deepcopy(spec)
    a0["retry_on_fail"] = False
    a0["constraint_overlays"] = []
    _write("a0_baseline", a0, EMPTY_TM, "Ablation: memory agent only, no overlays/experience/retry")

    a1 = copy.deepcopy(spec)
    a1["retry_on_fail"] = False
    a1["constraint_overlays"] = overlays
    _write("a1_overlays", a1, EMPTY_TM, "Ablation: constraint overlays only")

    a2 = copy.deepcopy(spec)
    a2["retry_on_fail"] = False
    a2["constraint_overlays"] = []
    _write("a2_experience", a2, exp_tm, "Ablation: distilled skills/pitfalls/experiences only")

    a3 = copy.deepcopy(spec)
    a3["retry_on_fail"] = True
    a3["constraint_overlays"] = []
    _write("a3_retry", a3, EMPTY_TM, "Ablation: failure retry only")

    a4 = copy.deepcopy(spec)
    a4["retry_on_fail"] = True
    a4["constraint_overlays"] = overlays
    _write("a4_full", a4, exp_tm, "Ablation: overlays + experience + retry (v3 recipe, fresh LTM)")

    print(f"prepared ablations under {OUT}")
    for d in sorted(OUT.iterdir()):
        if d.is_dir():
            h = json.loads((d / "harness.json").read_text())
            tm = json.loads((d / "task_memory.json").read_text())
            print(
                f"  {d.name}: overlays={len(h.get('constraint_overlays') or [])} "
                f"retry={h.get('retry_on_fail')} "
                f"skills={len(tm.get('skills') or {})} "
                f"pitfalls={len(tm.get('pitfalls') or {})} "
                f"experiences={len(tm.get('experiences') or {})}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
