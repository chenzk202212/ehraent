#!/usr/bin/env python3
"""Summarize 80-q ablation slice, including v2/v3 subsets from existing full evals."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/czk/EhrAgent/ehrsql-ehragent/mimic_iii/valid_preprocessed.json")
START = 400
N = 80
OUT = ROOT / "artifacts" / "ablation_80" / "summary.json"


def _slice_ids() -> list[str]:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    return [str(r["id"]) for r in rows[START : START + N]]


def _metrics(traces: list[dict], ids: set[str]) -> dict:
    sel = [t for t in traces if str(t.get("question_id")) in ids]
    n = len(sel)
    ok = sum(1 for t in sel if t.get("passed"))
    unfinished = sum(1 for t in sel if t.get("unfinished"))
    incorrect = sum(1 for t in sel if t.get("terminated") and not t.get("passed"))
    return {
        "total": n,
        "correct": ok,
        "incorrect": incorrect,
        "unfinished": unfinished,
        "sr": (100.0 * ok / n) if n else None,
        "cr": (100.0 * (ok + incorrect) / n) if n else None,
    }


def _load_traces(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ids = _slice_ids()
    idset = set(ids)
    rows = []

    for label, traces_path, kind in [
        (
            "v2_full_eval_subset",
            ROOT / "artifacts/ehr_qwen_fail_v2/runs/qwen_fail_v2_eval/traces.json",
            "subset_of_581",
        ),
        (
            "v3_full_eval_subset",
            ROOT / "artifacts/ehr_qwen_fail_v3/runs/qwen_fail_v3_eval/traces.json",
            "subset_of_581",
        ),
    ]:
        m = _metrics(_load_traces(traces_path), idset)
        m.update({"name": label, "kind": kind, "status": "done" if m["total"] else "missing"})
        rows.append(m)

    for name, title in [
        ("a0_baseline", "A0 baseline (none)"),
        ("a1_overlays", "A1 overlays only"),
        ("a2_experience", "A2 experience only"),
        ("a3_retry", "A3 retry only"),
        ("a4_full", "A4 full (overlays+exp+retry)"),
    ]:
        p = ROOT / "artifacts" / "ablation_80" / name / "runs" / "ablation80" / "metrics.json"
        if p.is_file():
            m = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "name": title,
                    "kind": "fresh_80",
                    "status": "done",
                    "total": m.get("total"),
                    "correct": m.get("correct"),
                    "incorrect": m.get("incorrect"),
                    "unfinished": m.get("unfinished"),
                    "sr": m.get("sr"),
                    "cr": m.get("cr"),
                }
            )
        else:
            rows.append(
                {
                    "name": title,
                    "kind": "fresh_80",
                    "status": "pending",
                    "total": N,
                    "correct": None,
                    "incorrect": None,
                    "unfinished": None,
                    "sr": None,
                    "cr": None,
                }
            )

    summary = {
        "slice": {"start_id": START, "num_questions": N, "ids": ids},
        "note": (
            "Fresh 80-q runs start with seed LTM only. "
            "v2/v3 subsets are the same IDs taken from the 581-q eval "
            "(those items had LTM accumulated from earlier questions)."
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
