#!/usr/bin/env python3
"""Aggregate EHRSQL-style EHRAgent logs for MIMIC-III or eICU experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any


DATASET_TABLES = {
    "mimic_iii": frozenset(
        {
            "admissions",
            "chartevents",
            "cost",
            "d_icd_diagnoses",
            "d_icd_procedures",
            "d_items",
            "d_labitems",
            "diagnoses_icd",
            "icustays",
            "inputevents_cv",
            "labevents",
            "microbiologyevents",
            "outputevents",
            "patients",
            "prescriptions",
            "procedures_icd",
            "transfers",
        }
    ),
    "eicu": frozenset(
        {
            "allergy",
            "cost",
            "diagnosis",
            "intakeoutput",
            "lab",
            "medication",
            "microlab",
            "patient",
            "treatment",
            "vitalperiodic",
        }
    ),
}


def judge(pred: str, ans: Any) -> bool:
    if isinstance(ans, list):
        ans_str = ", ".join(str(x) for x in ans)
    else:
        ans_str = str(ans)
    old_flag = ans_str in pred
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    for src, dst in (("False", "0"), ("false", "0"), ("True", "1"), ("true", "1"), ("No", "0"), ("no", "0"), ("Yes", "1"), ("yes", "1"), ("None", "0"), ("none", "0")):
        if ans == src:
            ans = dst
    if isinstance(ans, str) and ", " in ans:
        ans = ans.split(", ")
    if isinstance(ans, str) and len(ans) >= 2 and ans[-2:] == ".0":
        ans = ans[:-2]
    if not isinstance(ans, list):
        ans = [ans]
    return old_flag or all(str(x) in pred for x in ans)


def prediction_region(logs: str) -> str:
    marker = "***** Response from calling function (python) *****"
    i = logs.rfind(marker)
    if i >= 0:
        return logs[i:]
    te = logs.rfind("TERMINATE")
    if te >= 0:
        return logs[: te + len("TERMINATE")]
    return logs


def complexity_level(sql: str | None, dataset: str) -> int:
    if not sql:
        return 0
    tables = DATASET_TABLES.get(dataset, DATASET_TABLES["mimic_iii"])
    s = sql.lower()
    n = sum(1 for t in tables if re.search(r"\b" + re.escape(t) + r"\b", s))
    return min(n, 4) if n else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate EHRAgent EHRSQL logs.")
    ap.add_argument("--dataset", choices=sorted(DATASET_TABLES), required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--logs_dir", required=True, help="Directory containing <id>.txt logs, e.g. logs/4")
    args = ap.parse_args()

    with open(args.data_path, encoding="utf-8") as f:
        rows = json.load(f)

    totals = defaultdict(int)
    correct = defaultdict(int)
    finished = defaultdict(int)
    total_all = correct_all = finished_all = 0
    missing = []

    for row in rows:
        qid = row["id"]
        path = os.path.join(args.logs_dir, f"{qid}.txt")
        if not os.path.isfile(path):
            missing.append(qid)
            continue
        with open(path, encoding="utf-8") as f:
            logs = f.read()
        lv = complexity_level(row.get("query"), args.dataset)
        terminated = "TERMINATE" in logs
        ok = judge(prediction_region(logs), row.get("answer")) if terminated else False
        totals[lv] += 1
        total_all += 1
        if terminated:
            finished[lv] += 1
            finished_all += 1
            if ok:
                correct[lv] += 1
                correct_all += 1

    def pct(a: int, b: int) -> float:
        return 100.0 * a / b if b else 0.0

    print(f"{args.dataset} logs: {args.logs_dir}")
    for lv in (1, 2, 3, 4):
        print(
            f"  Level {lv}: SR={pct(correct[lv], totals[lv]):.2f}% "
            f"CR={pct(finished[lv], totals[lv]):.2f}% "
            f"(correct={correct[lv]}/{totals[lv]}, finished={finished[lv]}/{totals[lv]})"
        )
    print(
        f"  All:     SR={pct(correct_all, total_all):.2f}% "
        f"CR={pct(finished_all, total_all):.2f}% "
        f"(correct={correct_all}/{total_all}, finished={finished_all}/{total_all})"
    )
    if missing:
        print(f"  Missing logs: {len(missing)}; examples: {', '.join(missing[:5])}")


if __name__ == "__main__":
    main()
