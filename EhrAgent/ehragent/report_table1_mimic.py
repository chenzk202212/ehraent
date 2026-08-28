#!/usr/bin/env python3
"""
Aggregate MIMIC-III validation logs into Table-1-style metrics (per complexity + All SR, CR).

Complexity I–IV follows the paper (Appendix A.2): by the number of distinct MIMIC tables
appearing in the gold SQL ``query`` (1→I, 2→II, 3→III, 4+→IV).

Usage (after ``main.py`` has written ``<logs_dir>/<id>.txt``)::

  python report_table1_mimic.py \\
    --data_path \"$EHRAGENT_DATA_ROOT/mimic_iii/valid_preprocessed.json\" \\
    --logs_dir ./logs/4

Match the paper's EHRAgent row setup: ``gpt-4-0613``, temperature 0, K=4, LTM on success
(``--num_shots 4``), full val (``--num_questions -1``). Paper Table 1 target (GPT-4):
SR: 71.58 / 66.34 / 49.70 / 49.14 / 58.97, CR: 85.86 (percent).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

# Same table names as ``tools/tabtools.db_loader`` (lowercase in SQL).
_MIMIC_TABLES = frozenset(
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
)


def _table_count_in_gold_sql(sql: str | None) -> int:
    if not sql:
        return 0
    s = sql.lower()
    found: set[str] = set()
    for t in _MIMIC_TABLES:
        if re.search(r"\b" + re.escape(t) + r"\b", s):
            found.add(t)
    return len(found)


def complexity_level(sql: str | None) -> int:
    """1–4 for I–IV; 0 if unknown."""
    n = _table_count_in_gold_sql(sql)
    if n <= 0:
        return 0
    return min(n, 4)


def judge(pred: str, ans) -> bool:
    if isinstance(ans, list):
        ans_str = ", ".join(str(x) for x in ans)
    else:
        ans_str = str(ans)
    old_flag = ans_str in pred
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if isinstance(ans, str) and ", " in ans:
        ans = ans.split(", ")
    if isinstance(ans, str) and len(ans) >= 2 and ans[-2:] == ".0":
        ans = ans[:-2]
    if not isinstance(ans, list):
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if str(ans[i]) not in pred:
            new_flag = False
            break
    return old_flag or new_flag


def _prediction_region_for_judge(logs_joined: str) -> str:
    """Match official EHRAgent main.py: take the region from the start of the
    last code cell (or 'Solution:' fallback) up to the last TERMINATE, so the
    Ground-Truth Answer line at the end of the log is excluded from judging."""
    if '"cell": "' in logs_joined:
        last_code_end = logs_joined.rfind('"\n}')
        if last_code_end < 0:
            last_code_end = logs_joined.rfind('"}')
    else:
        last_code_end = logs_joined.rfind("Solution:")
    te = logs_joined.rfind("TERMINATE")
    if last_code_end < 0:
        last_code_end = 0
    if te < 0:
        return logs_joined[last_code_end:]
    return logs_joined[last_code_end : te + len("TERMINATE")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Table-1-style SR/CR by complexity (MIMIC-III).")
    ap.add_argument("--data_path", required=True, help="valid_preprocessed.json")
    ap.add_argument("--logs_dir", required=True, help="e.g. ./logs/4")
    args = ap.parse_args()

    with open(args.data_path, encoding="utf-8") as f:
        rows = json.load(f)

    meta = {}
    for r in rows:
        qid = r["id"]
        meta[qid] = {
            "level": complexity_level(r.get("query")),
            "answer": r["answer"],
        }

    logs_dir = args.logs_dir.rstrip(os.sep)
    sep = "\n----------------------------------------------------------\n"

    # Per level: total, correct, finished (terminated)
    tot = defaultdict(int)
    corr = defaultdict(int)
    fin = defaultdict(int)
    tot_all = corr_all = fin_all = 0

    missing_logs: list[str] = []
    unknown_level = 0

    for qid, m in meta.items():
        lv = m["level"]
        path = os.path.join(logs_dir, f"{qid}.txt")
        if not os.path.isfile(path):
            missing_logs.append(qid)
            continue
        if lv == 0:
            unknown_level += 1
        with open(path, encoding="utf-8") as f:
            logs = f.read()
        terminated = "TERMINATE" in logs
        pred_region = _prediction_region_for_judge(logs)
        ok = judge(pred_region, m["answer"]) if terminated else False

        tot[lv] += 1
        tot_all += 1
        if terminated:
            fin[lv] += 1
            fin_all += 1
            if ok:
                corr[lv] += 1
                corr_all += 1

    def pct(num: float, den: float) -> float:
        return 100.0 * num / den if den else 0.0

    print("MIMIC-III validation (questions with a log file under {})".format(logs_dir))
    print("Complexity level (paper A.2): distinct MIMIC tables in gold SQL (capped at IV for 4+).")
    print("")
    for lv in (1, 2, 3, 4):
        print(
            "  Level {}:  SR={:.2f}%  (correct={}/{}, unfinished={})".format(
                lv,
                pct(corr[lv], tot[lv]),
                corr[lv],
                tot[lv],
                tot[lv] - fin[lv],
            )
        )
    print(
        "  All:      SR={:.2f}%  CR={:.2f}%  (correct={}/{}, finished={}/{})".format(
            pct(corr_all, tot_all),
            pct(fin_all, tot_all),
            corr_all,
            tot_all,
            fin_all,
            tot_all,
        )
    )
    print("")
    print("Paper Table 1 (EHRAgent, GPT-4-0613) reference — MIMIC-III:")
    print("  SR:  71.58 / 66.34 / 49.70 / 49.14 / All 58.97   CR: 85.86")
    if missing_logs:
        print("")
        print("Warning: {} questions have no log (not counted above). Example ids: {}".format(
            len(missing_logs),
            ", ".join(missing_logs[:5]),
        ))
    if unknown_level:
        print("")
        print("Note: {} logged questions had level 0 (no table matched in gold query).".format(unknown_level))


if __name__ == "__main__":
    main()
