"""
Build WorldMM-compatible timeline JSON from EHRSQL-EHRAgent MIMIC-III CSVs.

Patient and encounter anchors are inferred only from the natural-language question or
non-gold row metadata. Reference SQL and gold answers are never read by this module.
Rows without an unambiguous patient encounter return None.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


_ADM_PATH: Optional[str] = None
_ADM_DF: Optional[pd.DataFrame] = None
_ICU_PATH: Optional[str] = None
_ICU_DF: Optional[pd.DataFrame] = None
_DLAB_PATH: Optional[str] = None
_DLAB_MAP: Optional[Dict[int, str]] = None
_DICD_PATH: Optional[str] = None
_DICD_DF: Optional[pd.DataFrame] = None


def mimic_iii_csv_dir() -> str:
    root = os.environ.get("EHRAGENT_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        return ""
    for sub in (os.path.join(root, "ehrsql", "mimic_iii"), os.path.join(root, "mimic_iii")):
        if os.path.isfile(os.path.join(sub, "ADMISSIONS.csv")):
            return sub
    return os.path.join(root, "mimic_iii")


def _get_admissions(mimic_dir: str) -> pd.DataFrame:
    global _ADM_PATH, _ADM_DF
    path = os.path.join(mimic_dir, "ADMISSIONS.csv")
    if _ADM_DF is not None and _ADM_PATH == path:
        return _ADM_DF
    _ADM_PATH = path
    _ADM_DF = pd.read_csv(path)
    return _ADM_DF


def _get_icustays(mimic_dir: str) -> pd.DataFrame:
    global _ICU_PATH, _ICU_DF
    path = os.path.join(mimic_dir, "ICUSTAYS.csv")
    if _ICU_DF is not None and _ICU_PATH == path:
        return _ICU_DF
    _ICU_PATH = path
    _ICU_DF = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
    return _ICU_DF


def _get_d_lab_map(mimic_dir: str) -> Dict[int, str]:
    global _DLAB_PATH, _DLAB_MAP
    path = os.path.join(mimic_dir, "D_LABITEMS.csv")
    if _DLAB_MAP is not None and _DLAB_PATH == path:
        return _DLAB_MAP
    _DLAB_PATH = path
    df = pd.read_csv(path)
    _DLAB_MAP = {int(r["ITEMID"]): str(r["LABEL"]) for _, r in df.iterrows() if pd.notna(r.get("ITEMID"))}
    return _DLAB_MAP


def _get_d_icd(mimic_dir: str) -> pd.DataFrame:
    global _DICD_PATH, _DICD_DF
    path = os.path.join(mimic_dir, "D_ICD_DIAGNOSES.csv")
    if _DICD_DF is not None and _DICD_PATH == path:
        return _DICD_DF
    _DICD_PATH = path
    _DICD_DF = pd.read_csv(path)
    return _DICD_DF


def _same_hadm(a: Any, hadm: int) -> bool:
    try:
        return int(float(a)) == int(hadm)
    except (TypeError, ValueError):
        return False


def _in_icu_stay(ts: pd.Timestamp, icu_rows: pd.DataFrame) -> bool:
    if icu_rows.empty or pd.isna(ts):
        return False
    for _, r in icu_rows.iterrows():
        a, b = pd.to_datetime(r["INTIME"], errors="coerce"), pd.to_datetime(r["OUTTIME"], errors="coerce")
        if pd.isna(a) or pd.isna(b):
            continue
        if a <= ts <= b:
            return True
    return False


def _labs_for_hadm(
    mimic_dir: str,
    hadm_id: int,
    max_labs: int,
    chunksize: int,
) -> pd.DataFrame:
    path = os.path.join(mimic_dir, "LABEVENTS.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    usecols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"]
    parts: List[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=lambda c: c in usecols):
        sub = chunk[chunk["HADM_ID"].map(lambda x: _same_hadm(x, hadm_id))]
        if sub.empty:
            continue
        parts.append(sub)
        total += len(sub)
        if total >= max_labs:
            break
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("CHARTTIME").head(max_labs)


def build_timeline_dict(
    mimic_dir: str,
    hadm_id: int,
    *,
    max_labs: int = 120,
    lab_chunksize: int = 100_000,
) -> Dict[str, Any]:
    adm = _get_admissions(mimic_dir)
    row = adm[adm["HADM_ID"].map(lambda x: _same_hadm(x, hadm_id))]
    if row.empty:
        raise ValueError(f"HADM_ID {hadm_id} not in ADMISSIONS.csv")
    r0 = row.iloc[0]
    subject_id = int(r0["SUBJECT_ID"])
    admittime = pd.to_datetime(r0["ADMITTIME"], errors="coerce")
    dischtime = pd.to_datetime(r0["DISCHTIME"], errors="coerce")

    def t_enc(charttime: Any) -> int:
        if pd.isna(charttime):
            return 0
        ts = pd.to_datetime(charttime, errors="coerce")
        if pd.isna(ts) or pd.isna(admittime):
            return 0
        return int(max(0, (ts - admittime).total_seconds()))

    until_time = int(max(t_enc(dischtime), 0))

    icu_df = _get_icustays(mimic_dir)
    icu_df = icu_df[icu_df["HADM_ID"].map(lambda x: _same_hadm(x, hadm_id))] if not icu_df.empty else icu_df

    d_icd = _get_d_icd(mimic_dir)
    dx = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    dx = dx[dx["HADM_ID"].map(lambda x: _same_hadm(x, hadm_id))]
    dx = dx.copy()
    dx["ICD9_CODE"] = dx["ICD9_CODE"].astype(str).str.strip()
    d_icd = d_icd.copy()
    d_icd["ICD9_CODE"] = d_icd["ICD9_CODE"].astype(str).str.strip()
    merged = dx.merge(d_icd, on="ICD9_CODE", how="left")

    conditions: List[str] = []
    seen: Set[str] = set()
    semantic_triples: List[Dict[str, Any]] = []
    pn = f"patient:{subject_id}"
    for _, r in merged.iterrows():
        title = str(r.get("LONG_TITLE") or r["ICD9_CODE"]).strip()
        if title and title not in seen:
            seen.add(title)
            conditions.append(title)
        semantic_triples.append(
            {
                "subj": pn,
                "pred": "has_icd_diagnosis",
                "obj": f"{r['ICD9_CODE']} ({title})",
                "end_ts": 0,
                "source": "diagnoses_icd",
            }
        )

    item_label = _get_d_lab_map(mimic_dir)
    events: List[Dict[str, Any]] = []

    rx_path = os.path.join(mimic_dir, "PRESCRIPTIONS.csv")
    rx = pd.read_csv(rx_path) if os.path.isfile(rx_path) else pd.DataFrame()
    if not rx.empty:
        rx = rx[rx["HADM_ID"].map(lambda x: _same_hadm(x, hadm_id))]
        for _, r in rx.iterrows():
            st = r.get("STARTDATE")
            if pd.isna(st):
                continue
            st = pd.to_datetime(st, errors="coerce")
            if pd.isna(st):
                continue
            drug = str(r.get("DRUG", "") or "")
            parts = [
                drug,
                str(r.get("DOSE_VAL_RX", "") or ""),
                str(r.get("DOSE_UNIT_RX", "") or ""),
                str(r.get("ROUTE", "") or ""),
            ]
            text = " ".join(p for p in parts if p).strip()
            icu = _in_icu_stay(st, icu_df)
            events.append(
                {
                    "t": t_enc(st),
                    "type": "medication",
                    "name": drug,
                    "text": text or drug,
                    "icu": icu,
                    "resolution": "icu_hours" if icu else "admission_days",
                }
            )

    labs = _labs_for_hadm(mimic_dir, hadm_id, max_labs=max_labs, chunksize=lab_chunksize)
    for _, r in labs.iterrows():
        ct = r.get("CHARTTIME")
        if pd.isna(ct):
            continue
        ts = pd.to_datetime(ct, errors="coerce")
        if pd.isna(ts):
            continue
        iid = int(float(r["ITEMID"])) if pd.notna(r.get("ITEMID")) else -1
        name = str(item_label.get(iid, f"itemid_{iid}"))
        val = r.get("VALUENUM")
        uom = r.get("VALUEUOM", "")
        if pd.notna(val):
            value_str = f"{val} {uom}".strip() if pd.notna(uom) and str(uom).strip() else str(val)
        else:
            value_str = ""
        icu = _in_icu_stay(ts, icu_df)
        events.append(
            {
                "t": t_enc(ts),
                "type": "lab",
                "name": name,
                "value": value_str,
                "icu": icu,
                "resolution": "icu_hours" if icu else "admission_days",
            }
        )

    events.sort(key=lambda e: (int(e["t"]), str(e.get("type", ""))))

    if not rx.empty:
        active_meds = sorted({str(r["DRUG"]) for _, r in rx.iterrows() if pd.notna(r.get("DRUG"))})[:24]
    else:
        active_meds = []

    out: Dict[str, Any] = {
        "subject_id": str(subject_id),
        "hadm_id": str(hadm_id),
        "until_time": max(until_time, max((int(e["t"]) for e in events), default=0)),
        "conditions": conditions[:20],
        "active_medications": active_meds,
        "labs": {},
        "semantic_triples": semantic_triples[:80],
        "clinical_knowledge_entries": [],
        "events": events,
        "mimic_meta": {
            "admittime": str(admittime),
            "dischtime": str(dischtime),
            "source": "ehrsql_csv_timeline",
            "max_labs": max_labs,
        },
    }
    return out


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def infer_hadm_id_from_question(row: Dict[str, Any], mimic_dir: str) -> Optional[int]:
    """Resolve one admission without consulting reference SQL or gold answers."""
    value = row.get("value") or {}
    if isinstance(value, dict):
        for key in ("hadm_id", "HADM_ID"):
            raw = value.get(key)
            if raw not in (None, ""):
                try:
                    return int(float(raw))
                except (TypeError, ValueError):
                    pass

    question = str(row.get("template") or row.get("question") or "")
    match = re.search(r"\bpatient\s+(\d+)\b", question, flags=re.IGNORECASE)
    if not match:
        return None
    subject_id = int(match.group(1))
    admissions = _get_admissions(mimic_dir)
    selected = admissions[admissions["SUBJECT_ID"] == subject_id].copy()
    if selected.empty:
        return None
    selected["ADMITTIME"] = pd.to_datetime(selected["ADMITTIME"], errors="coerce")
    q = question.lower()
    if any(token in q for token in ("first", "earliest", "initial")):
        selected = selected.sort_values("ADMITTIME", ascending=True)
    elif any(token in q for token in ("last", "latest", "most recent")):
        selected = selected.sort_values("ADMITTIME", ascending=False)
    elif len(selected) != 1:
        return None
    return int(float(selected.iloc[0]["HADM_ID"]))


def timeline_path_for_benchmark_row(
    row: Dict[str, Any],
    *,
    cache_dir: str,
    mimic_dir: Optional[str] = None,
    max_labs: int = 120,
) -> Optional[str]:
    """
    Return path to a JSON file loadable by ``EHRMWorldMemory.load_mimic_json``, or None.
    Caches by ``HADM_ID`` under ``cache_dir``.
    """
    mdir = (mimic_dir or "").strip() or mimic_iii_csv_dir()
    if not mdir or not os.path.isfile(os.path.join(mdir, "ADMISSIONS.csv")):
        return None

    hadm = infer_hadm_id_from_question(row, mdir)
    if hadm is None:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"hadm_{hadm}.json")
    if os.path.isfile(cache_path):
        return cache_path

    try:
        data = build_timeline_dict(mdir, hadm, max_labs=max_labs)
    except (OSError, ValueError, KeyError):
        return None
    _atomic_write_json(cache_path, data)
    return cache_path
