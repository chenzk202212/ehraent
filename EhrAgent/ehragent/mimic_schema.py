"""MIMIC-III CSV column registry — one place for table/column facts (not per-question hacks)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

# Question wording → canonical column (checked against loaded CSV headers).
_PHRASE_TO_COLUMN: Tuple[Tuple[str, str], ...] = (
    (r"\bmarital\s+status\b", "MARITAL_STATUS"),
    (r"\bmarital\b", "MARITAL_STATUS"),
    (r"\bintake\s+method\b", "ROUTE"),
    (r"\bintake\s+route\b", "ROUTE"),
    (r"\badministration\s+route\b", "ROUTE"),
    (r"\binsurance\b", "INSURANCE"),
    (r"\bethnicity\b", "ETHNICITY"),
    (r"\blanguage\b", "LANGUAGE"),
    (r"\bgender\b", "GENDER"),
    (r"\bdate of birth\b", "DOB"),
    (r"\bdod\b", "DOD"),
    (r"\badmit\s*time\b", "ADMITTIME"),
    (r"\bdischarge\s*time\b", "DISCHTIME"),
    (r"\bicu\b.*\bintime\b", "INTIME"),
    (r"\bcharttime\b", "CHARTTIME"),
    (r"\bspec_type\b", "SPEC_TYPE_DESC"),
    (r"\bspecimen\b", "SPEC_TYPE_DESC"),
    (r"\bculture\b", "SPEC_TYPE_DESC"),
    (r"\bcareunit\b", "CAREUNIT"),
)

# Columns that must not be read from the wrong table (common agent mistakes).
_COLUMN_TABLE_HINTS: Dict[str, str] = {
    "MARITAL_STATUS": "admissions (not patients)",
    "INSURANCE": "admissions",
    "LANGUAGE": "admissions",
    "ETHNICITY": "admissions",
    "AGE": "admissions",
    "GENDER": "patients",
    "DOB": "patients",
    "DOD": "patients",
    "ROUTE": "prescriptions (DRUG + ROUTE)",
    "VALUENUM": "chartevents or labevents (not VALUE on chartevents)",
    "VALUE": "outputevents (chartevents uses VALUENUM)",
    "DRUG": "prescriptions",
    "SHORT_TITLE": "d_icd_diagnoses or d_icd_procedures",
    "SPEC_TYPE_DESC": "microbiologyevents",
    "CAREUNIT": "transfers.careunit (icustays uses FIRST_CAREUNIT/LAST_CAREUNIT, not CAREUNIT)",
}


def _mimic_csv_dir() -> str:
    root = os.environ.get("EHRAGENT_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        return ""
    for sub in (os.path.join(root, "ehrsql", "mimic_iii"), os.path.join(root, "mimic_iii")):
        if os.path.isfile(os.path.join(sub, "ADMISSIONS.csv")):
            return sub
    return os.path.join(root, "mimic_iii")


@lru_cache(maxsize=1)
def load_table_columns() -> Dict[str, List[str]]:
    """Uppercase table name → column names from CSV headers."""
    mimic_dir = _mimic_csv_dir()
    out: Dict[str, List[str]] = {}
    if not mimic_dir or not os.path.isdir(mimic_dir):
        return out
    for name in os.listdir(mimic_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(mimic_dir, name)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()
            if not header:
                continue
            cols = [c.strip().upper() for c in header.split(",")]
            table = name[:-4].lower()
            out[table] = cols
        except OSError:
            continue
    return out


def tables_with_column(column: str) -> List[str]:
    col = (column or "").strip().upper()
    schema = load_table_columns()
    return sorted(t for t, cols in schema.items() if col in cols)


def column_lookup_hint(column: str) -> str:
    col = (column or "").strip().upper()
    tables = tables_with_column(col)
    extra = _COLUMN_TABLE_HINTS.get(col, "")
    if not tables:
        return f"Column {col}: not found in loaded CSV headers under EHRAGENT_DATA_ROOT."
    loc = ", ".join(tables[:4])
    msg = f"Column {col} is on: {loc}."
    if extra:
        msg += f" ({extra})"
    return msg


def columns_mentioned_in_question(question: str) -> List[str]:
    q = (question or "").lower()
    found: List[str] = []
    seen: Set[str] = set()
    for pat, col in _PHRASE_TO_COLUMN:
        if re.search(pat, q, flags=re.IGNORECASE) and col not in seen:
            seen.add(col)
            found.append(col)
    return found


def schema_hints_for_question(question: str, *, max_hints: int = 4) -> List[str]:
    """
  Generic planner constraints from schema + question text (all q_tags).
  Does not need per-question Python templates.
    """
    hints: List[str] = []
    ql = (question or "").lower()
    # GLOBAL: any date/time arithmetic must use DB-relative time, never the
    # real-world system clock. This is the single most common Level IV error
    # (datetime.now() against ~2100s MIMIC dates yields garbage ~700000-hour deltas).
    _time_arith_signal = (
        re.search(r"how many\s+(hours?|days?|months?|years?|weeks?)\b", ql)
        or re.search(r"\b(since|ago|elapsed|duration|length of (stay|icu)|how long)\b", ql)
        or re.search(r"\bage[ds]?\b", ql)
        or "datetime.now" in ql
    )
    if _time_arith_signal:
        hints.append(
            "NEVER use Python datetime.now() or any real-world system clock for date math "
            "(MIMIC dates are ~2100s; real-world time differs by decades and gives garbage "
            "results like -697000 hours). For 'how many X have passed since Y', use SQLInterpreter "
            "with strftime('%j',current_time)-strftime('%j',TIMESTAMP) (current_time = DB clock, "
            "negative answers are valid). For ages/durations computed purely from two MIMIC "
            "timestamps (e.g. ADMITTIME to DISCHTIME, DOB to ADMITTIME), use datetime.strptime "
            "on both values and subtract them directly -- never mix in datetime.now()."
        )
    for col in columns_mentioned_in_question(question):
        hint = column_lookup_hint(col)
        if hint not in hints:
            hints.append(hint)
        if len(hints) >= max_hints:
            break
    if re.search(r"patient\s+\d+", ql) and "marital" in ql:
        hints.append(
            "First/current hospital visit: filter admissions by SUBJECT_ID "
            "(min(ADMITTIME) or dischtime is null per wording), then read admissions columns."
        )
    if "cost" in ql and "hospital" in ql and "maximum" in ql:
        hints.append("Total hospital cost: sum cost per HADM_ID, not line-item EVENT_ID joins.")
    elif "cost" in ql and not any(
        w in ql for w in ("lab test", "procedure named", "drug named", "diagnosing")
    ):
        pass  # item-cost families handle line items
    if "specimen" in ql or "culture" in ql:
        if not any("microbiologyevents" in h for h in hints):
            hints.append("Specimens/cultures: microbiologyevents.SPEC_TYPE_DESC, not labevents ITEMID.")
    # GLOBAL: common value-column name confusion across events tables.
    if re.search(r"\b(value|amount|level|dose|reading|measurement)\b", ql):
        hints.append(
            "Value-column names differ per table: chartevents/labevents use VALUENUM (numeric) "
            "and VALUEUOM (unit) -- there is NO 'VALUE' column on these. outputevents uses VALUE. "
            "inputevents_cv uses AMOUNT (and RATE). Check the exact column list returned in error "
            "messages rather than assuming VALUE/VALUENUM/AMOUNT are interchangeable."
        )
    if "output event" in ql or ("frequent" in ql and "output" in ql):
        hints.append("Frequent outputs: outputevents + d_items.LABEL (outputevents.VALUE exists).")
    if re.search(r"how many\s+(hours?|days?)\s+.*passed since", ql):
        hints.append(
            "Elapsed time: one SQLInterpreter with strftime('%j',current_time)-strftime('%j',TIMESTAMP); "
            "never Python datetime.now() (MIMIC dates are ~2100s; answers may be negative)."
        )
        if "careunit" in ql:
            hints.append(
                "Careunit timing uses transfers.careunit + transfers.intime on current admission "
                "(admissions.dischtime is null), not icustays.CAREUNIT."
            )
    if re.search(r"(value of|level of|maximum|minimum|highest|lowest)\s+\w+", ql) and (
        "lab" in ql or re.search(
            r"\b(phosphate|glucose|sodium|potassium|creatinine|hemoglobin|hematocrit|"
            r"platelet|wbc|bun|bicarbonate|chloride|magnesium|calcium|lactate|"
            r"bilirubin|albumin|ph|pao2|paco2|troponin)\b",
            ql,
        )
    ):
        hints.append(
            "Lab item name lookup: do NOT guess ITEMID or LABEL spelling. First run "
            "SQLInterpreter(\"select itemid, label from d_labitems where label like '%<term>%'\") "
            "to find the real LABEL/ITEMID (labels are often abbreviated, e.g. 'phosphate' -> "
            "'phosphate' or similar variants), then filter labevents on that ITEMID. "
            "Never hardcode a placeholder ITEMID like 12345."
        )
    return hints[:max_hints]


def fix_for_column_error(error: str, code: str = "") -> Optional[str]:
    """Map tool 'column incorrect' errors to schema-based fixes (any question)."""
    err = (error or "").lower()
    if "column name" not in err and "column" not in err:
        return None
    m = re.search(r"column name\s+(\w+)\s+is incorrect", err, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"columns in this table include\s+([^\.]+)", err, flags=re.IGNORECASE)
        if m and "marital" in err:
            return column_lookup_hint("MARITAL_STATUS")
        return None
    bad = m.group(1).upper()
    tables = tables_with_column(bad)
    if tables:
        return column_lookup_hint(bad)
    # Wrong table: error lists available columns — infer intended column from code
    c_low = (code or "").lower()
    for col in ("MARITAL_STATUS", "VALUENUM", "ROUTE", "DRUG", "SHORT_TITLE"):
        if col.lower() in c_low:
            return column_lookup_hint(col)
    return None
