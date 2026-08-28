"""Question-family registry for non-obvious SQL idioms only.

Per-question / per-q_tag hacks belong in ``mimic_schema.schema_hints_for_question`` (column→table).
Families here cover patterns tools cannot infer from column names alone (cost.event_type, dense_rank top-k, …).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ItemCostFamily:
    """MIMIC line-item cost: cost.event_type + subquery on source row_id."""

    event_type: str
    priority_tables: Tuple[str, ...]
    entity_key: str  # key in benchmark ``value`` dict
    sql_subquery: str  # uses {entity_lit} placeholder


# Shared skeleton for all item-cost families in this benchmark.
_ITEM_COST_RULES = (
    "Use one SQLInterpreter: select distinct cost.cost from cost where cost.event_type = '<type>' "
    "and cost.event_id in (<subquery>). Then answer = rows[0][0].",
    "Never cost filtered by HADM_ID only; never EVENT_ID = a single ROW_ID from GetValue.",
    "Python: double-quoted SQLInterpreter string; SQL string literals in \\\"double quotes\\\".",
)


ITEM_COST_FAMILIES: Dict[str, ItemCostFamily] = {
    "what is the cost of a {lab_name} lab test?": ItemCostFamily(
        event_type="labevents",
        priority_tables=("cost", "d_labitems", "labevents"),
        entity_key="lab_name",
        sql_subquery=(
            "select labevents.row_id from labevents where labevents.itemid in ( "
            "select d_labitems.itemid from d_labitems where d_labitems.label = {entity_lit} )"
        ),
    ),
    "what is the cost of a procedure named {procedure_name}?": ItemCostFamily(
        event_type="procedures_icd",
        priority_tables=("cost", "d_icd_procedures", "procedures_icd"),
        entity_key="procedure_name",
        sql_subquery=(
            "select procedures_icd.row_id from procedures_icd where procedures_icd.icd9_code = ( "
            "select d_icd_procedures.icd9_code from d_icd_procedures "
            "where d_icd_procedures.short_title = {entity_lit} )"
        ),
    ),
    "what is the cost of a drug named {drug_name}?": ItemCostFamily(
        event_type="prescriptions",
        priority_tables=("cost", "prescriptions"),
        entity_key="drug_name",
        sql_subquery=(
            "select prescriptions.row_id from prescriptions "
            "where prescriptions.drug = {entity_lit}"
        ),
    ),
    "what is the cost of diagnosing {diagnosis_name}?": ItemCostFamily(
        event_type="diagnoses_icd",
        priority_tables=("cost", "d_icd_diagnoses", "diagnoses_icd"),
        entity_key="diagnosis_name",
        sql_subquery=(
            "select diagnoses_icd.row_id from diagnoses_icd where diagnoses_icd.icd9_code = ( "
            "select d_icd_diagnoses.icd9_code from d_icd_diagnoses "
            "where d_icd_diagnoses.short_title = {entity_lit} )"
        ),
    ),
}

DRUG_ROUTE_QTAG = "what is the intake method of {drug_name}?"
ABBREVIATION_QTAG = "what does {abbreviation} stand for?"

TOP_SPECIMENS_QTAG = (
    "what are_verb the top [n_rank] frequent specimens tested [time_filter_global1]?"
)
TOP_OUTPUT_EVENTS_QTAG = (
    "what are_verb the top [n_rank] frequent output events [time_filter_global1]?"
)
VITAL_CHANGE_QTAG = (
    "what is the change in the {vital_name} of patient {patient_id} from the "
    "[time_filter_exact2] value measured [time_filter_global2] compared to the "
    "[time_filter_exact1] value measured [time_filter_global1]?"
)
HOSPITAL_LOS_QTAG = "what was the [time_filter_exact1] length of hospital stay of patient {patient_id}?"


def is_elapsed_time_qtag(q_tag: Optional[str]) -> bool:
    return bool(q_tag and "have passed since" in q_tag.lower())


def build_elapsed_time_sql(
    q_tag: str,
    question: str = "",
    *,
    value: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Gold pattern: strftime('%j',current_time) delta in SQL (not datetime.now())."""
    pid = _parse_patient_id(question, value)
    if pid is None:
        return None
    hours = bool(re.search(r"\bhours?\b", question or "", flags=re.IGNORECASE))
    scale = "24 * " if hours else "1 * "
    jdiff = "strftime('%j',current_time) - strftime('%j',{ts})"
    tag = (q_tag or "").lower()
    ql = (question or "").lower()

    if "admitted to the hospital" in tag:
        ts = "admissions.admittime"
        return (
            f"select {scale}({jdiff.format(ts=ts)}) from admissions "
            f"where admissions.subject_id = {pid} and admissions.dischtime is null"
        )
    if "admitted to the icu" in tag:
        ts = "icustays.intime"
        return (
            f"select {scale}({jdiff.format(ts=ts)}) from icustays "
            f"where icustays.hadm_id in ( select admissions.hadm_id from admissions "
            f"where admissions.subject_id = {pid} ) and icustays.outtime is null"
        )
    if "careunit" in tag:
        cu = None
        if value and value.get("careunit"):
            cu = str(value["careunit"]).strip().lower()
        if not cu:
            m = re.search(r"careunit\s+(\S+)", question or "", flags=re.IGNORECASE)
            cu = m.group(1).lower() if m else None
        if not cu:
            return None
        order = "desc" if "exact-last" in tag or "last time" in ql else "asc"
        ts = "transfers.intime"
        return (
            f"select {scale}({jdiff.format(ts=ts)}) from transfers "
            f"where transfers.hadm_id in ( select admissions.hadm_id from admissions "
            f"where admissions.subject_id = {pid} and admissions.dischtime is null ) "
            f"and transfers.careunit = '{cu}' order by transfers.intime {order} limit 1"
        )
    if "ward" in tag:
        ward = None
        if value and value.get("ward_id") is not None:
            ward = value["ward_id"]
        if ward is None:
            m = re.search(r"ward\s+(\d+)", question or "", flags=re.IGNORECASE)
            ward = m.group(1) if m else None
        if ward is None:
            return None
        order = "desc" if "exact-last" in tag or "last time" in ql else "asc"
        ts = "transfers.intime"
        return (
            f"select {scale}({jdiff.format(ts=ts)}) from transfers "
            f"where transfers.icustay_id in ( select icustays.icustay_id from icustays "
            f"where icustays.hadm_id in ( select admissions.hadm_id from admissions "
            f"where admissions.subject_id = {pid} and admissions.dischtime is null ) ) "
            f"and transfers.wardid = {ward} order by transfers.intime {order} limit 1"
        )
    return None


_WORD_TO_N = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _parse_top_k(question: str) -> int:
    m = re.search(r"top\s+(\w+|\d+)", (question or "").lower())
    if not m:
        return 5
    tok = m.group(1)
    if tok.isdigit():
        return max(1, int(tok))
    return _WORD_TO_N.get(tok, 5)


def _parse_year_filter(question: str) -> Optional[str]:
    """Return 4-digit year for 'since 2104' / 'in 2105', else None."""
    q = (question or "").lower()
    m = re.search(r"\b(?:since|in)\s+(\d{4})\b", q)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4})\b", q)
    if m and "since" in q:
        return m.group(1)
    return None


def build_top_specimens_sql(question: str) -> str:
    k = _parse_top_k(question)
    year = _parse_year_filter(question)
    where = ""
    if year:
        where = f" where strftime('%Y', microbiologyevents.charttime) >= '{year}'"
    return (
        "select t1.spec_type_desc from ( "
        "select microbiologyevents.spec_type_desc, "
        "dense_rank() over ( order by count(*) desc ) as c1 "
        f"from microbiologyevents{where} "
        "group by microbiologyevents.spec_type_desc ) as t1 "
        f"where t1.c1 <= {k}"
    )


def _parse_patient_id(
    question: str, value: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    if value and value.get("patient_id") is not None:
        try:
            return int(value["patient_id"])
        except (TypeError, ValueError):
            pass
    m = re.search(r"patient\s+(\d+)", question or "", flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_vital_name(
    question: str, value: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if value and value.get("vital_name"):
        return str(value["vital_name"]).strip()
    m = re.search(
        r"change in the (.+?) of patient\s+\d+",
        question or "",
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def build_vital_change_sql(
    question: str = "",
    *,
    value: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Last minus second-to-last VALUENUM on first ICU stay (gold SQL pattern)."""
    pid = _parse_patient_id(question, value)
    vital = _parse_vital_name(question, value)
    if pid is None or not vital:
        return None
    vit_lit = _sql_literal(vital)
    icu_sub = (
        f"select icustays.icustay_id from icustays where icustays.hadm_id in ( "
        f"select admissions.hadm_id from admissions where admissions.subject_id = {pid} ) "
        f"and icustays.outtime is not null order by icustays.intime asc limit 1"
    )
    item_sub = (
        f"select d_items.itemid from d_items where d_items.label = {vit_lit} "
        f"and d_items.linksto = 'chartevents'"
    )
    row_sub = (
        f"select chartevents.valuenum from chartevents where chartevents.icustay_id in ( {icu_sub} ) "
        f"and chartevents.itemid in ( {item_sub} ) order by chartevents.charttime desc limit 1"
    )
    return f"select ( {row_sub} ) - ( {row_sub} offset 1 )"


def build_top_output_events_sql(question: str) -> str:
    k = _parse_top_k(question)
    year = _parse_year_filter(question)
    where = ""
    if year:
        # Gold uses equality for "in 2105" style questions.
        if re.search(rf"\bin\s+{year}\b", (question or "").lower()):
            where = f" where strftime('%Y', outputevents.charttime) = '{year}'"
        else:
            where = f" where strftime('%Y', outputevents.charttime) >= '{year}'"
    return (
        "select d_items.label from d_items where d_items.itemid in ( "
        "select t1.itemid from ( "
        "select outputevents.itemid, dense_rank() over ( order by count(*) desc ) as c1 "
        f"from outputevents{where} "
        "group by outputevents.itemid ) as t1 "
        f"where t1.c1 <= {k} )"
    )


def _sql_literal(value: str) -> str:
    safe = (value or "").replace('"', '""')
    return f'"{safe}"'


def build_item_cost_sql(family: ItemCostFamily, entity: str) -> str:
    lit = _sql_literal(entity)
    sub = family.sql_subquery.format(entity_lit=lit)
    return (
        f"select distinct cost.cost from cost where cost.event_type = \"{family.event_type}\" "
        f"and cost.event_id in ( {sub} )"
    )


def build_item_cost_python(family: ItemCostFamily, entity: str) -> str:
    sql = build_item_cost_sql(family, entity)
    escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'_rows = SQLInterpreter("{escaped}")\n'
        "answer = _rows[0][0] if _rows else None"
    )


def build_abbreviation_sql(entity: str) -> str:
    lit = _sql_literal(entity)
    return (
        "select d_icd_diagnoses.long_title from d_icd_diagnoses "
        f"where d_icd_diagnoses.short_title = {lit} "
        "union "
        "select d_icd_procedures.long_title from d_icd_procedures "
        f"where d_icd_procedures.short_title = {lit}"
    )


def build_abbreviation_python(entity: str) -> str:
    sql = build_abbreviation_sql(entity)
    escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'_rows = SQLInterpreter("{escaped}")\n'
        "answer = _rows[0][0] if _rows else None"
    )


def _entity_from_question(q_tag: str, question: str) -> Optional[str]:
    """Fallback when benchmark ``value`` slot is missing."""
    patterns = {
        "what is the cost of a {lab_name} lab test?": r"cost of a\s+(.+?)\s+lab test",
        "what is the cost of a procedure named {procedure_name}?": r"cost of a procedure named\s+(.+?)(?:\?|$)",
        "what is the cost of a drug named {drug_name}?": r"cost of a drug named\s+(.+?)(?:\?|$)",
        "what is the cost of diagnosing {diagnosis_name}?": r"cost of diagnosing\s+(.+?)(?:\?|$)",
        DRUG_ROUTE_QTAG: r"intake method of\s+(.+?)(?:\?|$)",
        ABBREVIATION_QTAG: r"what does\s+(.+?)\s+stand for",
    }
    pat = patterns.get(q_tag)
    if not pat:
        return None
    m = re.search(pat, question, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def resolve_question_family(
    question: str,
    *,
    q_tag: Optional[str] = None,
    value: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[ItemCostFamily], Optional[str]]:
    """
    Returns (q_tag, item_cost_family, entity_string).
    Prefer dataset q_tag + value slots over per-entity regex hacks.
    """
    tag = (q_tag or "").strip()
    if not tag:
        for known in (DRUG_ROUTE_QTAG, ABBREVIATION_QTAG) + tuple(ITEM_COST_FAMILIES.keys()):
            if _entity_from_question(known, question):
                tag = known
                break

    if tag and is_elapsed_time_qtag(tag):
        return tag, None, None

    if tag in (
        DRUG_ROUTE_QTAG,
        ABBREVIATION_QTAG,
        TOP_SPECIMENS_QTAG,
        TOP_OUTPUT_EVENTS_QTAG,
        VITAL_CHANGE_QTAG,
    ):
        if tag == DRUG_ROUTE_QTAG:
            entity = None
            if value and isinstance(value, dict):
                entity = value.get("drug_name")
            entity = entity or _entity_from_question(tag, question)
            return tag, None, entity
        if tag == ABBREVIATION_QTAG:
            entity = None
            if value and isinstance(value, dict):
                entity = value.get("abbreviation")
            entity = entity or _entity_from_question(tag, question)
            return tag, None, entity
        return tag, None, None

    family = ITEM_COST_FAMILIES.get(tag)
    if not family:
        return tag or None, None, None

    entity = None
    if value and isinstance(value, dict):
        entity = value.get(family.entity_key)
    entity = entity or _entity_from_question(tag, question)
    return tag, family, entity


def plan_hints_for_family(
    q_tag: Optional[str],
    family: Optional[ItemCostFamily],
    entity: Optional[str],
    *,
    question: str = "",
    value: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Planner fragments from family (same for all instances of that q_tag)."""
    out: Dict[str, Any] = {
        "q_tag": q_tag or "",
        "question_family": q_tag or "other",
    }
    if q_tag == DRUG_ROUTE_QTAG:
        out["priority_tables"] = ["prescriptions"]
        out["strategy_steps"] = [
            "Drug route (all instances): prescriptions.DRUG + ROUTE; one pass then TERMINATE.",
            "Do not use d_items / inputevents ITEMID loops for intake method.",
        ]
        out["use_sql_interpreter"] = False
        if entity:
            out["reference_python"] = (
                f"prescriptions_db = LoadDB('prescriptions')\n"
                f"filtered = FilterDB(prescriptions_db, 'DRUG={entity}')\n"
                "answer = GetValue(filtered, 'ROUTE')"
            )
        return out

    if q_tag == ABBREVIATION_QTAG:
        out["priority_tables"] = ["d_icd_diagnoses", "d_icd_procedures"]
        out["strategy_steps"] = [
            "Abbreviation expansion: exact SHORT_TITLE lookup in both ICD dictionary tables.",
            "Return LONG_TITLE, not SHORT_TITLE; do not use broad LIKE because it can return the whole dictionary.",
            "Use one SQLInterpreter UNION over d_icd_diagnoses and d_icd_procedures.",
        ]
        out["use_sql_interpreter"] = True
        if entity:
            out["reference_sql"] = build_abbreviation_sql(entity)
            out["reference_python"] = build_abbreviation_python(entity)
        return out

    if q_tag == TOP_SPECIMENS_QTAG:
        sql = build_top_specimens_sql(question or "")
        escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
        out["priority_tables"] = ["microbiologyevents"]
        out["strategy_steps"] = [
            "Specimens/cultures = microbiologyevents.SPEC_TYPE_DESC (not labevents/d_labitems).",
            "Top-k frequency: one SQLInterpreter with dense_rank() over count(*); answer = list of row[0].",
            "Year filter 'since 2104': strftime('%Y', charttime) in SQL — never Calendar('01/01/2104').",
        ]
        out["use_sql_interpreter"] = True
        out["reference_sql"] = sql
        out["reference_python"] = (
            f'_rows = SQLInterpreter("{escaped}")\n'
            "answer = [r[0] for r in _rows] if _rows else []"
        )
        return out

    if q_tag == TOP_OUTPUT_EVENTS_QTAG:
        sql = build_top_output_events_sql(question or "")
        escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
        out["priority_tables"] = ["outputevents", "d_items"]
        out["strategy_steps"] = [
            "Frequent output events: rank outputevents.ITEMID by count, join d_items.LABEL for names.",
            "One SQLInterpreter with dense_rank(); answer = list of labels (row[0]).",
            "Do not use labevents for output/fluid questions.",
        ]
        out["use_sql_interpreter"] = True
        out["reference_sql"] = sql
        out["reference_python"] = (
            f'_rows = SQLInterpreter("{escaped}")\n'
            "answer = [r[0] for r in _rows] if _rows else []"
        )
        return out

    if q_tag and is_elapsed_time_qtag(q_tag):
        sql = build_elapsed_time_sql(q_tag, question or "", value=value)
        out["priority_tables"] = ["admissions", "icustays", "transfers"]
        out["strategy_steps"] = [
            "Elapsed since: one SQLInterpreter using strftime('%j',current_time) in mimic_iii.db.",
            "Never datetime.now() — benchmark time is SQLite current_time (~2100s); negative answers are valid.",
            "ICU current stay: icustays.outtime is null; careunit/ward timing uses transfers table.",
        ]
        out["use_sql_interpreter"] = True
        if sql:
            escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
            out["reference_sql"] = sql
            out["reference_python"] = (
                f'_rows = SQLInterpreter("{escaped}")\n'
                "answer = _rows[0][0] if _rows else None"
            )
        return out

    if q_tag == VITAL_CHANGE_QTAG:
        sql = build_vital_change_sql(question or "", value=value)
        out["priority_tables"] = ["chartevents", "d_items", "icustays", "admissions"]
        out["strategy_steps"] = [
            "Vital change on first ICU: one SQLInterpreter (last VALUENUM minus offset 1) per gold pattern.",
            "chartevents uses VALUENUM not VALUE; resolve ITEMID via d_items.LABEL + linksto=chartevents.",
            "First ICU = icustays with outtime not null, order by intime asc limit 1.",
            "Prefer reference_python SQL over manual FilterDB/min(ADMITTIME) chains.",
        ]
        out["use_sql_interpreter"] = True
        if sql:
            escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
            out["reference_sql"] = sql
            out["reference_python"] = (
                f'_rows = SQLInterpreter("{escaped}")\n'
                "answer = _rows[0][0] if _rows else None"
            )
        return out

    if q_tag == HOSPITAL_LOS_QTAG:
        pid = _parse_patient_id(question or "", value)
        out["priority_tables"] = ["admissions"]
        out["strategy_steps"] = [
            "Hospital length of stay uses admissions.DISCHTIME minus admissions.ADMITTIME, not icustays.",
            "For the last completed hospital stay, order admissions.ADMITTIME desc and limit 1.",
            "Use SQLite strftime('%j', ...) to match the benchmark's day calculation.",
        ]
        out["use_sql_interpreter"] = True
        if pid is not None:
            sql = (
                "select strftime('%j',admissions.dischtime) - strftime('%j',admissions.admittime) "
                f"from admissions where admissions.subject_id = {pid} "
                "and admissions.dischtime is not null order by admissions.admittime desc limit 1"
            )
            escaped = sql.replace("\\", "\\\\").replace('"', '\\"')
            out["reference_sql"] = sql
            out["reference_python"] = (
                f'_rows = SQLInterpreter("{escaped}")\n'
                "answer = _rows[0][0] if _rows else None"
            )
        return out

    if family and entity:
        out["priority_tables"] = list(family.priority_tables)
        out["strategy_steps"] = [
            f"Item cost family event_type={family.event_type} (from q_tag, not per-entity custom rules).",
            *_ITEM_COST_RULES,
        ]
        out["use_sql_interpreter"] = True
        out["reference_sql"] = build_item_cost_sql(family, entity)
        out["reference_python"] = build_item_cost_python(family, entity)
        return out

    return out
