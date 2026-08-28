"""Memory-driven planner: structured memory state -> actionable plan JSON."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from medagent import _chat_completion
from mimic_schema import schema_hints_for_question
from question_families import (
    DRUG_ROUTE_QTAG,
    ITEM_COST_FAMILIES,
    TOP_OUTPUT_EVENTS_QTAG,
    TOP_SPECIMENS_QTAG,
    VITAL_CHANGE_QTAG,
    is_elapsed_time_qtag,
    plan_hints_for_family,
    resolve_question_family,
)

PLANNER_SYSTEM = (
    "You are the memory-policy module of an EHR coding agent. "
    "Given structured memories (beliefs, events, task pitfalls, active states, skills, few-shot hints), "
    "output a single JSON object that constrains how the executor writes Python using "
    "LoadDB/FilterDB/GetValue/SQLInterpreter/Calendar only. "
    "Do not write code. Output valid JSON only."
)

PLANNER_USER_TEMPLATE = """Question:
{question}

Structured WorldMM memory (patient-specific, may be empty):
{worldmm_json}

Task memory (cross-question tool pitfalls):
{task_json}

Reusable skill memory (long-lived task procedures):
{skill_json}

Recent active task states (execution-state traces, not passive logs):
{state_json}

LTM few-shot summary (nearest examples):
{ltm_json}

Output JSON with keys:
- task_type: one of patient_specific | population_aggregate | temporal | other
- priority_tables: list of MIMIC table names to use first (max 6)
- strategy_steps: list of 3-6 short imperative steps
- constraints: list of hard rules (column names, gender codes f/m, etc.)
- use_sql_interpreter: boolean
- skip_patient_memory: boolean (true if question is population-level with no patient id)
- recovery_hints: list of what to try after FilterDB/column errors
- gates: object with read_psm, read_dm, use_sql, write_psm, write_dm, update
- retrieval_budget: object limiting beliefs, events, triples, pitfalls, skills, executable traces, and context chars

JSON:"""


DEFAULT_RETRIEVAL_BUDGET = {
    "psm_beliefs": 6,
    "psm_events": 4,
    "psm_triples": 8,
    "task_pitfalls": 3,
    "skills": 3,
    "executable_traces": 3,
    "recent_states": 2,
    "max_context_chars": 6500,
}


def _attach_controller_policy(
    question: str,
    plan: Dict[str, Any],
    worldmm: Dict[str, Any],
    task_memory: Dict[str, Any],
) -> Dict[str, Any]:
    """Make the paper's adaptive gates and bounded retrieval explicit and deterministic."""
    out = dict(plan)
    task_type = str(out.get("task_type") or "other")
    patient_specific = task_type == "patient_specific" or bool(re.search(r"patient\s+\d+", question.lower()))
    has_psm = bool(worldmm.get("has_timeline"))
    has_dm = any(
        task_memory.get(k)
        for k in (
            "pitfalls",
            "skills",
            "dynamic_experiences",
            "executable_memory",
            "recent_active_states",
        )
    )
    psm_items = len(worldmm.get("beliefs") or []) + len(worldmm.get("episodic") or [])
    dm_candidates = (task_memory.get("dynamic_experiences") or []) + (task_memory.get("skills") or [])
    dm_relevance = max(
        [float(item.get("retrieval_score", item.get("score", 0.0)) or 0.0) for item in dm_candidates] or [0.0]
    )
    complexity = min(1.0, 0.18 * len(out.get("priority_tables") or []) + (0.25 if out.get("use_sql_interpreter") else 0.0))
    uncertainty = max(0.05, min(0.95, 0.65 * complexity + (0.20 if patient_specific and not has_psm else 0.0) + (0.15 if has_dm and dm_relevance < 0.2 else 0.0)))
    read_psm = has_psm and patient_specific and not bool(out.get("skip_patient_memory"))
    read_dm = has_dm and (dm_relevance >= 0.12 or uncertainty >= 0.55)
    gates = {
        "read_psm": read_psm,
        "read_dm": read_dm,
        "use_sql": bool(out.get("use_sql_interpreter")),
        "write_psm": has_psm and patient_specific,
        "write_dm": True,
        "update": True,
    }
    out["gates"] = gates
    out["skip_patient_memory"] = not read_psm
    out["context_mode"] = (
        "both" if read_psm and read_dm else "psm_only" if read_psm else "dm_only" if read_dm else "neither"
    )
    budget = dict(DEFAULT_RETRIEVAL_BUDGET)
    if uncertainty >= 0.65:
        budget.update({"psm_beliefs": 8, "psm_events": 6, "psm_triples": 10, "task_pitfalls": 4, "skills": 4, "executable_traces": 4, "max_context_chars": 7800})
    elif uncertainty <= 0.30:
        budget.update({"psm_beliefs": 4, "psm_events": 2, "psm_triples": 5, "task_pitfalls": 2, "skills": 2, "executable_traces": 2, "max_context_chars": 5000})
    supplied = out.get("retrieval_budget")
    if isinstance(supplied, dict):
        for key, default in DEFAULT_RETRIEVAL_BUDGET.items():
            try:
                ceiling = max(default, int(budget.get(key, default)))
                budget[key] = max(0, min(int(supplied.get(key, budget.get(key, default))), ceiling))
            except (TypeError, ValueError):
                budget[key] = default
    out["retrieval_budget"] = budget
    out["controller"] = {
        "uncertainty": round(uncertainty, 3),
        "complexity": round(complexity, 3),
        "psm_items": psm_items,
        "dm_relevance": round(dm_relevance, 3),
        "policy": "expand" if uncertainty >= 0.65 else "compress" if uncertainty <= 0.30 else "standard",
    }
    return out


def _is_quota_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "403" in s or "insufficient" in s or "quota" in s or "balance" in s


def _is_hospital_cost_aggregate_question(q: str) -> bool:
    """Total/max hospital stay cost — uses HADM_ID sum loop like few-shot; not line-item cost."""
    ql = q.lower()
    if resolve_question_family(q)[1] is not None:
        return False
    return any(
        p in ql
        for p in (
            "maximum total hospital cost",
            "max total hospital cost",
            "total hospital cost",
            "maximum hospital cost",
        )
    )


def _infer_tables_from_question(
    q: str,
    *,
    q_tag: Optional[str] = None,
    value: Optional[Dict[str, Any]] = None,
) -> List[str]:
    ql = q.lower()
    tables: List[str] = []
    _tag, family, _entity = resolve_question_family(q, q_tag=q_tag, value=value)
    if family:
        tables.extend(family.priority_tables)
    elif _tag == DRUG_ROUTE_QTAG:
        tables.append("prescriptions")
    if any(w in ql for w in ("diagnosed", "diagnosis", "diagnose", "icd")):
        tables.extend(["d_icd_diagnoses", "diagnoses_icd"])
    if any(
        w in ql
        for w in (
            "prescribed",
            "prescription",
            "drug",
            "medication",
            "intake method",
            "intake route",
            "route of",
            "administration",
            "ointment",
            "oral",
            "intravenous",
            "topical",
        )
    ):
        tables.extend(["prescriptions"])
    if any(w in ql for w in ("procedure", "surgery")) and family is None:
        tables.extend(["procedures_icd", "d_icd_procedures"])
    if any(w in ql for w in ("specimen", "culture", "cultures")):
        tables.extend(["microbiologyevents"])
    if family is None and any(w in ql for w in ("output", "foley", "drain")):
        tables.extend(["outputevents", "d_items"])
    if family is None and any(w in ql for w in ("lab", "phosphate", "glucose")) and "specimen" not in ql and "culture" not in ql:
        tables.extend(["labevents", "d_labitems"])
    if any(w in ql for w in ("cost", "charge", "bill")):
        tables.append("cost")
    if any(w in ql for w in ("icu", "careunit", "ward")) or "change in the" in ql:
        tables.append("icustays")
    if "change in the" in ql or any(
        w in ql for w in ("bp", "weight", "temperature", "heart rate", "spo2", "glucose")
    ):
        tables.extend(["chartevents", "d_items"])
    if any(w in ql for w in ("dead", "died", "death", "mortality")):
        tables.extend(["patients", "admissions"])
    if re.search(r"patient\s+\d+", ql):
        tables.extend(["admissions", "patients"])
    if not tables:
        tables = ["admissions", "patients"]
    out: List[str] = []
    for t in tables:
        if t not in out:
            out.append(t)
    return out[:6]


def _strategy_steps_for_question(q: str, tables: List[str]) -> List[str]:
    ql = q.lower()
    steps: List[str] = []
    if "d_icd_diagnoses" in tables:
        steps.append(
            "Resolve diagnosis SHORT_TITLE in d_icd_diagnoses → ICD9_CODE, then filter diagnoses_icd for HADM_ID(s)."
        )
    if "prescriptions" in tables and any(w in ql for w in ("top", "frequently", "prescribed", "drug")):
        steps.append(
            "On matching HADM_ID(s), load prescriptions; count DRUG within same visit/window; return top-k list in answer."
        )
    if "prescriptions" in tables and any(
        w in ql for w in ("intake method", "intake route", "route", "administration", "ointment")
    ):
        steps.append(
            "Filter prescriptions on DRUG (exact or close drug name); read ROUTE for intake/administration (e.g. tp=topical, po=oral). "
            "Do not use d_items/inputevents_cv ITEMID loops for outpatient drug route questions."
        )
        steps.append(
            "After python returns ROUTE once (e.g. po), set answer and reply TERMINATE immediately — "
            "do not re-run the same prescriptions code."
        )
    # Item-cost families are filled from q_tag in _default_plan (not per-entity rules here).
    if any(w in ql for w in ("dead", "died", "death")) and "patients" in tables:
        steps.append("Use patients.DOD or discharge fields with admissions timing for mortality within the visit window.")
    if re.search(r"\b(?:since|in)\s+\d{4}\b", ql):
        steps.append(
            "Absolute year (e.g. since 2104): filter with SQLInterpreter strftime('%Y', CHARTTIME) — "
            "not Calendar('01/01/2104') (returns NULL)."
        )
    elif "since" in ql and ("year" in ql or "month" in ql or "day" in ql):
        steps.append("Use Calendar('-1 year') style relative modifiers before filtering CHARTTIME/ADMITTIME.")
    if re.search(r"patient\s+\d+", ql):
        steps.append("Anchor on SUBJECT_ID via admissions (min/max ADMITTIME as needed) before child tables.")
    if not steps:
        steps = [
            "Load dictionary tables (d_*) before event tables.",
            "Chain FilterDB with || ; use GetValue(..., list) for multiple IDs when needed.",
            "Set answer then reply TERMINATE.",
        ]
    return steps[:6]


def _recovery_hints_for_plan(
    *,
    is_item_cost: bool,
    is_top_k_sql: bool,
    is_vital_change: bool = False,
    is_elapsed: bool = False,
    resolved_tag: Optional[str],
) -> List[str]:
    hints = [
        "On column error, re-read tool error for valid column names.",
        "On empty filter, verify ICD9_CODE and HADM_ID via d_icd_diagnoses + diagnoses_icd first.",
    ]
    if resolved_tag == DRUG_ROUTE_QTAG:
        hints.append(
            "On ITEMID filter errors for drug route questions, switch to prescriptions filtered by DRUG and read ROUTE."
        )
    if is_item_cost:
        hints.extend(
            [
                "On lab cost returning hundreds of COST values, use SQLInterpreter with "
                "cost.event_type=labevents and cost.event_id in (select labevents.row_id ... d_labitems.label=...).",
                "On lab SQL returning 2+ distinct costs (e.g. 7.24 and 9.53), you omitted event_type='labevents' — "
                "rewrite with the subquery pattern; do not use min/max.",
                "On SQL syntax error near hr or labevents, you used '' inside SQLInterpreter — "
                "use double-quoted Python string with \"labevents\" and \"24 hr creatinine\" inside SQL.",
                "On procedure/drug cost errors, join cost via procedures_icd/prescriptions ROW_ID, not HADM_ID on cost alone.",
                "On procedure SQL syntax error near comma, you passed a list into EVENT_ID= — use the full subquery, not one ROW_ID.",
            ]
        )
    if is_top_k_sql:
        hints.extend(
            [
                "Specimens/cultures use microbiologyevents.SPEC_TYPE_DESC, not labevents ITEMID loops.",
                "Output events use outputevents + d_items.LABEL, not labevents.",
                "For since/in YEAR use strftime in SQL, not Calendar('01/01/YEAR').",
            ]
        )
    if is_vital_change:
        hints.extend(
            [
                "On chartevents column VALUE error, use VALUENUM.",
                "On NameError datetime, it is already imported — remove redundant import or use datetime.strptime.",
            ]
        )
    if is_elapsed:
        hints.extend(
            [
                "Replace datetime.now() with SQLInterpreter strftime('%j',current_time) pattern.",
                "Careunit uses transfers.careunit, not icustays.CAREUNIT.",
            ]
        )
    return hints


def _default_plan(
    question: str,
    task_mem: Dict[str, Any],
    worldmm: Dict[str, Any],
    *,
    q_tag: Optional[str] = None,
    value: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    q = question.lower()
    resolved_tag, item_family, entity = resolve_question_family(
        question, q_tag=q_tag, value=value
    )
    family_hints = plan_hints_for_family(
        resolved_tag, item_family, entity, question=question, value=value
    )
    is_item_cost = resolved_tag in ITEM_COST_FAMILIES
    is_top_k_sql = resolved_tag in (TOP_SPECIMENS_QTAG, TOP_OUTPUT_EVENTS_QTAG)
    is_vital_change = resolved_tag == VITAL_CHANGE_QTAG
    is_elapsed = is_elapsed_time_qtag(resolved_tag)
    patient = bool(re.search(r"patient\s+\d+", q)) or "subject_id" in q
    # "how many hours since patient 90663" is patient-specific, not population aggregate.
    population = not patient and any(
        w in q
        for w in (
            "how many",
            "count the",
            "count the number",
            "top five",
            "top 5",
            "top four",
            "top 4",
            "top three",
            "top 3",
            "frequently",
            "female patients",
            "male patients",
            "aged ",
        )
    )
    task_type = "patient_specific" if patient and not population else (
        "population_aggregate" if population else "other"
    )
    tables = list(
        dict.fromkeys(
            (family_hints.get("priority_tables") or [])
            + (task_mem.get("suggested_tables") or [])
            + _infer_tables_from_question(question, q_tag=q_tag, value=value)
        )
    )
    if not tables:
        tables = ["admissions", "patients"]
    use_sql = bool(family_hints.get("use_sql_interpreter")) or "sql" in q or "select " in q
    constraints = [
        "Use exact MIMIC-III CSV column names (prescriptions.DRUG not DRUG_NAME).",
        "GENDER values are f/m not female/male.",
        "Store final result in variable answer; end with TERMINATE when done.",
        "FilterDB conditions must not use extra quotes around values (use SHORT_TITLE=chf nos not SHORT_TITLE=\"chf nos\").",
        "For drug intake/route questions use prescriptions.DRUG + prescriptions.ROUTE, not d_items LABEL like / ITEMID loops.",
        "GetValue(..., COL, list) returns a Python list — iterate it directly; never .split() that list or a comma string.",
    ]
    if is_item_cost:
        constraints.append(
            "For item costs use one SQLInterpreter subquery with cost.event_type + cost.event_id; "
            "set answer = rows[0][0] from the result list — do not assign answer = entire SQL result tuple."
        )
        constraints.extend(
            [
                "Never answer item-specific cost with FilterDB(cost, HADM_ID in (...)) then GetValue(COST) — "
                "that returns many unrelated charges; use EVENT_TYPE and EVENT_ID instead.",
                "If GetValue returns many comma-separated COST values but the question asks for one lab/procedure cost, "
                "switch to SQLInterpreter with event_type and event_id join (see gold SQL pattern).",
            ]
        )
    elif is_top_k_sql:
        constraints.append(
            "Use reference_python SQL pattern; answer = [r[0] for r in rows] for top-k lists."
        )
    elif is_vital_change:
        constraints.extend(
            [
                "chartevents/labevents: use VALUENUM only (never GetValue(..., 'VALUE') on chartevents).",
                "datetime is pre-imported in the code header; do not forget datetime.strptime if sorting CHARTTIME.",
            ]
        )
    elif is_elapsed:
        constraints.append(
            "Use reference_python SQL only; do not use Python datetime.now() for hours/days passed."
        )
    elif use_sql:
        constraints.append(
            "Prefer SQLInterpreter when joins or dense_rank() top-k are simpler (needs mimic_iii.db under EHRAGENT_DATA_ROOT)."
        )
    else:
        constraints.append(
            "Prefer LoadDB + FilterDB + GetValue; SQLInterpreter only when joins are simpler (needs mimic_iii.db under EHRAGENT_DATA_ROOT)."
        )
    for sh in schema_hints_for_question(question):
        if sh not in constraints:
            constraints.append(sh)
    if resolved_tag:
        constraints.append(f"Question family (q_tag): {resolved_tag}")
    if family_hints.get("reference_python"):
        constraints.append("Use reference_python below (same pattern for all instances of this q_tag).")
    strategy_steps = _strategy_steps_for_question(question, tables)
    if family_hints.get("strategy_steps"):
        strategy_steps = list(family_hints["strategy_steps"]) + strategy_steps
    for p in (task_mem.get("pitfalls") or [])[:3]:
        fix = p.get("fix")
        if fix:
            constraints.append(str(fix))
    for skill in (task_mem.get("skills") or [])[:3]:
        desc = skill.get("description")
        if desc:
            constraints.append("Relevant skill: " + str(desc))
        for step in (skill.get("steps") or [])[:2]:
            if step not in strategy_steps:
                strategy_steps.append(str(step))
        for table in skill.get("tables") or []:
            if table not in tables:
                tables.append(str(table))
    for state in (task_mem.get("recent_active_states") or [])[:2]:
        failed = state.get("failed_filters") or []
        if failed:
            constraints.append("Avoid repeated failed filters: " + "; ".join(map(str, failed[:3])))
        src = state.get("final_answer_source")
        if src:
            constraints.append("Prior final answer source for similar task: " + str(src)[:160])
    if worldmm.get("beliefs"):
        constraints.append("Respect belief memory for active diagnoses/medications when filtering.")
    if resolved_tag == DRUG_ROUTE_QTAG:
        constraints.append(
            "Drug route: one LoadDB/FilterDB/GetValue(ROUTE) pass is enough; then TERMINATE."
        )
    plan = {
        "task_type": task_type,
        "priority_tables": tables[:6],
        "strategy_steps": strategy_steps[:6],
        "constraints": constraints[:16],
        "skip_patient_memory": (
            (task_type == "population_aggregate" and not worldmm.get("beliefs"))
            or is_elapsed
        ),
        "recovery_hints": _recovery_hints_for_plan(
            is_item_cost=is_item_cost,
            is_top_k_sql=is_top_k_sql,
            is_vital_change=is_vital_change,
            is_elapsed=is_elapsed,
            resolved_tag=resolved_tag,
        ),
        "use_sql_interpreter": use_sql,
        "reference_sql": family_hints.get("reference_sql", ""),
        "reference_python": family_hints.get("reference_python", ""),
        "question_family": family_hints.get("question_family", ""),
    }
    return _attach_controller_policy(question, plan, worldmm, task_mem)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def plan_from_memory(
    config: dict,
    question: str,
    *,
    worldmm_state: Dict[str, Any],
    task_memory: Dict[str, Any],
    ltm_summary: List[Dict[str, str]],
    max_tokens: int = 700,
    use_llm: bool = True,
    q_tag: Optional[str] = None,
    value: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not use_llm:
        plan = _default_plan(
            question, task_memory, worldmm_state, q_tag=q_tag, value=value
        )
        plan["planner_source"] = "heuristic"
        return plan

    user_msg = PLANNER_USER_TEMPLATE.format(
        question=question,
        worldmm_json=json.dumps(worldmm_state, ensure_ascii=False)[:4000],
        task_json=json.dumps(task_memory, ensure_ascii=False)[:2000],
        skill_json=json.dumps(task_memory.get("skills") or [], ensure_ascii=False)[:2500],
        state_json=json.dumps(task_memory.get("recent_active_states") or [], ensure_ascii=False)[:1500],
        ltm_json=json.dumps(ltm_summary, ensure_ascii=False)[:3000],
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = _chat_completion(config, messages, max_tokens=max_tokens)
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_object(raw)
        if parsed:
            base = _default_plan(
                question, task_memory, worldmm_state, q_tag=q_tag, value=value
            )
            base.update({k: v for k, v in parsed.items() if v is not None})
            base["planner_source"] = "llm"
            return _attach_controller_policy(question, base, worldmm_state, task_memory)
    except Exception as e:
        if _is_quota_error(e):
            print(
                "[MemoryAgent] planner LLM failed: API quota/balance (403). "
                "Recharge AIHubMix or use --planner_heuristic_only to skip planner API calls.",
                flush=True,
            )
        else:
            print(f"[MemoryAgent] planner LLM failed: {e}", flush=True)
    plan = _default_plan(
        question, task_memory, worldmm_state, q_tag=q_tag, value=value
    )
    plan["planner_source"] = "heuristic_fallback"
    return plan


def format_plan_for_prompt(plan: Dict[str, Any]) -> str:
    lines = ["### Memory Agent Plan (policy — follow before coding)"]
    lines.append(f"task_type: {plan.get('task_type', 'other')}")
    lines.append(f"context_mode: {plan.get('context_mode', 'neither')}")
    gates = plan.get("gates") or {}
    if gates:
        active = [name for name, enabled in gates.items() if enabled]
        lines.append("active_gates: " + ", ".join(active))
    budget = plan.get("retrieval_budget") or {}
    if budget:
        lines.append("retrieval_budget: " + ", ".join(f"{k}={v}" for k, v in budget.items()))
    controller = plan.get("controller") or {}
    if controller:
        lines.append("controller: " + ", ".join(f"{k}={v}" for k, v in controller.items()))
    if plan.get("question_family"):
        lines.append(f"question_family: {plan.get('question_family')}")
    tables = plan.get("priority_tables") or []
    if tables:
        lines.append("priority_tables: " + ", ".join(tables))
    for i, step in enumerate(plan.get("strategy_steps") or [], 1):
        lines.append(f"{i}. {step}")
    lines.append("constraints:")
    for c in plan.get("constraints") or []:
        lines.append(f"  - {c}")
    for h in plan.get("recovery_hints") or []:
        lines.append(f"recovery: {h}")
    if plan.get("use_sql_interpreter"):
        fam = plan.get("question_family") or ""
        if fam == TOP_SPECIMENS_QTAG:
            lines.append("prefer: SQLInterpreter on microbiologyevents.SPEC_TYPE_DESC with dense_rank top-k")
        elif fam == TOP_OUTPUT_EVENTS_QTAG:
            lines.append("prefer: SQLInterpreter on outputevents + d_items.LABEL with dense_rank top-k")
        elif fam in ITEM_COST_FAMILIES:
            lines.append(
                "prefer: SQLInterpreter for item costs (lab/procedure/prescription/diagnosis) via cost.event_type + cost.event_id"
            )
        elif fam == VITAL_CHANGE_QTAG:
            lines.append(
                "prefer: SQLInterpreter for vital change (last minus offset 1 VALUENUM on first ICU stay)"
            )
        elif is_elapsed_time_qtag(fam):
            lines.append(
                "prefer: SQLInterpreter for elapsed hours/days (strftime julianday vs current_time in DB)"
            )
        else:
            lines.append("prefer: SQLInterpreter when joins or ranking are simpler than FilterDB chains")
    ref_py = (plan.get("reference_python") or "").strip()
    if ref_py:
        lines.append("reference_python (copy this line):")
        lines.append(f"  {ref_py}")
    else:
        ref_sql = (plan.get("reference_sql") or "").strip()
        if ref_sql:
            lines.append("reference_sql:")
            lines.append(f"  {ref_sql}")
    return "\n".join(lines)


def format_worldmm_compact(state: Dict[str, Any]) -> str:
    if not state.get("has_timeline") and not state.get("beliefs"):
        return ""
    lines = ["### WorldMM Memory (structured)"]
    for b in state.get("beliefs") or []:
        tag = " [CRITICAL]" if b.get("critical") else ""
        lines.append(f"  belief {b.get('attr')}: {b.get('hyp')} (p={b.get('prob')}){tag}")
    for ev in state.get("episodic") or []:
        lines.append(f"  event: {ev}")
    for tr in state.get("semantic_triples") or []:
        if len(tr) >= 3:
            lines.append(f"  triple: {tr[0]} —{tr[1]}→ {tr[2]}")
    return "\n".join(lines) if len(lines) > 1 else ""
