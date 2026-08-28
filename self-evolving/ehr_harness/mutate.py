"""Propose harness mutations from traces. Never touches model weights."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from .spec import HarnessSpec
from .traces import QuestionTrace


def _norm_error(err: str, max_len: int = 160) -> str:
    t = re.sub(r"\s+", " ", (err or "").lower()).strip()
    return t[:max_len]


def _infer_fix(error: str, code: str = "") -> str:
    err = (error or "").lower()
    c = code or ""
    if "gender" in err and ("female" in err or "male" in err):
        return "Use GENDER=f or GENDER=m (never female/male)."
    if "column name" in err and "incorrect" in err:
        if "value" in err and "chartevents" in err:
            return "chartevents uses VALUENUM not VALUE."
        if "drug_name" in err:
            return "prescriptions use DRUG not DRUG_NAME."
        return "Match exact MIMIC CSV column names from the tool error message."
    if "datetime" in err and "not defined" in err:
        return "datetime is pre-imported; do not re-import; use datetime.strptime."
    if "hadm_id" in err and "cost" in c.lower():
        return (
            "Item cost: join via cost.event_type + cost.event_id → source ROW_ID; "
            "do not filter cost by HADM_ID alone."
        )
    if "indentation" in err or "unexpected indent" in err:
        return "Keep consistent indentation inside if/for blocks."
    return ""


def _skill_id(question: str, tables: List[str]) -> str:
    key = (question[:80] + "|" + ",".join(tables[:4])).lower()
    return "sk_" + hashlib.md5(key.encode()).hexdigest()[:10]


def _tables_from_code(code: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"LoadDB\(['\"](\w+)['\"]\)", code or "")))


def distill_successes_from_traces(
    task_memory: Dict[str, Any],
    traces: List[QuestionTrace],
    *,
    max_new_skills: int = 24,
) -> Tuple[Dict[str, Any], List[str]]:
    """Feed gold/judge PASS traces into shared skills+experiences for later optimization."""
    tm = copy.deepcopy(task_memory)
    skills = tm.setdefault("skills", {})
    experiences = tm.setdefault("experiences", {})
    notes: List[str] = []
    successes = [t for t in traces if t.passed and t.code_snippets]
    # Prefer longer verified code (usually more complete solutions)
    successes.sort(key=lambda t: len(t.code_snippets[-1] if t.code_snippets else ""), reverse=True)
    added = 0
    for tr in successes:
        if added >= max_new_skills:
            break
        code = (tr.code_snippets[-1] or "").strip()
        if not code or "answer" not in code.lower():
            continue
        tables = _tables_from_code(code)
        sid = _skill_id(tr.question, tables)
        prev = skills.get(sid, {})
        prev_succ = int(prev.get("successes", 0) or 0)
        skills[sid] = {
            "id": sid,
            "description": f"Verified success: {tr.question[:140]}",
            "tables": tables[:6] or list(prev.get("tables") or [])[:6],
            "steps": [ln.strip() for ln in code.splitlines() if ln.strip()][:8],
            "example_code": code[:3000],
            "successes": prev_succ + 1,
            "score": int(prev.get("score", 0) or 0) + 2,
            "source": "distill_pass",
            "last_error": "",
        }
        experiences[sid] = {
            "question": tr.question[:300],
            "code": code[:3000],
            "skill": skills[sid]["description"],
            "successes": skills[sid]["successes"],
            "score": float(skills[sid]["score"]),
            "source": "distill_pass",
        }
        notes.append(f"success→skill {sid}")
        added += 1

    if len(skills) > 64:
        ranked = sorted(skills.items(), key=lambda x: int(x[1].get("successes", 0) or 0), reverse=True)
        skills = dict(ranked[:64])
        notes.append("compress skills→64")
    if len(experiences) > 80:
        ranked_e = sorted(
            experiences.items(),
            key=lambda x: int(x[1].get("successes", 0) or 0),
            reverse=True,
        )
        experiences = dict(ranked_e[:80])
        notes.append("compress experiences→80")

    tm["skills"] = skills
    tm["experiences"] = experiences
    return tm, notes


def _classify_failure(error: str, code: str = "") -> str:
    e = (error or "").lower()
    c = (code or "").lower()
    if not e.strip():
        return "no_terminate"
    if any(k in e for k in ("syntax", "unterminated", "invalid syntax", "unexpected character", "eol while")):
        return "syntax"
    if "no such column" in e or ("incorrect" in e and "column" in e):
        return "bad_column"
    if "gender" in e or (("female" in e or "male" in e) and "gender" in (e + c)):
        return "gender"
    if "datetime" in e and ("import" in e or "not defined" in e):
        return "datetime"
    if "cost" in e or ("cost" in c and "hadm" in e):
        return "cost"
    if "maximum context length" in e or "insufficient_user_quota" in e:
        return "context" if "context" in e or "maximum" in e else "api"
    if "list index out of range" in e:
        return "empty_result"
    return "logic"


def _category_fix(category: str) -> str:
    mapping = {
        "syntax": "Rewrite a complete valid python cell with real newlines; set answer=... then TERMINATE.",
        "bad_column": "Use only exact MIMIC CSV column names from LoadDB / tool errors; do not invent columns.",
        "gender": "Filter with GENDER='f' or GENDER='m' only (never female/male).",
        "datetime": "datetime is pre-imported; use datetime.strptime — do not import datetime again.",
        "cost": "Item cost: join via cost.event_type + cost.event_id → source ROW_ID; never filter cost by HADM_ID alone.",
        "context": "Shorten the solution; fewer prints; finish with answer=... and TERMINATE.",
        "api": "Transient API failure; retry the same correct program without changing schema logic.",
        "empty_result": "Ensure the query returns ≥1 row before indexing; broaden filters or check IDs.",
        "no_terminate": "Produce a complete cell that sets answer=... and ends with TERMINATE.",
        "logic": "Keep working parts of prior code; change only the failing step; prefer SQLInterpreter for joins.",
    }
    return mapping.get(category) or mapping["logic"]


def _fail_knowledge_id(category: str, error: str, tables: List[str]) -> str:
    key = f"{category}|{','.join(tables[:3])}|{_norm_error(error, 80)}"
    return "fk_" + hashlib.md5(key.encode()).hexdigest()[:10]


def distill_failures_from_traces(
    task_memory: Dict[str, Any],
    traces: List[QuestionTrace],
    *,
    max_new: int = 24,
) -> Tuple[Dict[str, Any], List[str]]:
    """Extract structured diagnostic knowledge from FAIL/unfinished traces.

    Stores under ``fail_knowledge`` (and compact ``pitfalls`` with fix text).
    Does NOT dump raw Error: strings into constraint overlays.
    """
    tm = copy.deepcopy(task_memory)
    fail_kb = tm.setdefault("fail_knowledge", {})
    pitfalls = tm.setdefault("pitfalls", {})
    notes: List[str] = []

    fails = [t for t in traces if (not t.passed)]
    # Prefer failures with errors / code (more transferable)
    fails.sort(
        key=lambda t: (1 if t.last_error else 0, 1 if t.code_snippets else 0, len(t.last_error or "")),
        reverse=True,
    )
    added = 0
    for tr in fails:
        if added >= max_new:
            break
        code = (tr.code_snippets[-1] if tr.code_snippets else "") or ""
        err = tr.last_error or ("unfinished/no TERMINATE" if tr.unfinished or not tr.terminated else "")
        if not err and not code:
            continue
        cat = _classify_failure(err, code)
        if cat == "api":
            # quota / gateway failures are not transferable EHR knowledge
            continue
        tables = _tables_from_code(code)
        fid = _fail_knowledge_id(cat, err, tables)
        fix = _infer_fix(err, code) or _category_fix(cat)
        prev = fail_kb.get(fid, {})
        count = int(prev.get("count", 0) or 0) + 1
        fail_kb[fid] = {
            "id": fid,
            "kind": "diagnostic",
            "category": cat,
            "trigger": _norm_error(err, 200) or cat,
            "fix": fix,
            "tables": tables[:6] or list(prev.get("tables") or [])[:6],
            "question_hint": (tr.question or "")[:160],
            "bad_code_tail": code[-800:] if code else "",
            "count": count,
            "score": int(prev.get("score", 0) or 0) + 1,
            "source": "distill_fail",
            "unfinished": bool(tr.unfinished or (not tr.terminated)),
        }
        # Compact pitfall entry consumable by existing planner retrieve
        pkey = _norm_error(f"{cat}:{fix}", 160)
        pe = pitfalls.get(pkey, {"error": f"[{cat}] {err[:180]}", "fix": fix, "count": 0})
        pe["count"] = int(pe.get("count", 0) or 0) + 1
        pe["fix"] = fix
        pe["category"] = cat
        pe["source"] = "distill_fail"
        pitfalls[pkey] = pe
        notes.append(f"fail→{cat}/{fid}")
        added += 1

    if len(fail_kb) > 64:
        ranked = sorted(fail_kb.items(), key=lambda x: int(x[1].get("count", 0) or 0), reverse=True)
        fail_kb = dict(ranked[:64])
        notes.append("compress fail_knowledge→64")
    if len(pitfalls) > 64:
        ranked_p = sorted(pitfalls.items(), key=lambda x: int(x[1].get("count", 0) or 0), reverse=True)
        pitfalls = dict(ranked_p[:64])
        notes.append("compress pitfalls→64")

    tm["fail_knowledge"] = fail_kb
    tm["pitfalls"] = pitfalls
    return tm, notes


def distill_knowledge_from_traces(
    task_memory: Dict[str, Any],
    traces: List[QuestionTrace],
    *,
    max_new_skills: int = 24,
    max_new_fails: int = 24,
) -> Tuple[Dict[str, Any], List[str]]:
    """Bilateral knowledge: PASS → how-to skills; FAIL → diagnostic fail_knowledge."""
    tm, n1 = distill_successes_from_traces(task_memory, traces, max_new_skills=max_new_skills)
    tm, n2 = distill_failures_from_traces(tm, traces, max_new=max_new_fails)
    return tm, n1 + n2


def mutate_task_memory_from_traces(
    task_memory: Dict[str, Any],
    traces: List[QuestionTrace],
    *,
    max_new_pitfalls: int = 8,
    max_new_skills: int = 6,
) -> Tuple[Dict[str, Any], List[str]]:
    """Update procedural memory from traces: PASS how-to + FAIL diagnostics."""
    return distill_knowledge_from_traces(
        task_memory,
        traces,
        max_new_skills=max_new_skills,
        max_new_fails=max_new_pitfalls,
    )


def mutate_spec_from_metrics(
    spec: HarnessSpec,
    metrics: Dict[str, Any],
    *,
    prev_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[HarnessSpec, List[str]]:
    """Heuristic knob search over harness hyper-parameters (not weights)."""
    nxt = copy.deepcopy(spec)
    notes: List[str] = []
    unfinished = float(metrics.get("unfinished", 0) or 0)
    total = max(1, int(metrics.get("total", 1) or 1))
    unfinished_rate = unfinished / total
    sr = float(metrics.get("sr", 0.0) or 0.0)
    prev_sr = float((prev_metrics or {}).get("sr", sr) or sr)

    # Too many unfinished → compress context / fewer shots
    if unfinished_rate >= 0.25:
        if not nxt.compress_prompt:
            nxt.compress_prompt = True
            notes.append("enable compress_prompt")
        if nxt.num_shots > 2:
            nxt.num_shots = max(2, nxt.num_shots - 1)
            notes.append(f"num_shots→{nxt.num_shots}")
        if nxt.retrieval_budget.skills > 1:
            nxt.retrieval_budget.skills = max(1, nxt.retrieval_budget.skills - 1)
            notes.append("skills budget-1")
        if nxt.ltm_code_max_lines > 16:
            nxt.ltm_code_max_lines = max(16, nxt.ltm_code_max_lines - 4)
            notes.append(f"ltm_code_max_lines→{nxt.ltm_code_max_lines}")

    # Low unfinished but low SR → give slightly richer memory
    if unfinished_rate < 0.15 and sr < 45.0 and nxt.num_shots < 4:
        nxt.num_shots += 1
        notes.append(f"num_shots→{nxt.num_shots}")

    # Persist planner overlays from common failure modes already in constraint_overlays
    # Ablation: extra overlays on local Qwen raise unfinished; do not add here.

    if not notes:
        notes.append("noop")
    return nxt.bump("; ".join(notes)), notes


def apply_overlays_to_task_memory(spec: HarnessSpec, task_memory: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize harness constraint overlays into task-memory pitfalls (consumed by planner)."""
    tm = copy.deepcopy(task_memory)
    pitfalls = tm.setdefault("pitfalls", {})
    for i, c in enumerate(spec.constraint_overlays):
        key = _norm_error(f"harness_overlay_{i}_{c}")
        pitfalls[key] = {
            "error": f"[harness overlay] {c}",
            "fix": c,
            "count": max(1, int(pitfalls.get(key, {}).get("count", 1) or 1)),
            "last_ok": 1,
        }
    for fam, rules in (spec.family_overlays or {}).items():
        for j, rule in enumerate(rules):
            key = _norm_error(f"family_{fam}_{j}_{rule}")
            pitfalls[key] = {
                "error": f"[family:{fam}] {rule}",
                "fix": rule,
                "count": 2,
                "last_ok": 1,
            }
    tm["pitfalls"] = pitfalls
    # Always-on overlays for MedAgentMemoryAgent harness sidecar / task-memory fallback
    tm["_harness_constraint_overlays"] = list(spec.constraint_overlays or [])
    tm["_harness_family_overlays"] = dict(spec.family_overlays or {})
    tm["_harness_retrieval_budget"] = {
        "task_pitfalls": spec.retrieval_budget.task_pitfalls,
        "skills": spec.retrieval_budget.skills,
        "executable_traces": spec.retrieval_budget.executable_traces,
        "recent_states": spec.retrieval_budget.recent_states,
    }
    tm["_harness_ltm_code_max_lines"] = spec.ltm_code_max_lines
    return tm
