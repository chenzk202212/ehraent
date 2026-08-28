"""LLM meta-planner for harness evolution (frozen weights; plans scaffold changes).

Two planning layers:
  1) Outer: this module — analyze traces → propose HarnessPlan JSON
  2) Inner: EhrAgent Memory Planner — plan tools/tables before coding
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import chat_completion, parse_json_object
from .spec import HarnessSpec, RetrievalBudget
from .traces import QuestionTrace, summarize_traces

META_SYSTEM = """You are the meta-planner of a self-evolving EHR coding agent harness.
The backbone LLM weights are FROZEN. You may ONLY propose changes to the harness:
- knobs (num_shots, compress_prompt, planner_heuristic_only, ltm_code_max_lines, retrieval budgets)
- new_skills (reusable procedures distilled from SUCCESS / PASS traces)  << PRIMARY
- rationale (why these changes should raise SUCCESS RATE / accuracy)

Policy: distill BOTH verified PASS solutions (how-to skills) AND structured FAIL
diagnostics (category + trigger + fix). Do not dump raw Error: strings into overlays.

Priority: maximize accuracy (SR).
Never propose fine-tuning, LoRA, or weight updates.
Do NOT reduce num_shots unless hold evidence clearly shows context overflow is killing SR.
Output a single JSON object only."""

META_USER = """Current harness:
{harness_json}

Train metrics:
{metrics_json}

Failure cases (sample) — context only; do NOT copy these into overlays:
{failures_json}

Success cases (sample) — MUST distill reusable skills/example_code from these:
{successes_json}

Return JSON with keys:
{{
  "rationale": "short string focused on raising SR by keeping PASS skills",
  "knobs": {{
    "num_shots": 2|3|4,
    "compress_prompt": true|false,
    "planner_heuristic_only": true|false,
    "retry_on_fail": true|false,
    "ltm_code_max_lines": 16-32,
    "retrieval_budget": {{"task_pitfalls":int,"skills":int,"executable_traces":int,"recent_states":int}}
  }},
  "constraint_overlays_add": [],
  "family_overlays_add": {{}},
  "new_pitfalls": [],
  "new_skills": [{{"id":"...","description":"...","tables":["..."],"steps":["..."],"example_code":"..."}}]
}}

Rules:
- Preserve retry_on_fail / planner_heuristic_only from the current harness unless hold evidence justifies a change.
- Do NOT add constraint_overlays that restate Error: strings.
- Do NOT add new_pitfalls that restate Error: strings.
- If any success samples exist, new_skills MUST include at least 1 grounded in a success (copy/adapt its code_tail).
- At most 3 new skills. Leave overlay/pitfall lists empty unless a short schema hint is clearly reusable.
JSON:"""


def _trace_brief(tr: QuestionTrace) -> Dict[str, Any]:
    return {
        "id": tr.question_id,
        "question": (tr.question or "")[:220],
        "passed": tr.passed,
        "terminated": tr.terminated,
        "error": (tr.last_error or "")[:280],
        "code_tail": (tr.code_snippets[-1] if tr.code_snippets else "")[:500],
    }


def build_meta_plan_prompt(
    spec: HarnessSpec,
    traces: List[QuestionTrace],
    metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    metrics = metrics or summarize_traces(traces)
    fails = [_trace_brief(t) for t in traces if not t.passed][:8]
    wins = [_trace_brief(t) for t in traces if t.passed][:8]
    user = META_USER.format(
        harness_json=json.dumps(spec.to_dict(), ensure_ascii=False)[:3500],
        metrics_json=json.dumps(metrics, ensure_ascii=False),
        failures_json=json.dumps(fails, ensure_ascii=False)[:5000],
        successes_json=json.dumps(wins, ensure_ascii=False)[:3000],
    )
    return META_SYSTEM, user


def plan_harness_with_llm(
    spec: HarnessSpec,
    traces: List[QuestionTrace],
    *,
    model: str,
    metrics: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Ask LLM meta-planner for a structured HarnessPlan."""
    system, user = build_meta_plan_prompt(spec, traces, metrics=metrics)
    raw = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        max_tokens=1400,
    )
    parsed = parse_json_object(raw)
    if not parsed:
        raise RuntimeError(f"Meta-planner returned non-JSON: {raw[:400]}")
    parsed["_raw"] = raw
    parsed["planner_source"] = "llm"
    return parsed


def apply_meta_plan(
    spec: HarnessSpec,
    task_memory: Dict[str, Any],
    plan: Dict[str, Any],
) -> Tuple[HarnessSpec, Dict[str, Any], List[str]]:
    """Materialize an LLM HarnessPlan into spec + task_memory (no weights)."""
    notes: List[str] = []
    nxt = copy.deepcopy(spec)
    tm = copy.deepcopy(task_memory)

    knobs = plan.get("knobs") or {}
    if "num_shots" in knobs:
        nxt.num_shots = int(max(1, min(6, int(knobs["num_shots"]))))
        notes.append(f"num_shots→{nxt.num_shots}")
    if "compress_prompt" in knobs:
        nxt.compress_prompt = bool(knobs["compress_prompt"])
        notes.append(f"compress_prompt→{nxt.compress_prompt}")
    if "planner_heuristic_only" in knobs:
        nxt.planner_heuristic_only = bool(knobs["planner_heuristic_only"])
        notes.append(f"planner_heuristic_only→{nxt.planner_heuristic_only}")
    # else: preserve parent planner_heuristic_only (paper Memory Agent may keep heuristic)
    if "ltm_code_max_lines" in knobs:
        nxt.ltm_code_max_lines = int(max(12, min(48, int(knobs["ltm_code_max_lines"]))))
        notes.append(f"ltm_code_max_lines→{nxt.ltm_code_max_lines}")
    if "retry_on_fail" in knobs:
        nxt.retry_on_fail = bool(knobs["retry_on_fail"])
        notes.append(f"retry_on_fail→{nxt.retry_on_fail}")
    # else: preserve parent retry_on_fail (failure-triggered harness stays on)

    rb = knobs.get("retrieval_budget") or {}
    if isinstance(rb, dict) and rb:
        cur = nxt.retrieval_budget
        nxt.retrieval_budget = RetrievalBudget(
            task_pitfalls=int(rb.get("task_pitfalls", cur.task_pitfalls)),
            skills=int(rb.get("skills", cur.skills)),
            executable_traces=int(rb.get("executable_traces", cur.executable_traces)),
            recent_states=int(rb.get("recent_states", cur.recent_states)),
        )
        notes.append("retrieval_budget updated")

    skipped_ov = 0
    for c in plan.get("constraint_overlays_add") or []:
        c = str(c).strip()
        low = c.lower()
        if not c or c in nxt.constraint_overlays:
            continue
        if low.startswith("error:") or "potential reasons" in low:
            skipped_ov += 1
            continue
        # Do not grow overlays; PASS skills are the learning signal.
        skipped_ov += 1
    if skipped_ov:
        notes.append(f"skip overlays x{skipped_ov}")

    fam_add = plan.get("family_overlays_add") or {}
    if isinstance(fam_add, dict) and fam_add:
        notes.append("skip family overlays")

    pitfalls = tm.setdefault("pitfalls", {})
    n_skip_pit = len(plan.get("new_pitfalls") or [])
    if n_skip_pit:
        notes.append(f"skip pitfalls x{n_skip_pit}")

    skills = tm.setdefault("skills", {})
    experiences = tm.setdefault("experiences", {})
    for item in (plan.get("new_skills") or [])[:3]:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip() or None
        if not sid:
            continue
        skills[sid] = {
            "id": sid,
            "description": str(item.get("description") or "")[:300],
            "tables": list(item.get("tables") or [])[:6],
            "steps": [str(s) for s in (item.get("steps") or [])][:6],
            "example_code": str(item.get("example_code") or "")[:2500],
            "successes": int(skills.get(sid, {}).get("successes", 0) or 0) + 1,
            "score": int(skills.get(sid, {}).get("score", 0) or 0) + 1,
        }
        experiences[sid] = {
            "question": skills[sid]["description"],
            "code": skills[sid]["example_code"],
            "skill": skills[sid]["description"],
            "successes": skills[sid]["successes"],
        }
        notes.append(f"skill(llm)+ {sid}")

    rationale = str(plan.get("rationale") or "").strip()
    bump_note = rationale or "; ".join(notes[:6]) or "llm meta-plan"
    nxt = nxt.bump(bump_note)
    tm["pitfalls"] = pitfalls
    tm["skills"] = skills
    tm["experiences"] = experiences
    return nxt, tm, notes
