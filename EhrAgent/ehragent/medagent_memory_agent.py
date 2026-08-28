"""EHR Memory Agent: WorldMM state + task memory drive planner policy, then executor runs code."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


def _looks_population_question(question: str) -> bool:
    q = question.lower()
    if re.search(r"patient\s+\d+", q):
        return False
    return any(
        w in q
        for w in (
            "how many",
            "count the",
            "top five",
            "top 5",
            "top four",
            "top 4",
            "frequently prescribed",
            "frequently",
            "female patients",
            "male patients",
            "aged ",
        )
    )

from medagent_worldmm import MedAgentWorldMM
from memory_planner import format_plan_for_prompt, format_worldmm_compact, plan_from_memory
from task_memory import TaskMemoryStore


def _truncate_code(code: str, max_lines: int = 28) -> str:
    lines = (code or "").splitlines()
    if len(lines) <= max_lines:
        return code
    head = lines[: max_lines - 2]
    return "\n".join(head) + f"\n# ... ({len(lines) - max_lines} lines truncated in memory agent LTM) ..."


def _format_skill_memory(skills: List[Dict[str, Any]]) -> str:
    if not skills:
        return ""
    lines = ["### Skill Memory (reusable procedures — apply when relevant)"]
    for skill in skills[:5]:
        sid = skill.get("id") or "skill"
        desc = skill.get("description") or ""
        score = skill.get("score", 0)
        lines.append(f"- {sid} score={score}: {desc}")
        tables = skill.get("tables") or []
        if tables:
            lines.append("  tables: " + ", ".join(map(str, tables[:6])))
        for step in (skill.get("steps") or [])[:3]:
            lines.append("  step: " + str(step))
        if skill.get("last_error"):
            lines.append("  avoid: " + str(skill.get("last_error"))[:180])
    return "\n".join(lines)


class MedAgentMemoryAgent(MedAgentWorldMM):
    """
    Memory agent loop (per question):
      1. retrieve structured WorldMM + task memory + LTM summary
      2. planner LLM -> MemoryPlan (tables, constraints, strategy)
      3. executor prompt = plan + compact memory + compressed examples
      4. after run: update task memory + WorldMM belief from trace
    """

    def __init__(
        self,
        *,
        task_memory_path: Optional[str] = None,
        planner_max_tokens: int = 700,
        ltm_code_max_lines: int = 28,
        skip_llm_knowledge_when_planned: bool = True,
        planner_heuristic_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._task_memory = TaskMemoryStore(persist_path=task_memory_path)
        self._planner_max_tokens = planner_max_tokens
        self._ltm_code_max_lines = ltm_code_max_lines
        self._skip_llm_knowledge_when_planned = skip_llm_knowledge_when_planned
        self._planner_heuristic_only = planner_heuristic_only
        self._current_plan: Dict[str, Any] = {}
        self._worldmm_state: Dict[str, Any] = {}
        self._last_error: str = ""
        self._active_state: Dict[str, Any] = {}
        self._current_task_ctx: Dict[str, Any] = {}
        self._harness_overlays: List[str] = []
        self._harness_family_overlays: Dict[str, List[str]] = {}
        self._harness_retrieval_budget: Dict[str, int] = {}
        self._failure_boost: bool = False
        self._failure_error: str = ""
        self._failure_code: str = ""
        self._failure_category: str = ""
        self._failure_extra_constraints: List[str] = []
        self._failure_repaired_code: str = ""
        self._load_harness_sidecar(task_memory_path)

    def _load_harness_sidecar(self, task_memory_path: Optional[str]) -> None:
        """Consume self-evolving harness.json next to .task_memory.json (knobs + overlays)."""
        if not task_memory_path:
            return
        harness_path = os.path.join(os.path.dirname(os.path.abspath(task_memory_path)), "harness.json")
        if not os.path.isfile(harness_path):
            # Also accept overlays stashed inside task memory by the outer harness.
            try:
                raw = getattr(self._task_memory, "persist_path", None)
                if raw and os.path.isfile(raw):
                    with open(raw, encoding="utf-8") as f:
                        tm = json.load(f)
                    self._harness_overlays = list(tm.get("_harness_constraint_overlays") or [])
                    self._harness_family_overlays = dict(tm.get("_harness_family_overlays") or {})
                    self._harness_retrieval_budget = dict(tm.get("_harness_retrieval_budget") or {})
                    if tm.get("_harness_ltm_code_max_lines"):
                        self._ltm_code_max_lines = int(tm["_harness_ltm_code_max_lines"])
            except Exception:
                pass
            return
        try:
            with open(harness_path, encoding="utf-8") as f:
                h = json.load(f)
            self._harness_overlays = [str(x) for x in (h.get("constraint_overlays") or []) if str(x).strip()]
            fam = h.get("family_overlays") or {}
            if isinstance(fam, dict):
                self._harness_family_overlays = {
                    str(k): [str(r) for r in (v or []) if str(r).strip()] for k, v in fam.items()
                }
            rb = h.get("retrieval_budget") or {}
            if isinstance(rb, dict):
                self._harness_retrieval_budget = {str(k): int(v) for k, v in rb.items() if str(v).isdigit() or isinstance(v, int)}
            if h.get("ltm_code_max_lines"):
                self._ltm_code_max_lines = int(h["ltm_code_max_lines"])
            print(
                f"[MemoryAgent] harness sidecar v{h.get('version')} "
                f"overlays={len(self._harness_overlays)} ltm_lines={self._ltm_code_max_lines}",
                flush=True,
            )
        except Exception as e:
            print(f"[MemoryAgent] harness sidecar load failed: {e}", flush=True)

    def _apply_harness_to_plan(self, plan: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Always inject harness overlays into the Memory Plan (not retrieval-gated)."""
        out = dict(plan or {})
        constraints = list(out.get("constraints") or [])
        for c in self._harness_overlays:
            if c not in constraints:
                constraints.append(c)
        q = (question or "").lower()
        for fam, rules in (self._harness_family_overlays or {}).items():
            if fam.lower() in q or any(tok and tok in q for tok in re.findall(r"[a-z0-9_]+", fam.lower()) if len(tok) > 3):
                for r in rules:
                    if r not in constraints:
                        constraints.append(r)
        out["constraints"] = constraints[:16]
        if self._harness_retrieval_budget:
            budget = dict(out.get("retrieval_budget") or {})
            budget.update(self._harness_retrieval_budget)
            out["retrieval_budget"] = budget
        return out

    def clear_failure_harness(self) -> None:
        self._failure_boost = False
        self._failure_error = ""
        self._failure_code = ""
        self._failure_category = ""
        self._failure_extra_constraints = []
        self._failure_repaired_code = ""

    def _tight_context(self) -> bool:
        """Local 8k vLLM (and similar) need aggressive prompt budgets."""
        env = (os.environ.get("EHRAGENT_TIGHT_CONTEXT") or "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        try:
            bu = str((self.config_list or [{}])[0].get("base_url") or "")
            return "127.0.0.1" in bu or "localhost" in bu
        except Exception:
            return False

    @staticmethod
    def _classify_failure(error: str, code: str) -> str:
        e = (error or "").lower()
        c = (code or "").lower()
        if not e.strip():
            return "no_terminate"
        if any(k in e for k in ("syntax", "unterminated", "invalid syntax", "unexpected character", "eol while")):
            return "syntax"
        if "no such column" in e or ("incorrect" in e and "column" in e):
            return "bad_column"
        if "gender" in e or "female" in e or "male" in e:
            return "gender"
        if "datetime" in e and ("import" in e or "not defined" in e):
            return "datetime"
        if "cost" in e or "cost" in c:
            return "cost"
        if "maximum context length" in e:
            return "context"
        return "logic"

    @staticmethod
    def _deterministic_code_repair(code: str, error: str) -> str:
        """Cheap, reversible mechanical fixes before asking the LLM to patch."""
        c = code or ""
        if not c.strip():
            return c
        # vLLM/Hermes often emits literal \\n instead of newlines
        if "\\n" in c and c.count("\n") <= 1:
            c = re.sub(r"\\+n", "\n", c)
        if "\\_" in c:
            c = re.sub(r"\\+_", "\n", c)
        err = (error or "").lower()
        if "gender" in err or "female" in c.lower() or "male" in c.lower():
            c = re.sub(r"(['\"])female\1", r"\1f\1", c, flags=re.IGNORECASE)
            c = re.sub(r"(['\"])male\1", r"\1m\1", c, flags=re.IGNORECASE)
            c = re.sub(r"\bGENDER\s*=\s*['\"]female['\"]", "GENDER='f'", c, flags=re.IGNORECASE)
            c = re.sub(r"\bGENDER\s*=\s*['\"]male['\"]", "GENDER='m'", c, flags=re.IGNORECASE)
        if "datetime" in err:
            c = re.sub(r"(?m)^\s*import\s+datetime\s*$", "", c)
            c = re.sub(r"(?m)^\s*from\s+datetime\s+import\s+.+$", "", c)
        if "valuenum" in err or ("value" in err and "chartevents" in (c + err).lower()):
            c = re.sub(r"\bGetValue\(([^,]+),\s*['\"]VALUE['\"]\)", r"GetValue(\1, 'VALUENUM')", c)
            c = re.sub(r"\bVALUE\b", "VALUENUM", c)
        return c.strip()

    def _category_fix_lines(self, category: str) -> List[str]:
        mapping = {
            "syntax": [
                "Rewrite the whole cell cleanly: real newlines, matched quotes/parens, no backslash-n literals.",
                "Call the python tool once with a complete valid program; then set answer and TERMINATE.",
            ],
            "bad_column": [
                "Use ONLY column names shown in the tool error / LoadDB schema; do not invent columns.",
                "If a column is missing, LoadDB the table again and pick a real field name.",
            ],
            "gender": ["Filter gender with GENDER='f' or GENDER='m' only (never female/male)."],
            "datetime": ["datetime is pre-imported; use datetime.strptime — do not import datetime again."],
            "cost": [
                "Item cost: join via cost.event_type + cost.event_id to source ROW_ID; never filter cost by HADM_ID alone.",
            ],
            "context": ["Write a shorter solution: fewer prints, no huge intermediate dumps, finish with answer=..."],
            "no_terminate": [
                "Previous attempt did not finish. Produce a complete cell that sets answer=... and stop with TERMINATE.",
            ],
            "logic": [
                "Keep the working parts of the prior code; change only the failing step.",
                "Prefer SQLInterpreter for joins/aggregations; set answer then TERMINATE.",
            ],
        }
        return list(mapping.get(category) or mapping["logic"])

    def arm_failure_harness(self, *, last_error: str = "", code: str = "") -> List[str]:
        """Enable failure-triggered harness for a targeted code-repair retry."""
        self._failure_boost = True
        raw_code = (code or getattr(self, "code", "") or "")
        self._failure_error = (last_error or self._last_error or "")[:500]
        self._failure_code = raw_code[:2500]
        self._failure_category = self._classify_failure(self._failure_error, self._failure_code)
        self._failure_repaired_code = self._deterministic_code_repair(self._failure_code, self._failure_error)[:2500]

        extras: List[str] = [
            f"FAILURE RETRY [{self._failure_category}]: patch the broken code below; do not restart from scratch unless necessary.",
            "Emit one complete fixed python cell, assign answer=..., then TERMINATE.",
        ]
        extras.extend(self._category_fix_lines(self._failure_category))
        try:
            ctx = self._task_memory.retrieve_for_question(
                (self.question or "") + " " + (self._failure_error or "")
            )
            for p in (ctx.get("pitfalls") or [])[:2]:
                fix = (p.get("fix") or "").strip()
                if fix:
                    extras.append("Known fix: " + fix[:140])
            for sk in (ctx.get("skills") or [])[:1]:
                desc = (sk.get("description") or "").strip()
                if desc:
                    extras.append("Reuse skill: " + desc[:100])
        except Exception:
            pass
        self._failure_extra_constraints = extras[:8]
        print(
            f"[MemoryAgent] failure-harness armed cat={self._failure_category} "
            f"extras={len(self._failure_extra_constraints)} "
            f"err={self._failure_error[:100]!r} code_len={len(self._failure_code)}",
            flush=True,
        )
        return list(self._failure_extra_constraints)

    def _build_failure_retry_message(self, question: str) -> str:
        """Compact repair prompt: question + error + broken code (no few-shot bloat)."""
        cat = self._failure_category or "logic"
        err = (self._failure_error or "(no tool error captured — previous run unfinished or wrong answer)").strip()
        broken = (self._failure_repaired_code or self._failure_code or "").strip()
        if not broken:
            broken = "# (no prior code captured)"
        if len(broken) > 1800:
            broken = broken[:1800] + "\n# ... truncated ..."
        lines = [
            "### Failure-Triggered Repair (maximize correctness)",
            f"category: {cat}",
            f"Question: {question}",
            "",
            "Previous error:",
            err[:450],
            "",
            "Broken code to PATCH (start from this; do not ignore it):",
            "```python",
            broken,
            "```",
            "",
            "Repair rules:",
        ]
        for i, rule in enumerate(self._failure_extra_constraints[:6], 1):
            lines.append(f"{i}. {rule}")
        lines.extend(
            [
                "",
                "Now call the python tool with the FIXED full program. "
                "Store the final result in variable answer. Reply TERMINATE when done.",
            ]
        )
        msg = "\n".join(lines)
        hard_cap = int(os.environ.get("EHRAGENT_MAX_INIT_CHARS", "9000") or 9000)
        if hard_cap > 0 and len(msg) > hard_cap:
            msg = msg[:hard_cap] + "\n# ... repair prompt truncated ..."
        return msg

    @property
    def task_memory(self) -> TaskMemoryStore:
        return self._task_memory

    @property
    def current_plan(self) -> Dict[str, Any]:
        return self._current_plan

    def _ltm_summary(self, query: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for idx in self._nearest_example_indices(query):
            item = self.memory[idx]
            out.append(
                {
                    "question": (item.get("question") or "")[:200],
                    "knowledge": (item.get("knowledge") or "")[:400],
                    "code_head": _truncate_code(item.get("code") or "", self._ltm_code_max_lines)[:600],
                }
            )
        for exp in self._task_memory.retrieve_experiences(query, top_k=3):
            out.append(
                {
                    "question": (exp.get("question") or "")[:200],
                    "knowledge": (exp.get("skill") or "")[:400],
                    "code_head": _truncate_code(exp.get("code") or "", self._ltm_code_max_lines)[:600],
                }
            )
        for skill in self._task_memory.retrieve_skills(query, top_k=3):
            out.append(
                {
                    "question": "Reusable skill: " + (skill.get("id") or ""),
                    "knowledge": (skill.get("description") or "")[:400],
                    "code_head": _truncate_code(skill.get("example_code") or "", self._ltm_code_max_lines)[:600],
                }
            )
        return out

    def _build_worldmm_state(self, question: str) -> Dict[str, Any]:
        # ablation: skip WorldMM context entirely
        if getattr(self, "_no_worldmm_context", False):
            return {"has_timeline": False, "beliefs": [], "episodic": [], "semantic_triples": []}
        p = self._worldmm_timeline_path
        if not p or not os.path.isfile(p):
            return {"has_timeline": False, "beliefs": [], "episodic": [], "semantic_triples": []}
        try:
            self._ensure_worldmm()
            assert self._ehrmm is not None
            self._ehrmm.load_mimic_json(p)
            with open(p, encoding="utf-8") as f:
                until = int(json.load(f).get("until_time", 0))
            if hasattr(self._ehrmm, "structured_state_for_agent"):
                return self._ehrmm.structured_state_for_agent(question, until_time=until)
            ctx = self._ehrmm.memory_context_for_prompt(question, until_time=until)
            return {"has_timeline": True, "raw_context": ctx[:800], "beliefs": [], "episodic": [], "semantic_triples": []}
        except Exception as e:
            print(f"[MemoryAgent] WorldMM state failed: {e}", flush=True)
            return {"has_timeline": False, "error": str(e)}

    def _worldmm_for_planner(self, question: str) -> Dict[str, Any]:
        wm = dict(self._worldmm_state)
        if _looks_population_question(question):
            wm = {**wm, "beliefs": [], "episodic": [], "semantic_triples": [], "skip_patient_memory": True}
        return wm

    def _run_planner(
        self,
        question: str,
        task_ctx: Dict[str, Any],
        *,
        q_tag: Optional[str] = None,
        value: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        plan = plan_from_memory(
            self.config_list[0],
            question,
            worldmm_state=self._worldmm_for_planner(question),
            task_memory=task_ctx,
            ltm_summary=self._ltm_summary(question),
            max_tokens=self._planner_max_tokens,
            use_llm=not self._planner_heuristic_only,
            q_tag=q_tag,
            value=value,
        )
        if plan.get("skip_patient_memory"):
            plan["priority_tables"] = list(
                dict.fromkeys((plan.get("priority_tables") or []) + (task_ctx.get("suggested_tables") or []))
            )[:6]
        return plan

    def retrieve_examples_compressed(self, query: str) -> str:
        selected = self._nearest_example_indices(query)
        blocks = []
        seen_questions = set()
        for i in selected:
            item = self.memory[i]
            seen_questions.add((item.get("question") or "").strip().lower())
            code = _truncate_code(item.get("code") or "", self._ltm_code_max_lines)
            blocks.append(
                "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
                    item.get("question", ""),
                    item.get("knowledge", ""),
                    code,
                )
            )
        gates = self._current_plan.get("gates") or {}
        budget = self._current_plan.get("retrieval_budget") or {}
        if not gates.get("read_dm", True):
            return "\n".join(blocks)
        exp_k = min(2, int(budget.get("executable_traces", 2) or 0))
        for exp in self._task_memory.retrieve_experiences(query, top_k=exp_k):
            if not exp.get("code") or int(exp.get("successes", 0) or 0) <= 0:
                continue
            q_key = (exp.get("question") or "").strip().lower()
            if q_key in seen_questions:
                continue
            blocks.append(
                "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
                    exp.get("question", ""),
                    "Dynamic experience memory: " + (exp.get("skill") or ""),
                    _truncate_code(exp.get("code") or "", self._ltm_code_max_lines),
                )
            )
            seen_questions.add(q_key)
        skill_k = min(2, int(budget.get("skills", 2) or 0))
        for skill in self._task_memory.retrieve_skills(query, top_k=skill_k):
            if not skill.get("example_code") or int(skill.get("successes", 0) or 0) <= 0:
                continue
            sid = "skill:" + (skill.get("id") or "")
            if sid in seen_questions:
                continue
            blocks.append(
                "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
                    "Reusable EHR skill " + (skill.get("id") or ""),
                    "Skill memory: " + (skill.get("description") or ""),
                    _truncate_code(skill.get("example_code") or "", self._ltm_code_max_lines),
                )
            )
            seen_questions.add(sid)
        return "\n".join(blocks)

    def _apply_retrieval_budget(self, task_ctx: Dict[str, Any]) -> Dict[str, Any]:
        budget = self._current_plan.get("retrieval_budget") or {}
        bounded = dict(task_ctx)
        limits = {
            "pitfalls": "task_pitfalls",
            "skills": "skills",
            "dynamic_experiences": "executable_traces",
            "executable_memory": "executable_traces",
            "recent_active_states": "recent_states",
        }
        for key, budget_key in limits.items():
            value = bounded.get(key)
            if isinstance(value, list):
                bounded[key] = value[: max(0, int(budget.get(budget_key, len(value)) or 0))]
        if not (self._current_plan.get("gates") or {}).get("read_dm", True):
            for key in limits:
                bounded[key] = []
            bounded["suggested_tables"] = []
        return bounded

    def _reset_active_state(self, question: str) -> None:
        self._active_state = {
            "question": question[:300],
            "tables_checked": [],
            "columns_verified": [],
            "failed_filters": [],
            "sql_tables": [],
            "tool_calls": 0,
            "error_count": 0,
            "final_answer_source": "",
        }

    @staticmethod
    def _dedupe_extend(target: List[str], values: List[str], limit: int = 16) -> List[str]:
        return list(dict.fromkeys((target or []) + [v for v in values if v]))[:limit]

    def _update_active_state_from_code(self, code: str, output: str, ok: bool) -> None:
        state = self._active_state
        state["tool_calls"] = int(state.get("tool_calls", 0) or 0) + 1
        tables = re.findall(r"LoadDB\(['\"](\w+)['\"]\)", code or "")
        sql_tables = re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", code or "", flags=re.IGNORECASE)
        state["tables_checked"] = self._dedupe_extend(state.get("tables_checked", []), tables)
        state["sql_tables"] = self._dedupe_extend(state.get("sql_tables", []), [t.lower() for t in sql_tables])
        cols = re.findall(r"GetValue\([^,]+,\s*['\"]([A-Za-z0-9_]+)['\"]", code or "")
        cols += re.findall(r"\b(?:select|where|order by|group by)\s+([A-Za-z_][A-Za-z0-9_]*)", code or "", flags=re.IGNORECASE)
        state["columns_verified"] = self._dedupe_extend(state.get("columns_verified", []), cols)
        if not ok or "error" in (output or "").lower():
            state["error_count"] = int(state.get("error_count", 0) or 0) + 1
            filters = re.findall(r"FilterDB\([^,]+,\s*['\"]([^'\"]+)['\"]", code or "")
            state["failed_filters"] = self._dedupe_extend(state.get("failed_filters", []), filters, limit=10)
        if "answer" in (code or "").lower() and ok and output and "error" not in output.lower():
            if sql_tables:
                state["final_answer_source"] = "SQLInterpreter:" + ",".join(
                    list(dict.fromkeys([t.lower() for t in sql_tables]))[:4]
                )
            elif tables:
                state["final_answer_source"] = "tool:" + ",".join(list(dict.fromkeys(tables))[:4])

    def generate_init_message(self, **context):
        if self.dataset == "mimic_iii":
            from prompts_mimic import EHRAgent_Message_Prompt
        else:
            from prompts_eicu import EHRAgent_Message_Prompt

        q = context["message"]
        self.question = q
        self._last_error = ""
        self._last_py_cell_norm = ""
        self._last_py_output = ""
        self._force_terminate_after_exec = False
        self._reset_active_state(q)

        # Targeted repair: skip few-shot + planner so the error/code signal is not drowned.
        if self._failure_boost:
            self._current_plan = {
                "task_type": "failure_repair",
                "planner_source": "failure_harness",
                "context_mode": "repair",
                "gates": {"read_dm": True, "read_psm": False, "use_sql": True},
                "strategy_steps": ["Patch broken code using the error and category rules."],
                "constraints": list(self._failure_extra_constraints)[:8],
                "retrieval_budget": {
                    "task_pitfalls": 0,
                    "skills": 0,
                    "executable_traces": 0,
                    "recent_states": 0,
                },
            }
            self._current_task_ctx = {}
            self.knowledge = "\n".join(self._failure_extra_constraints[:6])
            init_message = self._build_failure_retry_message(q)
            if getattr(self, "_memory_trace_flag", False):
                print(
                    f"[MemoryAgent] REPAIR mode cat={self._failure_category} "
                    f"code_len={len(self._failure_code)} err_len={len(self._failure_error)} "
                    f"prompt_chars={len(init_message)}",
                    flush=True,
                )
            return init_message

        self._worldmm_state = self._build_worldmm_state(q)
        task_ctx = self._task_memory.retrieve_for_question(q)
        self._current_plan = self._run_planner(
            q,
            task_ctx,
            q_tag=context.get("q_tag"),
            value=context.get("value"),
        )
        self._current_plan = self._apply_harness_to_plan(self._current_plan, q)
        tight = self._tight_context()
        task_ctx = self._apply_retrieval_budget(task_ctx)
        self._current_task_ctx = task_ctx

        plan_block = format_plan_for_prompt(self._current_plan)
        world_block = ""
        gates = self._current_plan.get("gates") or {}
        if gates.get("read_psm") and not self._current_plan.get("skip_patient_memory"):
            world_block = format_worldmm_compact(self._worldmm_state)

        if self._skip_llm_knowledge_when_planned and self._current_plan.get("strategy_steps"):
            knowledge_parts = [plan_block]
            if world_block:
                knowledge_parts.append(world_block)
            skill_block = _format_skill_memory(task_ctx.get("skills") or []) if gates.get("read_dm", True) else ""
            if skill_block:
                knowledge_parts.append(skill_block)
            self.knowledge = "\n\n".join(knowledge_parts)
        else:
            llm_knowledge = self.retrieve_knowledge(self.config_list[0], q)
            skill_block = _format_skill_memory(task_ctx.get("skills") or []) if gates.get("read_dm", True) else ""
            self.knowledge = "\n\n".join(filter(None, [plan_block, world_block, skill_block, llm_knowledge]))

        examples = self.retrieve_examples_compressed(q)
        if getattr(self, "compress_prompt", False) or tight:
            from prompt_compressor import compress_examples, compress_knowledge

            if tight:
                ex_chars, kn_chars = 3200, 2200
                ex_blocks = min(int(self.num_shots or 2), 3)
                kn_blocks = 5
            else:
                ex_chars, kn_chars = 8500, 6500
                ex_blocks = max(2, self.num_shots)
                kn_blocks = 6
            examples = compress_examples(examples, q, max_blocks=ex_blocks, max_chars=ex_chars)
            self.knowledge = compress_knowledge(self.knowledge, q, max_blocks=kn_blocks, max_chars=kn_chars)
        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=self.knowledge, question=q)
        hard_cap = int(os.environ.get("EHRAGENT_MAX_INIT_CHARS", "12000" if tight else "0") or 0)
        if hard_cap > 0 and len(init_message) > hard_cap:
            init_message = init_message[:hard_cap] + "\n# ... init prompt truncated for context ..."

        if getattr(self, "_memory_trace_flag", False):
            dyn = task_ctx.get("dynamic_experiences") or []
            skills = task_ctx.get("skills") or []
            print(
                f"[MemoryAgent] plan task_type={self._current_plan.get('task_type')} "
                f"source={self._current_plan.get('planner_source')} "
                f"mode={self._current_plan.get('context_mode')} "
                f"gates={self._current_plan.get('gates')} "
                f"controller={self._current_plan.get('controller')} "
                f"tables={self._current_plan.get('priority_tables')} "
                f"beliefs={len(self._worldmm_state.get('beliefs') or [])} "
                f"pitfalls={len(task_ctx.get('pitfalls') or [])} "
                f"dynamic_exp={len(dyn)} "
                f"skills={len(skills)}",
                flush=True,
            )
        return init_message

    def execute_function(self, func_call, **kwargs):
        is_ok, result = super().execute_function(func_call, **kwargs)
        content = str(result.get("content", ""))
        if not is_ok or "error" in content.lower():
            self._last_error = content[:500]
        self._update_active_state_from_code(self.code, content, bool(is_ok))
        if self._ehrmm is not None:
            self._ehrmm._current_code = self.code
        return is_ok, result

    def trace_is_valid(self, terminated: bool) -> bool:
        """Execution-only write signal; deliberately independent of gold answers."""
        state = self._active_state or {}
        return bool(
            terminated
            and not self._last_error
            and int(state.get("tool_calls", 0) or 0) > 0
            and state.get("final_answer_source")
            and "answer" in (self.code or "").lower()
        )

    def finish_question(self, question: str, execution_valid: bool) -> None:
        gates = self._current_plan.get("gates") or {}
        if gates.get("update", True) and gates.get("write_dm", True):
            self._task_memory.record_from_trace(
                question,
                success=execution_valid,
                code=self.code,
                last_error=self._last_error,
                plan=self._current_plan,
                active_state=self._active_state,
            )
        if gates.get("update", True) and gates.get("write_psm", False) and self._ehrmm is not None:
            self._ehrmm.update_from_experience(
                question=question,
                success=execution_valid,
                code=self.code,
                error=self._last_error,
            )

    def set_memory_trace(self, enabled: bool) -> None:
        self._memory_trace_flag = enabled
