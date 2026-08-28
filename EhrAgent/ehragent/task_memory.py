"""Cross-question procedural memory for the EHR memory agent (tool pitfalls, table hints)."""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from mimic_schema import fix_for_column_error


def _norm_key(text: str, max_len: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "").lower().strip())
    return t[:max_len]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


class TaskMemoryStore:
    """Task-level memory: survives across patients; complements WorldMM patient memory."""

    MAX_PITFALLS = 48
    MAX_HINTS = 32
    MAX_EXPERIENCES = 80
    MAX_SKILLS = 64
    MAX_ACTIVE_STATES = 120

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self.persist_path = persist_path
        self._pitfalls: Dict[str, Dict[str, Any]] = {}
        self._table_hints: Dict[str, List[str]] = defaultdict(list)
        self._experiences: Dict[str, Dict[str, Any]] = {}
        self._skills: Dict[str, Dict[str, Any]] = {}
        self._active_states: List[Dict[str, Any]] = []
        self._version = 0
        if persist_path and os.path.isfile(persist_path):
            self.load(persist_path)

    def load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._pitfalls = data.get("pitfalls", {})
            self._table_hints = defaultdict(list, data.get("table_hints", {}))
            self._experiences = data.get("experiences", {})
            self._skills = data.get("skills", {})
            self._active_states = data.get("active_states", [])
            self._version = int(data.get("version", 0) or 0)
        except (OSError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": self._version,
                    "pitfalls": self._pitfalls,
                    "table_hints": dict(self._table_hints),
                    "experiences": self._experiences,
                    "skills": self._skills,
                    "active_states": self._active_states[-self.MAX_ACTIVE_STATES :],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def compress(self) -> None:
        if len(self._pitfalls) > self.MAX_PITFALLS:
            ranked = sorted(
                self._pitfalls.items(),
                key=lambda x: (
                    int(x[1].get("count", 0) or 0),
                    int(x[1].get("last_ok", 0) or 0),
                    x[0],
                ),
                reverse=True,
            )
            self._pitfalls = dict(ranked[: self.MAX_PITFALLS])
        if len(self._experiences) > self.MAX_EXPERIENCES:
            ranked_exp = sorted(
                self._experiences.items(),
                key=lambda x: (
                    float(x[1].get("score", 0.0) or 0.0),
                    int(x[1].get("successes", 0) or 0),
                    float(x[1].get("last_seen", 0.0) or 0.0),
                ),
                reverse=True,
            )
            self._experiences = dict(ranked_exp[: self.MAX_EXPERIENCES])
        if len(self._skills) > self.MAX_SKILLS:
            ranked_skills = sorted(
                self._skills.items(),
                key=lambda x: (
                    float(x[1].get("score", 0.0) or 0.0),
                    int(x[1].get("uses", 0) or 0),
                    float(x[1].get("last_seen", 0.0) or 0.0),
                ),
                reverse=True,
            )
            self._skills = dict(ranked_skills[: self.MAX_SKILLS])
        if len(self._active_states) > self.MAX_ACTIVE_STATES:
            self._active_states = self._active_states[-self.MAX_ACTIVE_STATES :]

    def record_from_trace(
        self,
        question: str,
        *,
        success: bool,
        code: str = "",
        last_error: str = "",
        plan: Optional[Dict[str, Any]] = None,
        active_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        tables = self._tables_from_trace(code, plan)
        for m in re.finditer(r"LoadDB\(['\"](\w+)['\"]\)", code or ""):
            tbl = m.group(1)
            if tbl not in self._table_hints:
                self._table_hints[tbl] = []
        changed = False
        error_signature = self._stable_error_signature(last_error)
        if error_signature:
            key = _norm_key(error_signature)
            if key:
                entry = self._pitfalls.get(
                    key,
                    {
                        "error": error_signature[:300],
                        "fix": "",
                        "count": 0,
                        "evidence_count": 0,
                        "source": "execution_trace",
                    },
                )
                entry["count"] = int(entry.get("count", 0)) + 1
                entry["evidence_count"] = int(entry.get("evidence_count", 0)) + 1
                entry["last_seen"] = time.time()
                if success and code:
                    entry["fix"] = self._infer_fix(last_error, code)
                    entry["last_ok"] = entry.get("last_ok", 0) + 1
                self._pitfalls[key] = entry
                changed = True
        if success and "GENDER=female" in (code or ""):
            key = _norm_key("gender female filter")
            self._pitfalls[key] = {
                "error": "GENDER=female invalid; column uses f/m",
                "fix": "Use GENDER=f or GENDER=m",
                "count": self._pitfalls.get(key, {}).get("count", 0) + 1,
                "evidence_count": self._pitfalls.get(key, {}).get("evidence_count", 0) + 1,
                "source": "execution_trace",
            }
            changed = True
        # Executable memory contains verified successful traces only. Failures are
        # represented by normalized error-repair entries above, never as demos.
        if success and code and "answer" in code.lower():
            self._record_experience(
                question,
                success=True,
                code=code,
                last_error="",
                tables=tables,
                plan=plan,
            )
            self._record_skill(
                question,
                success=True,
                code=code,
                last_error="",
                tables=tables,
                plan=plan,
            )
            changed = True
        self._record_active_state(
            question,
            success=success,
            active_state=active_state,
            tables=tables,
            last_error=last_error,
        )
        if changed:
            self._version += 1
        self.compress()
        self.save()

    @staticmethod
    def _stable_error_signature(error: str) -> str:
        """Normalize reusable tool/schema failures and reject record-specific misses."""
        text = re.sub(r"\s+", " ", (error or "").strip())
        if not text:
            return ""
        low = text.lower()
        if any(x in low for x in ("no rows", "empty result", "patient not found", "hadm_id not found")):
            return ""
        if not any(
            x in low
            for x in (
                "column",
                "syntax",
                "typeerror",
                "attributeerror",
                "nameerror",
                "incorrect",
                "unsupported",
                "filterdb",
                "sqlinterpreter",
            )
        ):
            return ""
        text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
        text = re.sub(r"'[^']*'|\"[^\"]*\"", "<value>", text)
        return text[:300]

    @staticmethod
    def _tables_from_trace(code: str, plan: Optional[Dict[str, Any]] = None) -> List[str]:
        tables: List[str] = []
        for m in re.finditer(r"LoadDB\(['\"](\w+)['\"]\)", code or ""):
            tables.append(m.group(1))
        for m in re.finditer(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", code or "", flags=re.IGNORECASE):
            tables.append(m.group(1).lower())
        for t in (plan or {}).get("priority_tables") or []:
            tables.append(str(t))
        return list(dict.fromkeys(tables))[:10]

    @staticmethod
    def _compact_code(code: str, max_lines: int = 36) -> str:
        lines = [ln.rstrip() for ln in (code or "").splitlines() if ln.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[: max_lines - 2] + [f"# ... ({len(lines) - max_lines} lines omitted) ..."])

    @staticmethod
    def _skill_note(question: str, success: bool, code: str, last_error: str, tables: List[str]) -> str:
        q = question.lower()
        bits: List[str] = []
        if tables:
            bits.append("tables=" + ",".join(tables[:6]))
        if "sqlinterpreter" in (code or "").lower():
            bits.append("prefer SQLInterpreter for set/count/join logic")
        if "cost" in q:
            bits.append("link cost through EVENT_TYPE/EVENT_ID, not HADM_ID alone")
        if "route" in q or "intake method" in q:
            bits.append("drug route usually lives in prescriptions.ROUTE")
        if not success and last_error:
            bits.append("avoid prior failure: " + re.sub(r"\s+", " ", last_error.strip())[:180])
        return "; ".join(bits)[:400]

    def _record_experience(
        self,
        question: str,
        *,
        success: bool,
        code: str,
        last_error: str,
        tables: List[str],
        plan: Optional[Dict[str, Any]],
    ) -> None:
        if not question.strip():
            return
        key = _norm_key(question, max_len=180)
        now = time.time()
        entry = self._experiences.get(
            key,
            {
                "question": question[:500],
                "successes": 0,
                "failures": 0,
                "score": 0.0,
                "tables": [],
                "code": "",
                "last_error": "",
                "skill": "",
                "task_type": "",
                "question_family": "",
                "created_at": now,
            },
        )
        entry["question"] = question[:500]
        entry["last_seen"] = now
        entry["successes"] = int(entry.get("successes", 0) or 0) + (1 if success else 0)
        entry["failures"] = int(entry.get("failures", 0) or 0) + (0 if success else 1)
        entry["score"] = float(entry.get("score", 0.0) or 0.0) + (1.0 if success else -0.35)
        entry["tables"] = list(dict.fromkeys((entry.get("tables") or []) + tables))[:10]
        entry["task_type"] = (plan or {}).get("task_type") or entry.get("task_type") or ""
        entry["question_family"] = (plan or {}).get("question_family") or entry.get("question_family") or ""
        entry["skill"] = self._skill_note(question, success, code, last_error, entry["tables"])
        if success and code:
            entry["code"] = self._compact_code(code)
            entry["last_error"] = ""
        elif last_error:
            entry["last_error"] = last_error[:600]
        self._experiences[key] = entry

    @staticmethod
    def _skill_id(question: str, plan: Optional[Dict[str, Any]], tables: List[str]) -> str:
        family = (plan or {}).get("question_family") or ""
        task_type = (plan or {}).get("task_type") or ""
        q = question.lower()
        if family:
            base = family
        elif "route" in q or "intake method" in q:
            base = "drug_route_lookup"
        elif "cost" in q:
            base = "item_cost_lookup"
        elif "top" in q or "frequently" in q:
            base = "top_k_aggregate"
        elif "change in" in q or "since" in q:
            base = "temporal_window_lookup"
        elif "diagnos" in q:
            base = "diagnosis_lookup"
        else:
            base = task_type or "general_ehr_lookup"
        suffix = "_".join(tables[:2]) if tables else "generic"
        return _norm_key(f"{base}:{suffix}", max_len=96).replace(" ", "_")

    @staticmethod
    def _skill_steps(question: str, code: str, tables: List[str], plan: Optional[Dict[str, Any]]) -> List[str]:
        steps: List[str] = []
        for s in (plan or {}).get("strategy_steps") or []:
            if s not in steps:
                steps.append(str(s)[:220])
        c = (code or "").lower()
        if "sqlinterpreter" in c and "Prefer SQLInterpreter for joins/aggregates when table path is known." not in steps:
            steps.append("Prefer SQLInterpreter for joins/aggregates when table path is known.")
        if "filterdb" in c and "Load relevant tables, verify ids with FilterDB, then read final column with GetValue." not in steps:
            steps.append("Load relevant tables, verify ids with FilterDB, then read final column with GetValue.")
        if "cost" in question.lower():
            steps.append("For costs, bind cost.EVENT_TYPE to the source table and cost.EVENT_ID to source ROW_ID.")
        if "route" in question.lower() or "intake method" in question.lower():
            steps.append("For medication route, filter prescriptions.DRUG and read prescriptions.ROUTE.")
        return steps[:6]

    def _record_skill(
        self,
        question: str,
        *,
        success: bool,
        code: str,
        last_error: str,
        tables: List[str],
        plan: Optional[Dict[str, Any]],
    ) -> None:
        if not question.strip():
            return
        sid = self._skill_id(question, plan, tables)
        now = time.time()
        skill = self._skills.get(
            sid,
            {
                "id": sid,
                "name": sid.replace("_", " "),
                "description": "",
                "triggers": [],
                "tables": [],
                "steps": [],
                "successes": 0,
                "failures": 0,
                "uses": 0,
                "score": 0.0,
                "created_at": now,
                "last_error": "",
                "example_code": "",
            },
        )
        skill["uses"] = int(skill.get("uses", 0) or 0) + 1
        skill["successes"] = int(skill.get("successes", 0) or 0) + (1 if success else 0)
        skill["failures"] = int(skill.get("failures", 0) or 0) + (0 if success else 1)
        skill["score"] = float(skill.get("score", 0.0) or 0.0) + (1.0 if success else -0.5)
        skill["last_seen"] = now
        skill["tables"] = list(dict.fromkeys((skill.get("tables") or []) + tables))[:10]
        trigger_words = [w for w in re.findall(r"[a-z0-9_]+", question.lower()) if len(w) > 3]
        skill["triggers"] = list(dict.fromkeys((skill.get("triggers") or []) + trigger_words))[:24]
        skill["steps"] = list(dict.fromkeys((skill.get("steps") or []) + self._skill_steps(question, code, tables, plan)))[:8]
        skill["description"] = self._skill_note(question, success, code, last_error, skill["tables"])
        if success and code:
            skill["example_code"] = self._compact_code(code, max_lines=30)
            skill["last_error"] = ""
        elif last_error:
            skill["last_error"] = last_error[:500]
        self._skills[sid] = skill

    def _record_active_state(
        self,
        question: str,
        *,
        success: bool,
        active_state: Optional[Dict[str, Any]],
        tables: List[str],
        last_error: str,
    ) -> None:
        state = dict(active_state or {})
        state.update(
            {
                "question": question[:300],
                "success": bool(success),
                "tables": list(dict.fromkeys((state.get("tables_checked") or []) + tables))[:12],
                "last_error": last_error[:300],
                "timestamp": time.time(),
            }
        )
        self._active_states.append(state)

    @staticmethod
    def _infer_fix(error: str, code: str) -> str:
        err = error.lower()
        if "gender" in err and "female" in err:
            return "Use GENDER=f not GENDER=female"
        if "column name" in err and "incorrect" in err:
            schema_fix = fix_for_column_error(error, code)
            if schema_fix:
                return schema_fix
            if "value" in err and "chartevents" in err.lower():
                return "chartevents has VALUENUM not VALUE; use GetValue(..., 'VALUENUM') or SQLInterpreter on valuenum"
            return "Match exact CSV column names; prescriptions use DRUG not DRUG_NAME"
        if "datetime" in err and "not defined" in err:
            return "datetime is in CodeHeader; use datetime.strptime without re-importing, or add from datetime import datetime"
        if "hadm_id" in err and "incorrect" in err:
            return "Verify HADM_ID from admissions before filtering child tables"
        if "indentation" in err or "unexpected indent" in err:
            return "Keep consistent indentation inside if/for blocks"
        if "hadm_id in" in (code or "").lower() and "loaddb('cost')" in (code or "").lower():
            return (
                "Item cost: use cost.EVENT_TYPE + cost.EVENT_ID linked to labevents/procedures_icd ROW_ID; "
                "do not filter cost by HADM_ID in (...)"
            )
        c_low = (code or "").lower()
        if "join labevents" in c_low and "event_type" not in c_low:
            return (
                "Lab cost: use WHERE cost.event_type='labevents' AND cost.event_id IN "
                "(SELECT labevents.row_id FROM labevents WHERE itemid IN "
                "(SELECT itemid FROM d_labitems WHERE label=...)); never join on EVENT_ID alone"
            )
        if re.search(r"\bmin\s*\(", code or "") and "cost" in c_low:
            return (
                "Do not use min/max on COST; fix SQL with cost.event_type='labevents' (or matching event table)"
            )
        return (code or "")[:200]

    def retrieve_for_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        q = question.lower()
        words = set(re.findall(r"[a-z0-9_]+", q))
        scored: List[tuple] = []
        for key, entry in self._pitfalls.items():
            err = (entry.get("error") or "").lower()
            score = sum(1 for w in words if len(w) > 3 and w in err)
            if score > 0:
                scored.append((score * entry.get("count", 1), entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        pitfalls = [e for _, e in scored[:top_k]]
        tables: List[str] = []
        for tbl, _ in self._table_hints.items():
            if tbl in q or any(w in tbl for w in words):
                tables.append(tbl)
        if not tables:
            for hint in ("patient", "admission", "lab", "prescription", "diagnosis", "procedure"):
                if hint in q:
                    if hint == "patient":
                        tables.append("patients")
                    elif hint == "admission":
                        tables.append("admissions")
                    elif hint == "lab":
                        if "cost" in q:
                            tables.extend(["cost", "d_labitems", "labevents"])
                        else:
                            tables.extend(["labevents", "d_labitems"])
                    elif hint in ("prescription", "prescribed", "drug", "medication"):
                        tables.extend(["prescriptions", "d_icd_diagnoses", "diagnoses_icd"])
                    elif hint == "diagnosis":
                        tables.extend(["diagnoses_icd", "d_icd_diagnoses"])
                    elif hint == "procedure":
                        if "cost" in q:
                            tables.extend(["cost", "procedures_icd", "d_icd_procedures"])
                        else:
                            tables.extend(["procedures_icd", "d_icd_procedures"])
        return {
            "version": self._version,
            "pitfalls": pitfalls,
            "suggested_tables": list(dict.fromkeys(tables))[:8],
            "dynamic_experiences": self.retrieve_experiences(question, top_k=4),
            "task_memory": {"pitfalls": pitfalls, "suggested_tables": list(dict.fromkeys(tables))[:8]},
            "executable_memory": self.retrieve_experiences(question, top_k=4),
            "skills": self.retrieve_skills(question, top_k=5),
            "recent_active_states": self.retrieve_active_states(question, top_k=3),
        }

    def retrieve_experiences(self, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q = question.lower()
        words = _tokens(q)
        scored: List[tuple] = []
        for key, entry in self._experiences.items():
            text = " ".join(
                [
                    str(entry.get("question") or ""),
                    str(entry.get("skill") or ""),
                    " ".join(entry.get("tables") or []),
                    str(entry.get("task_type") or ""),
                ]
            ).lower()
            candidate_words = _tokens(text)
            lexical = _jaccard(words, candidate_words)
            overlap = len(words & candidate_words)
            if overlap <= 0:
                continue
            reliability = float(entry.get("score", 0.0) or 0.0)
            successes = int(entry.get("successes", 0) or 0)
            failures = int(entry.get("failures", 0) or 0)
            empirical = (successes + 1.0) / (successes + failures + 2.0)
            age_days = max(0.0, (time.time() - float(entry.get("last_seen", 0.0) or 0.0)) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            quality = max(0.0, min(1.0, 0.55 * empirical + 0.25 * recency + 0.20 * min(1.0, reliability / 5.0)))
            score = 0.55 * lexical + 0.30 * quality + 0.15 * min(1.0, overlap / 5.0)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for retrieval_score, entry in scored[:top_k]:
            out.append(
                {
                    "question": entry.get("question", ""),
                    "skill": entry.get("skill", ""),
                    "tables": entry.get("tables", [])[:8],
                    "successes": entry.get("successes", 0),
                    "failures": entry.get("failures", 0),
                    "score": round(float(entry.get("score", 0.0) or 0.0), 2),
                    "code": entry.get("code", ""),
                    "last_error": entry.get("last_error", ""),
                    "question_family": entry.get("question_family", ""),
                    "retrieval_score": round(float(retrieval_score), 4),
                    "retrieval_reason": "lexical+quality+recency",
                }
            )
        return out

    def retrieve_skills(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = question.lower()
        words = {w for w in re.findall(r"[a-z0-9_]+", q) if len(w) > 2}
        scored: List[tuple] = []
        for sid, skill in self._skills.items():
            text = " ".join(
                [
                    sid,
                    str(skill.get("description") or ""),
                    " ".join(skill.get("triggers") or []),
                    " ".join(skill.get("tables") or []),
                    " ".join(skill.get("steps") or []),
                ]
            ).lower()
            overlap = sum(1 for w in words if w in text)
            if overlap <= 0:
                continue
            score = overlap + float(skill.get("score", 0.0) or 0.0) + 0.15 * int(skill.get("successes", 0) or 0)
            scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": s.get("id", ""),
                "description": s.get("description", ""),
                "tables": s.get("tables", [])[:8],
                "steps": s.get("steps", [])[:6],
                "successes": s.get("successes", 0),
                "failures": s.get("failures", 0),
                "score": round(float(s.get("score", 0.0) or 0.0), 2),
                "example_code": s.get("example_code", ""),
                "last_error": s.get("last_error", ""),
            }
            for _, s in scored[:top_k]
        ]

    def retrieve_active_states(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_words = _tokens(question)
        scored: List[tuple] = []
        for state in self._active_states:
            text = " ".join(
                [
                    str(state.get("question") or ""),
                    " ".join(state.get("tables") or []),
                    " ".join(state.get("tables_checked") or []),
                    " ".join(state.get("columns_verified") or []),
                    " ".join(state.get("failed_filters") or []),
                ]
            ).lower()
            overlap = sum(1 for w in q_words if w in text)
            if overlap <= 0:
                continue
            recency = float(state.get("timestamp", 0.0) or 0.0) / 1_000_000_000.0
            scored.append((overlap + recency + (0.25 if state.get("success") else -0.25), state))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for _, state in scored[:top_k]:
            out.append(
                {
                    "success": state.get("success", False),
                    "tables": state.get("tables", [])[:8],
                    "columns_verified": state.get("columns_verified", [])[:10],
                    "failed_filters": state.get("failed_filters", [])[:6],
                    "final_answer_source": state.get("final_answer_source", ""),
                    "last_error": state.get("last_error", ""),
                }
            )
        return out
