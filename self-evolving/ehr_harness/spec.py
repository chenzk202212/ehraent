"""Harness specification: all mutable knobs live here; model weights never do."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalBudget:
    task_pitfalls: int = 5
    skills: int = 3
    executable_traces: int = 2
    recent_states: int = 3


@dataclass
class HarnessSpec:
    """Versioned harness state. Optimizing this is the only allowed 'learning'."""

    name: str = "ehr_memory_harness"
    version: int = 0
    # Frozen backbone — never updated by evolve loop
    llm: str = "gpt-4o-mini"
    # Scaffold knobs
    num_shots: int = 4
    planner_heuristic_only: bool = True
    compress_prompt: bool = True
    memory_agent: bool = True
    no_worldmm_context: bool = False
    ltm_disable: bool = False
    ltm_code_max_lines: int = 28
    max_consecutive_auto_reply: int = 10
    # Failure-triggered accuracy harness: retry once with memory fixes when FAIL
    retry_on_fail: bool = False
    retrieval_budget: RetrievalBudget = field(default_factory=RetrievalBudget)
    # Soft policy overlays injected into task memory / planner constraints
    constraint_overlays: List[str] = field(default_factory=list)
    family_overlays: Dict[str, List[str]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessSpec":
        data = copy.deepcopy(data or {})
        rb = data.pop("retrieval_budget", None) or {}
        if isinstance(rb, RetrievalBudget):
            budget = rb
        else:
            budget = RetrievalBudget(**{k: rb[k] for k in RetrievalBudget.__dataclass_fields__ if k in rb})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "retrieval_budget"}
        return cls(retrieval_budget=budget, **kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HarnessSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def bump(self, note: str = "") -> "HarnessSpec":
        nxt = copy.deepcopy(self)
        nxt.version = int(self.version) + 1
        if note:
            nxt.notes = (self.notes + "\n" if self.notes else "") + f"v{nxt.version}: {note}"
        return nxt

    def cli_flags(self) -> List[str]:
        flags = [
            "--llm",
            self.llm,
            "--num_shots",
            str(self.num_shots),
        ]
        if self.memory_agent:
            flags.append("--memory_agent")
        if self.planner_heuristic_only:
            flags.append("--planner_heuristic_only")
        if self.compress_prompt:
            flags.append("--compress_prompt")
        if self.no_worldmm_context:
            flags.append("--no_worldmm_context")
        if self.ltm_disable:
            flags.append("--ltm_disable")
        if self.retry_on_fail:
            flags.append("--harness_retry_on_fail")
        return flags
