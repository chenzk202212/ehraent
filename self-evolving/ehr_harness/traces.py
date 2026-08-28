"""Parse EhrAgent per-question logs into structured traces for harness evolution."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_LOG_SEP = "\n----------------------------------------------------------\n"


@dataclass
class QuestionTrace:
    question_id: str
    question: str = ""
    gold: str = ""
    terminated: bool = False
    passed: bool = False
    unfinished: bool = False
    last_error: str = ""
    code_snippets: List[str] = field(default_factory=list)
    log_path: str = ""
    judge_source: str = ""  # run_log | soft_judge | unknown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_errors(text: str) -> str:
    errs = []
    for m in re.finditer(r"(?im)^(?:Error:|Potential Reasons:).{0,400}", text or ""):
        errs.append(m.group(0).strip())
    for m in re.finditer(r"(?i)incorrect column name[^\n]{0,200}", text or ""):
        errs.append(m.group(0).strip())
    return " | ".join(errs[:5])


def _extract_code_cells(text: str) -> List[str]:
    cells: List[str] = []
    for m in re.finditer(r'(?s)"cell"\s*:\s*"((?:\\.|[^"\\])*)"', text or ""):
        try:
            raw = m.group(1).encode("utf-8").decode("unicode_escape")
        except Exception:
            raw = m.group(1)
        cells.append(raw)
    if not cells:
        for m in re.finditer(r"(?m)^(answer\s*=.*)$", text or ""):
            cells.append(m.group(1))
    return cells[-4:]


def _assistant_terminated(text: str) -> bool:
    """
    Avoid false positives from few-shot prompts that literally contain 'TERMINATE'.
    Prefer an assistant turn that ends with TERMINATE, or a standalone TERMINATE segment.
    """
    parts = [p.strip() for p in (text or "").split(_LOG_SEP) if p.strip()]
    if not parts:
        return False
    # Skip question/gold/ground-truth preamble when present
    body = parts[2:] if len(parts) >= 3 else parts
    for p in reversed(body):
        if p == "TERMINATE" or p.rstrip().endswith("TERMINATE"):
            return True
        if p.startswith("Ground-Truth Answer"):
            continue
        # once we hit a non-terminate content block near the end, stop scanning forever
        if "***** Response from calling function" in p or p.startswith("Error:"):
            continue
        break
    # run.log style: explicit flag
    if re.search(r"TERMINATE in log:\s*True", text or ""):
        return True
    return False


def parse_judges_from_run_log(run_log: Path) -> Dict[str, Tuple[bool, bool]]:
    """Map question_id -> (passed, terminated) from console run.log lines."""
    if not run_log.is_file():
        return {}
    text = run_log.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Tuple[bool, bool]] = {}
    current_id = None
    for line in text.splitlines():
        m = re.search(r"id=([0-9a-fA-F]+)", line)
        if m and ("--- item" in line or "id=" in line):
            current_id = m.group(1)
        if current_id and "judge=" in line:
            passed = "judge=PASS" in line
            # unfinished if judge line says TERMINATE in log: False
            term_m = re.search(r"TERMINATE in log:\s*(True|False)", line)
            terminated = True
            if term_m:
                terminated = term_m.group(1) == "True"
            elif "unfinished" in line.lower():
                terminated = False
            out[current_id] = (passed, terminated)
    return out


def parse_log_file(
    path: Path,
    *,
    gold_hint: Any = None,
    judge_hint: Optional[Tuple[bool, bool]] = None,
) -> QuestionTrace:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = text.split(_LOG_SEP)
    qid = path.stem
    question = parts[0].strip() if parts else ""
    gold = ""
    if len(parts) >= 2:
        gold = parts[1].strip()
    for p in parts:
        if p.startswith("Ground-Truth Answer --->"):
            gold = p.split("--->", 1)[-1].strip()

    judge_source = "unknown"
    if judge_hint is not None:
        passed, terminated = judge_hint
        judge_source = "run_log"
    else:
        terminated = _assistant_terminated(text)
        pred_region = parts[-3:] if len(parts) >= 3 else parts
        pred = _LOG_SEP.join(pred_region)
        ans = gold_hint if gold_hint is not None else gold
        passed = bool(terminated and ans is not None and _soft_judge(pred, ans))
        judge_source = "soft_judge"

    unfinished = not terminated
    if unfinished:
        passed = False

    return QuestionTrace(
        question_id=qid,
        question=question,
        gold=str(gold if gold_hint is None else gold_hint),
        terminated=terminated,
        passed=passed,
        unfinished=unfinished,
        last_error=_extract_errors(text),
        code_snippets=_extract_code_cells(text),
        log_path=str(path),
        judge_source=judge_source,
    )


def _soft_judge(pred: str, ans: Any) -> bool:
    if isinstance(ans, list):
        tokens = [str(x) for x in ans]
    else:
        tokens = [str(ans)]
    pred_l = pred or ""
    for t in tokens:
        if t and t not in pred_l:
            return False
    return True


def load_benchmark_index(data_path: str | Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return {str(r.get("id")): r for r in rows if r.get("id")}


def collect_traces(
    logs_dir: str | Path,
    *,
    data_path: Optional[str | Path] = None,
    run_log: Optional[str | Path] = None,
) -> List[QuestionTrace]:
    logs_dir = Path(logs_dir)
    index = load_benchmark_index(data_path) if data_path else {}
    judges = parse_judges_from_run_log(Path(run_log)) if run_log else {}
    if not judges:
        # default location: parent/run.log
        maybe = logs_dir.parent / "run.log"
        judges = parse_judges_from_run_log(maybe)
    traces: List[QuestionTrace] = []
    for path in sorted(logs_dir.glob("*.txt")):
        row = index.get(path.stem)
        gold = row.get("answer") if row else None
        tr = parse_log_file(path, gold_hint=gold, judge_hint=judges.get(path.stem))
        if row and row.get("template"):
            tr.question = str(row["template"])
        traces.append(tr)
    return traces


def summarize_traces(traces: List[QuestionTrace]) -> Dict[str, Any]:
    n = len(traces)
    n_ok = sum(1 for t in traces if t.passed)
    n_unfinished = sum(1 for t in traces if t.unfinished)
    n_fail = sum(1 for t in traces if t.terminated and not t.passed)
    n_from_runlog = sum(1 for t in traces if t.judge_source == "run_log")
    return {
        "total": n,
        "correct": n_ok,
        "incorrect": n_fail,
        "unfinished": n_unfinished,
        "sr": (100.0 * n_ok / n) if n else 0.0,
        "cr": (100.0 * (n_ok + n_fail) / n) if n else 0.0,
        "error_rate": (100.0 * sum(1 for t in traces if t.last_error) / n) if n else 0.0,
        "judge_from_run_log": n_from_runlog,
    }
