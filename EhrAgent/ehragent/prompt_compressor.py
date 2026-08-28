"""Structure-aware prompt compression for EHRAgent memory prompts.

This is a lightweight, training-free BEAVER-style compressor: preserve fixed
tool instructions, then select whole structured blocks from examples/memory by
question relevance instead of pruning arbitrary tokens.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple


_STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "were",
    "was",
    "has",
    "had",
    "have",
    "their",
    "patient",
    "patients",
    "hospital",
    "current",
    "last",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _char_budget(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Smooth at line/sentence boundary where possible.
    for sep in ("\n\n", "\n", ". "):
        i = cut.rfind(sep)
        if i > max_chars * 0.55:
            return cut[: i + len(sep)].rstrip() + "\n# ... compressed ..."
    return cut.rstrip() + "\n# ... compressed ..."


def split_example_blocks(examples: str) -> List[str]:
    text = (examples or "").strip()
    if not text:
        return []
    starts = [m.start() for m in re.finditer(r"(?m)^Question:\s", text)]
    if not starts:
        return [text]
    blocks: List[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def split_memory_blocks(knowledge: str) -> List[str]:
    text = (knowledge or "").strip()
    if not text:
        return []
    starts = [m.start() for m in re.finditer(r"(?m)^###\s+", text)]
    if not starts:
        return [text]
    if starts[0] != 0:
        starts = [0] + starts
    blocks: List[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def _score_block(block: str, query_words: set[str], *, base: float = 0.0) -> float:
    b_words = _words(block)
    overlap = len(query_words & b_words)
    score = base + overlap
    low = block.lower()
    # Domain-specific relevance bonuses.
    if any(w in query_words for w in ("route", "intake", "drug", "medication")) and (
        "route" in low or "prescriptions" in low or "medication" in low
    ):
        score += 4
    if any(w in query_words for w in ("cost", "charge")) and "cost" in low:
        score += 4
    if any(w in query_words for w in ("diagnosis", "diagnosed", "icd")) and ("diagnos" in low or "icd" in low):
        score += 3
    if any(w in query_words for w in ("lab", "glucose", "magnesium", "phosphate")) and (
        "lab" in low or "labevents" in low
    ):
        score += 3
    if "reference_python" in low or "memory agent plan" in low:
        score += 8
    if "skill memory" in low:
        score += 5
    return score


def _select_blocks(
    blocks: Iterable[str],
    query: str,
    *,
    max_blocks: int,
    max_chars: int,
    always_keep_first: bool = False,
) -> str:
    blocks = list(blocks)
    if not blocks:
        return ""
    if max_blocks <= 0 or max_chars <= 0:
        return ""
    q_words = _words(query)
    scored: List[Tuple[float, int, str]] = []
    for i, block in enumerate(blocks):
        base = 2.0 if always_keep_first and i == 0 else 0.0
        scored.append((_score_block(block, q_words, base=base), i, block))
    ranked = sorted(scored, key=lambda x: (x[0], -x[1]), reverse=True)[:max_blocks]
    # Restore original order for discourse coherence.
    selected = [b for _, _, b in sorted(ranked, key=lambda x: x[1])]
    out: List[str] = []
    used = 0
    for block in selected:
        room = max_chars - used
        if room <= 0:
            break
        b = _char_budget(block, room)
        out.append(b)
        used += len(b) + 2
    return "\n\n".join(out).strip()


def compress_examples(examples: str, question: str, *, max_blocks: int = 4, max_chars: int = 9000) -> str:
    return _select_blocks(
        split_example_blocks(examples),
        question,
        max_blocks=max_blocks,
        max_chars=max_chars,
        always_keep_first=True,
    )


def compress_knowledge(knowledge: str, question: str, *, max_blocks: int = 5, max_chars: int = 7000) -> str:
    return _select_blocks(
        split_memory_blocks(knowledge),
        question,
        max_blocks=max_blocks,
        max_chars=max_chars,
        always_keep_first=True,
    )
