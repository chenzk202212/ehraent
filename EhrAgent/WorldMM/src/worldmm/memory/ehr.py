from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CRITICAL_KEYWORDS = {
    "allergy", "allergic", "contraindication", "contraindicated",
    "anaphylaxis", "hypersensitivity", "adverse", "intolerance",
    "禁忌", "过敏",
}


def _is_critical(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRITICAL_KEYWORDS)


class BeliefStore:

    P_MIN = 0.70
    P_MAX = 0.90
    P_CRITICAL = 0.95
    P_CONTRADICT = 0.25
    DECAY = 0.95
    TEMPORAL_DECAY = 0.90
    MAX_CANDIDATES = 5

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._staleness: Dict[str, int] = defaultdict(int)
        self._critical: set = set()
        self._timestamps: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._evidence: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._conflicts: Dict[str, Dict[str, int]] = defaultdict(dict)

    def add(self, attribute: str, hypothesis: str, init_prob: Optional[float] = None, timestamp: int = 0) -> None:
        p = init_prob if init_prob is not None else (self.P_MIN + self.P_MAX) / 2
        if _is_critical(attribute) or _is_critical(hypothesis):
            self._critical.add(attribute)
            p = max(p, self.P_CRITICAL)
        else:
            p = max(self.P_MIN, min(self.P_MAX, p))
        if hypothesis not in self._store[attribute]:
            self._store[attribute][hypothesis] = p
            self._timestamps[attribute][hypothesis] = timestamp
            self._staleness[attribute] = 0
            self._evidence[attribute][hypothesis] = 1
            self._conflicts[attribute][hypothesis] = 0
        else:
            self.merge(attribute, hypothesis, delta=0.08, timestamp=timestamp)
        self._prune(attribute)

    def merge(self, attribute: str, hypothesis: str, delta: float, timestamp: int = 0) -> None:
        delta = max(0.0, min(1.0, delta))
        if attribute not in self._store or hypothesis not in self._store[attribute]:
            self.add(attribute, hypothesis, delta * self.P_MAX, timestamp)
            return
        p_old = self._store[attribute][hypothesis]
        p_new = min(1.0 - (1.0 - p_old) * (1.0 - delta), 0.99)
        if attribute in self._critical:
            p_new = max(p_new, self.P_CRITICAL)
        self._store[attribute][hypothesis] = p_new
        self._timestamps[attribute][hypothesis] = timestamp
        self._staleness[attribute] = 0
        self._evidence[attribute][hypothesis] = self._evidence[attribute].get(hypothesis, 1) + 1

    def contradict(self, attribute: str, hypothesis: str) -> None:
        if attribute in self._critical:
            return
        if attribute in self._store and hypothesis in self._store[attribute]:
            self._store[attribute][hypothesis] = self.P_CONTRADICT
            self._staleness[attribute] = 0
            self._conflicts[attribute][hypothesis] = self._conflicts[attribute].get(hypothesis, 0) + 1

    def _prune(self, attribute: str) -> None:
        candidates = self._store[attribute]
        if len(candidates) <= self.MAX_CANDIDATES:
            return
        if attribute in self._critical:
            return
        sorted_hyps = sorted(candidates.items(), key=lambda x: x[1])
        to_remove = len(candidates) - self.MAX_CANDIDATES
        for hyp, _ in sorted_hyps[:to_remove]:
            del self._store[attribute][hyp]
            self._timestamps[attribute].pop(hyp, None)
            self._evidence[attribute].pop(hyp, None)
            self._conflicts[attribute].pop(hyp, None)

    def top_beliefs(self, top_k: int = 8, current_time: int = 0) -> List[Tuple[str, str, float]]:
        scored: List[Tuple[float, str, str]] = []
        for attr, candidates in self._store.items():
            stale = self._staleness.get(attr, 0)
            staleness_decay = self.DECAY ** stale
            is_crit = attr in self._critical
            for hyp, prob in candidates.items():
                ts = self._timestamps.get(attr, {}).get(hyp, 0)
                time_gap = max(0, current_time - ts)
                time_steps = time_gap // 86400
                temporal_decay = 1.0 if is_crit else (self.TEMPORAL_DECAY ** time_steps)
                evidence = self._evidence.get(attr, {}).get(hyp, 1)
                conflicts = self._conflicts.get(attr, {}).get(hyp, 0)
                evidence_bonus = min(1.18, 1.0 + 0.04 * max(0, evidence - 1))
                conflict_penalty = 1.0 / (1.0 + 0.35 * conflicts)
                final_score = prob * staleness_decay * temporal_decay * evidence_bonus * conflict_penalty
                scored.append((final_score, attr, hyp))
        scored.sort(reverse=True)
        result = [(attr, hyp, prob) for prob, attr, hyp in scored[:top_k]]
        for _, attr, _ in result:
            self._staleness[attr] = self._staleness.get(attr, 0) + 1
        return result

    def format_for_prompt(self, top_k: int = 8, current_time: int = 0) -> str:
        beliefs = self.top_beliefs(top_k, current_time)
        if not beliefs:
            return ""
        lines = ["### Belief Memory"]
        for attr, hyp, prob in beliefs:
            tag = " [CRITICAL]" if attr in self._critical else ""
            bar = "█" * int(prob * 10) + "░" * (10 - int(prob * 10))
            lines.append(f"  [{bar}] {prob:.2f}{tag}  {attr} → {hyp}")
        return "\n".join(lines)

    def load_from_timeline(self, tl: Dict[str, Any]) -> None:
        until_time = int(tl.get("until_time", 0))
        for cond in tl.get("conditions", []):
            attr = "diagnosis"
            p = 0.95 if _is_critical(cond) else 0.85
            self.add(attr, cond, init_prob=p, timestamp=0)
        for med in tl.get("active_medications", []):
            p = 0.95 if _is_critical(med) else 0.80
            self.add("medication", med, init_prob=p, timestamp=0)
        for tr in tl.get("semantic_triples", []):
            attr = f"{tr.get('subj','?')}:{tr.get('pred','?')}"
            hyp = tr.get("obj", "")
            ts = int(tr.get("end_ts", 0))
            if hyp:
                p = 0.95 if (_is_critical(attr) or _is_critical(hyp)) else 0.75
                self.add(attr, hyp, init_prob=p, timestamp=ts)

    def evolve_from_success(self, question: str, code: str, current_time: int = 0) -> None:
        text = (question + " " + code).lower()
        for attr in list(self._store.keys()):
            key = attr.split(":")[-1].lower()
            if key and key in text:
                for hyp in self._store[attr]:
                    self.merge(attr, hyp, delta=0.15, timestamp=current_time)

    def evolve_from_failure(self, question: str, code: str, error: str = "") -> None:
        text = (question + " " + code + " " + error).lower()
        for attr in list(self._store.keys()):
            if attr in self._critical:
                continue
            key = attr.split(":")[-1].lower()
            if key and key in text:
                for hyp in list(self._store[attr].keys()):
                    old = self._store[attr][hyp]
                    self._store[attr][hyp] = max(old * 0.85, 0.10)
                    self._conflicts[attr][hyp] = self._conflicts[attr].get(hyp, 0) + 1

    def diagnostics(self) -> Dict[str, int]:
        """Compact evidence statistics for controller traces and ablations."""
        return {
            "attributes": len(self._store),
            "hypotheses": sum(len(v) for v in self._store.values()),
            "evidence": sum(sum(v.values()) for v in self._evidence.values()),
            "conflicts": sum(sum(v.values()) for v in self._conflicts.values()),
            "critical_attributes": len(self._critical),
        }


class EHRMWorldMemory:

    def __init__(
        self,
        embedding_model,
        retriever_llm_model,
        episodic_top_k: int = 5,
        semantic_top_k: int = 10,
        episodic_cache_root: str = ".cache/ehr_episodic",
    ) -> None:
        self._wm = None
        if embedding_model is not None and retriever_llm_model is not None:
            try:
                from worldmm.memory.memory import WorldMemory

                self._wm = WorldMemory(
                    embedding_model=embedding_model,
                    retriever_llm_model=retriever_llm_model,
                    episodic_cache_root=episodic_cache_root,
                )
                self._wm.episodic_top_k = episodic_top_k
                self._wm.semantic_top_k = semantic_top_k
            except (ImportError, ModuleNotFoundError) as exc:
                logger.warning("WorldMM video dependencies unavailable; using EHR-only retrieval: %s", exc)
        self._loaded_hadm: Optional[str] = None
        self._belief: BeliefStore = BeliefStore()
        self._current_question: str = ""
        self._current_code: str = ""
        self._until_time: int = 0
        self._events: List[Dict[str, Any]] = []
        self._raw_triples: List[Dict[str, Any]] = []

    def load_mimic_json(self, path: str) -> None:
        hadm = os.path.basename(path)
        if self._loaded_hadm == hadm:
            return

        with open(path, encoding="utf-8") as f:
            tl: Dict[str, Any] = json.load(f)

        if self._wm is not None:
            self._wm.reset()
        self._until_time = int(tl.get("until_time", 0))

        events: List[Dict[str, Any]] = tl.get("events", [])
        self._events = events
        captions = self._events_to_captions(events)
        if captions and self._wm is not None:
            self._wm.load_episodic_captions(caption_data={"30sec": captions})

        raw_triples = tl.get("semantic_triples", [])
        self._raw_triples = raw_triples
        semantic_data = self._triples_to_worldmm_format(raw_triples)
        if semantic_data and self._wm is not None:
            self._wm.load_semantic_triples(data=semantic_data)

        self._belief = BeliefStore()
        self._belief.load_from_timeline(tl)
        self._loaded_hadm = hadm
        logger.info("EHRMWorldMemory: loaded %s — %d events, %d triples", hadm, len(events), len(raw_triples))

    def memory_context_for_prompt(self, query: str, until_time: int = 0) -> str:
        self._current_question = query
        self._current_code = ""
        t = until_time or self._until_time

        index_ts = self._seconds_to_worldmm_ts(t)
        parts: List[str] = []

        belief_ctx = self._belief.format_for_prompt(top_k=8, current_time=t)
        if belief_ctx:
            parts.append(belief_ctx)

        if self._wm is None:
            events = self._lightweight_events(query, t, top_k=5)
            if events:
                parts.append("### Relevant Clinical Events\n" + "\n".join(events))
            triples = self._lightweight_triples(query, top_k=8)
            if triples:
                parts.append(
                    "### Clinical Relation Graph\n"
                    + "\n".join(f"  {s} -> {p} -> {o}" for s, p, o in triples)
                )
            return "\n\n".join(parts)

        try:
            self._wm.index(index_ts)
        except Exception as e:
            logger.warning("EHRMWorldMemory.index failed: %s", e)
            return "\n\n".join(parts)

        try:
            ep_ctx, _ = self._wm.retrieve_from_episodic(query)
            if ep_ctx and ep_ctx.strip():
                parts.append("### Relevant Clinical Events (episodic memory)\n" + ep_ctx)
            else:
                fb = self._belief_keyword_fallback(query, top_k=5)
                if fb:
                    parts.append("### Relevant Clinical Events (keyword fallback)\n" + fb)
        except Exception as e:
            logger.warning("EHRMWorldMemory episodic retrieval failed: %s", e)
            fb = self._belief_keyword_fallback(query, top_k=5)
            if fb:
                parts.append("### Relevant Clinical Events (keyword fallback)\n" + fb)

        try:
            sem_ctx, _ = self._wm.retrieve_from_semantic(query)
            if sem_ctx and sem_ctx.strip():
                parts.append("### Clinical Knowledge Graph (semantic memory)\n" + sem_ctx)
        except Exception as e:
            logger.warning("EHRMWorldMemory semantic retrieval failed: %s", e)

        return "\n\n".join(parts)

    def structured_state_for_agent(
        self,
        query: str,
        until_time: int = 0,
        *,
        belief_top_k: int = 6,
        episodic_lines: int = 4,
        semantic_triples: int = 8,
        line_max_chars: int = 160,
    ) -> Dict[str, Any]:
        """Compact JSON-friendly memory state for memory-agent planners (not raw prompt dump)."""
        self._current_question = query
        t = until_time or self._until_time
        state: Dict[str, Any] = {
            "has_timeline": bool(self._loaded_hadm),
            "until_time": t,
            "beliefs": [],
            "episodic": [],
            "semantic_triples": [],
            "relation_graph": [],
            "psm_diagnostics": self._belief.diagnostics(),
            "provenance": {"source": "ehr_timeline", "gold_used": False},
        }
        for attr, hyp, prob in self._belief.top_beliefs(belief_top_k, t):
            state["beliefs"].append(
                {
                    "attr": attr,
                    "hyp": hyp,
                    "prob": round(float(prob), 3),
                    "critical": attr in self._belief._critical,
                    "source": "record_supported_belief",
                    "evidence_count": self._belief._evidence.get(attr, {}).get(hyp, 1),
                    "conflict_count": self._belief._conflicts.get(attr, {}).get(hyp, 0),
                }
            )
        if self._wm is None:
            state["episodic"] = self._lightweight_events(query, t, top_k=episodic_lines)
            state["semantic_triples"] = self._lightweight_triples(query, top_k=semantic_triples)
            state["relation_graph"] = [
                {
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2],
                    "confidence": 1.0,
                    "source": "timeline_relation",
                }
                for triple in state["semantic_triples"]
            ]
            return state
        index_ts = self._seconds_to_worldmm_ts(t)
        try:
            self._wm.index(index_ts)
        except Exception as e:
            logger.warning("structured_state index failed: %s", e)
            return state
        try:
            ep_ctx, _ = self._wm.retrieve_from_episodic(query)
            if ep_ctx and ep_ctx.strip():
                for raw in ep_ctx.strip().splitlines()[:episodic_lines]:
                    line = raw.strip()
                    if line:
                        state["episodic"].append(line[:line_max_chars])
            else:
                fb = self._belief_keyword_fallback(query, top_k=min(3, episodic_lines))
                if fb:
                    state["episodic"] = [ln[:line_max_chars] for ln in fb.splitlines() if ln.strip()]
        except Exception as e:
            logger.warning("structured_state episodic failed: %s", e)
        try:
            sem_ctx, sem_meta = self._wm.retrieve_from_semantic(query)
            triples: List[List[str]] = []
            if sem_meta and isinstance(sem_meta, dict):
                for item in sem_meta.get("triples") or []:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        triples.append([str(item[0]), str(item[1]), str(item[2])])
            if not triples and sem_ctx:
                for raw in sem_ctx.strip().splitlines()[:semantic_triples]:
                    parts = [p.strip() for p in raw.replace("→", "->").split("->")]
                    if len(parts) >= 3:
                        triples.append(parts[:3])
            state["semantic_triples"] = triples[:semantic_triples]
            state["relation_graph"] = [
                {
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2],
                    "confidence": 1.0,
                    "source": "timeline_relation",
                }
                for triple in state["semantic_triples"]
            ]
        except Exception as e:
            logger.warning("structured_state semantic failed: %s", e)
        return state

    def update_from_experience(self, question: str, success: bool, code: str = "", error: str = "") -> None:
        q = question or self._current_question
        c = code or self._current_code
        t = self._until_time
        if success:
            self._belief.evolve_from_success(q, c, current_time=t)
        else:
            self._belief.evolve_from_failure(q, c, error=error)


    def _belief_keyword_fallback(self, query: str, top_k: int = 5) -> str:
        words = set(query.lower().split())
        scored = []
        for attr, candidates in self._belief._store.items():
            key = attr.split(":")[-1].lower()
            score = sum(1 for w in words if w in key or key in w)
            if score > 0:
                for hyp, prob in candidates.items():
                    scored.append((score * prob, attr, hyp))
        scored.sort(reverse=True)
        lines = []
        for _, attr, hyp in scored[:top_k]:
            lines.append(f"  {attr} → {hyp}")
        return "\n".join(lines)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(token) > 2}

    def _lightweight_events(self, query: str, until_time: int, top_k: int) -> List[str]:
        q_tokens = self._tokens(query)
        ranked: List[Tuple[float, int, str]] = []
        for event in self._events:
            event_time = int(event.get("t", 0) or 0)
            if until_time and event_time > until_time:
                continue
            text = " ".join(
                str(event.get(key) or "") for key in ("type", "name", "text", "value")
            ).strip()
            overlap = len(q_tokens & self._tokens(text))
            recency = event_time / max(1, until_time or self._until_time or 1)
            ranked.append((2.0 * overlap + 0.25 * recency, event_time, text[:220]))
        ranked.sort(reverse=True)
        return [f"  t={event_time}: {text}" for _, event_time, text in ranked[:top_k] if text]

    def _lightweight_triples(self, query: str, top_k: int) -> List[List[str]]:
        q_tokens = self._tokens(query)
        ranked: List[Tuple[int, List[str]]] = []
        for triple in self._raw_triples:
            values = [str(triple.get(key) or "") for key in ("subj", "pred", "obj")]
            if not all(values):
                continue
            score = len(q_tokens & self._tokens(" ".join(values)))
            ranked.append((score, values))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [values for _, values in ranked[:top_k]]

    @staticmethod
    def _seconds_to_worldmm_ts(seconds: int) -> int:
        if seconds <= 0:
            return 1_00_00_00
        total_seconds = int(seconds)
        day = 1 + total_seconds // 86400
        rem = total_seconds % 86400
        hh = rem // 3600
        mm = (rem % 3600) // 60
        ss = rem % 60
        time_part = hh * 10000 + mm * 100 + ss
        return int(f"{day}{time_part:06d}")

    @staticmethod
    def _events_to_captions(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        captions = []
        for ev in events:
            t = int(ev.get("t", 0))
            day = 1 + t // 86400
            rem = t % 86400
            hh = rem // 3600
            mm = (rem % 3600) // 60
            ss = rem % 60
            time_str = f"{hh:02d}{mm:02d}{ss:02d}"
            date_str = f"DAY{day}"
            ev_type = ev.get("type", "event")
            name = ev.get("name", "")
            text_body = ev.get("text") or ev.get("value") or name
            text = f"[{ev_type.upper()}] {name}: {text_body}".strip(": ")
            captions.append({"text": text, "start_time": time_str, "end_time": time_str, "date": date_str})
        return captions

    @staticmethod
    def _triples_to_worldmm_format(raw_triples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not raw_triples:
            return {}
        triples_list = [[tr.get("subj", ""), tr.get("pred", ""), tr.get("obj", "")] for tr in raw_triples]
        return {"0": {"consolidated_semantic_triples": triples_list}}
