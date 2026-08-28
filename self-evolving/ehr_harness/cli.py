#!/usr/bin/env python3
"""CLI for the self-evolving EHR harness (scaffold optimization only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m ehr_harness.cli` from self-evolving/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ehr_harness.artifacts import ArtifactStore
from ehr_harness.evolve import evaluate_only, evolve
from ehr_harness.spec import HarnessSpec


def _default_artifacts() -> Path:
    return _ROOT / "artifacts" / "ehr_default"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Self-evolving EHR agent harness: optimize prompts/memory/knobs only. "
            "Model weights are never updated."
        )
    )
    p.add_argument(
        "--artifacts",
        type=str,
        default=str(_default_artifacts()),
        help="Harness artifact directory (harness.json + task_memory.json).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create a fresh harness artifact dir")
    init.add_argument("--llm", type=str, default="gpt-4o-mini")
    init.add_argument("--name", type=str, default="ehr_memory_harness")

    ev = sub.add_parser("eval", help="Evaluate current harness on a slice")
    ev.add_argument("--num_questions", type=int, default=20)
    ev.add_argument("--start_id", type=int, default=0)
    ev.add_argument("--tag", type=str, default="eval")
    ev.add_argument("--llm", type=str, default="")

    evo = sub.add_parser("evolve", help="Run outer harness evolution loop")
    evo.add_argument("--generations", type=int, default=3)
    evo.add_argument("--train_questions", type=int, default=16)
    evo.add_argument("--holdout_questions", type=int, default=8)
    evo.add_argument("--train_start", type=int, default=0)
    evo.add_argument("--holdout_start", type=int, default=200)
    evo.add_argument("--seed", type=int, default=42)
    evo.add_argument("--llm", type=str, default="", help="Inner executor / Memory Agent LLM")
    evo.add_argument(
        "--meta_llm",
        type=str,
        default="",
        help="Outer meta-planner LLM (defaults to --llm). Proposes harness edits from traces.",
    )
    evo.add_argument(
        "--heuristic_only",
        action="store_true",
        help="Skip LLM meta-planner; use rule-based mutations only.",
    )
    evo.add_argument(
        "--no_inner_llm_planner",
        action="store_true",
        help="Force Memory Agent --planner_heuristic_only (ablation). Default keeps inner LLM planning ON.",
    )
    evo.add_argument(
        "--accept_min_holdout_delta",
        type=float,
        default=-0.0,
        help="Accept candidate if holdout SR improves by at least this (percentage points). Default 0.",
    )

    sh = sub.add_parser("show", help="Print current harness.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ArtifactStore(args.artifacts)

    if args.cmd == "init":
        store.paths.root.mkdir(parents=True, exist_ok=True)
        spec = HarnessSpec(
            name=args.name,
            llm=args.llm,
            compress_prompt=True,
            # Inner Memory Agent uses LLM planning by default; outer meta-planner evolves harness.
            planner_heuristic_only=False,
            retry_on_fail=False,
            constraint_overlays=[],
        )
        store.save_spec(spec)
        store.ensure_task_memory()
        print(f"Initialized harness at {store.paths.root}")
        print(json.dumps(spec.to_dict(), indent=2))
        return 0

    if args.cmd == "show":
        spec = store.load_spec()
        print(json.dumps(spec.to_dict(), indent=2))
        if store.paths.metrics.is_file():
            print("--- metrics ---")
            print(store.paths.metrics.read_text(encoding="utf-8"))
        return 0

    if args.cmd == "eval":
        result = evaluate_only(
            store,
            num_questions=args.num_questions,
            start_id=args.start_id,
            tag=args.tag,
            llm=args.llm or None,
        )
        print(json.dumps(result["metrics"], indent=2))
        return int(result.get("exit_code") or 0)

    if args.cmd == "evolve":
        out = evolve(
            store,
            generations=args.generations,
            train_questions=args.train_questions,
            holdout_questions=args.holdout_questions,
            train_start=args.train_start,
            holdout_start=args.holdout_start,
            seed=args.seed,
            llm=args.llm or None,
            meta_llm=args.meta_llm or None,
            heuristic_only=bool(args.heuristic_only),
            force_inner_llm_planner=not bool(args.no_inner_llm_planner),
            accept_min_holdout_delta=args.accept_min_holdout_delta,
        )
        print(json.dumps({"best": out["best"], "generations": len(out["history"])}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
