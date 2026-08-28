"""Outer self-evolution loop: LLM meta-plans harness changes (no weight updates)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .adapter import run_harness_eval
from .artifacts import ArtifactStore
from .meta_planner import apply_meta_plan, plan_harness_with_llm
from .mutate import distill_knowledge_from_traces, mutate_spec_from_metrics, mutate_task_memory_from_traces
from .spec import HarnessSpec


def evolve(
    store: ArtifactStore,
    *,
    generations: int = 3,
    train_questions: int = 16,
    holdout_questions: int = 8,
    train_start: int = 0,
    holdout_start: int = 200,
    seed: int = 42,
    llm: Optional[str] = None,
    meta_llm: Optional[str] = None,
    accept_min_holdout_delta: float = 0.0,
    heuristic_only: bool = False,
    force_inner_llm_planner: bool = True,
) -> Dict[str, Any]:
    """
    For each generation:
      1) evaluate current harness on a train slice (inner agent still LLM-plans by default)
      2) LLM meta-planner reads traces → HarnessPlan (fallback: heuristic mutate)
      3) apply plan to artifacts/knobs (never model weights)
      4) holdout eval → accept / reject (rollback task memory on reject)
    """
    spec = store.load_spec()
    if llm:
        spec.llm = llm
    if force_inner_llm_planner:
        spec.planner_heuristic_only = False
    store.save_spec(spec)
    store.ensure_task_memory()

    planner_model = (meta_llm or spec.llm or "gpt-4o-mini").strip()
    history: list[Dict[str, Any]] = []
    best = {
        "spec": spec.to_dict(),
        "holdout_sr": -1.0,
        "train_sr": -1.0,
    }

    # Establish a real baseline holdout BEFORE any mutation, so gen0 cannot
    # "accept" a no-op / SR-hurting change just because best was unset.
    if accept_min_holdout_delta >= 0:
        tag_base = f"baseline_hold_v{spec.version}"
        base = run_harness_eval(
            store,
            spec,
            tag=tag_base,
            num_questions=holdout_questions,
            start_id=holdout_start,
            seed=seed + 1,
        )
        base_metrics = base["metrics"]
        base_exit = base_metrics.get("exit_code", 1)
        base_ok = (base_exit is not None and int(base_exit) == 0) and not base_metrics.get("failed_run")
        if base_ok:
            best = {
                "spec": spec.to_dict(),
                "holdout_sr": float(base_metrics["sr"]),
                "train_sr": -1.0,
                "holdout": base_metrics,
            }
            print(
                f"[evolve] baseline hold_sr={best['holdout_sr']:.2f} "
                f"(accept needs +{accept_min_holdout_delta:g}pp)",
                flush=True,
            )
        else:
            print(
                f"[evolve] baseline hold failed exit={base_exit}; "
                f"will require positive hold SR to accept",
                flush=True,
            )

    for g in range(generations):
        parent_tm = copy.deepcopy(store.load_task_memory())
        parent_spec = copy.deepcopy(spec)

        tag_train = f"gen{g:02d}_train_v{spec.version}"
        train = run_harness_eval(
            store,
            spec,
            tag=tag_train,
            num_questions=train_questions,
            start_id=train_start,
            seed=seed,
        )
        train_metrics = train["metrics"]
        traces = train["traces"]

        # Snapshot memory after train (includes online skill writes + PASS/FAIL distill)
        tm_before_meta = copy.deepcopy(store.load_task_memory())
        # Bilateral knowledge: PASS → skills; FAIL → structured diagnostics
        tm_before_meta, succ_notes = distill_knowledge_from_traces(tm_before_meta, traces)
        if succ_notes:
            store.save_task_memory(tm_before_meta)
            print(f"[evolve] train knowledge distilled: {succ_notes[:8]}", flush=True)

        plan_source = "heuristic"
        plan: Dict[str, Any] = {}
        notes: list[str] = []
        cand_spec = copy.deepcopy(spec)

        if not heuristic_only:
            try:
                plan = plan_harness_with_llm(
                    spec,
                    traces,
                    model=planner_model,
                    metrics=train_metrics,
                )
                cand_spec, tm, notes = apply_meta_plan(spec, tm_before_meta, plan)
                store.save_task_memory(tm)
                plan_source = "llm"
                print(
                    f"[evolve] meta-plan rationale: {(plan.get('rationale') or '')[:200]}",
                    flush=True,
                )
            except Exception as e:
                print(f"[evolve] LLM meta-planner failed ({e}); falling back to heuristic.", flush=True)
                plan_source = "heuristic_fallback"

        if plan_source != "llm":
            # Bilateral knowledge + heuristic knob search
            tm, tm_notes = distill_knowledge_from_traces(tm_before_meta, traces)
            store.save_task_memory(tm)
            cand_spec, knob_notes = mutate_spec_from_metrics(
                spec,
                train_metrics,
                prev_metrics=best if best["holdout_sr"] >= 0 else None,
            )
            if force_inner_llm_planner:
                cand_spec.planner_heuristic_only = False
            notes = knob_notes + tm_notes

        tag_hold = f"gen{g:02d}_hold_v{cand_spec.version}"
        hold = run_harness_eval(
            store,
            cand_spec,
            tag=tag_hold,
            num_questions=holdout_questions,
            start_id=holdout_start,
            seed=seed + 1,
        )
        hold_metrics = hold["metrics"]

        accepted = False
        hold_exit = hold_metrics.get("exit_code", 1)
        train_exit = train_metrics.get("exit_code", 1)
        hold_ok = (hold_exit is not None and int(hold_exit) == 0) and not hold_metrics.get("failed_run")
        train_ok = (train_exit is not None and int(train_exit) == 0) and not train_metrics.get("failed_run")
        if hold_ok and train_ok:
            if best["holdout_sr"] < 0:
                # No usable baseline: only accept if hold SR is strictly positive.
                accepted = float(hold_metrics["sr"]) > 0.0
            else:
                delta = float(hold_metrics["sr"]) - float(best["holdout_sr"])
                accepted = delta >= accept_min_holdout_delta
                # Never accept a candidate that cuts shots without SR gain.
                if accepted and int(cand_spec.num_shots) < int(parent_spec.num_shots) and delta < 1.0:
                    accepted = False
                    notes = list(notes) + ["reject: fewer shots without ≥1pp hold SR gain"]
        else:
            print(
                f"[evolve] skip accept: train_ok={train_ok} hold_ok={hold_ok} "
                f"train_exit={train_exit} hold_exit={hold_exit}",
                flush=True,
            )

        record = {
            "generation": g,
            "accepted": accepted,
            "plan_source": plan_source,
            "train": train_metrics,
            "holdout": hold_metrics,
            "notes": notes,
            "rationale": (plan.get("rationale") if plan else "") or "",
            "candidate_version": cand_spec.version,
            "parent_version": parent_spec.version,
            "inner_planner_heuristic_only": cand_spec.planner_heuristic_only,
        }
        store.append_history(record)
        history.append(record)
        print(
            f"[evolve] gen={g} source={plan_source} "
            f"train_sr={train_metrics['sr']:.2f} hold_sr={hold_metrics['sr']:.2f} "
            f"accepted={accepted} inner_llm_plan={not cand_spec.planner_heuristic_only} "
            f"notes={notes[:5]}",
            flush=True,
        )

        if accepted:
            spec = cand_spec
            store.save_spec(spec)
            # Keep candidate memory, then fold holdout PASS+FAIL knowledge.
            hold_traces = hold.get("traces") or []
            tm_acc = store.load_task_memory()
            tm_acc, hold_succ = distill_knowledge_from_traces(tm_acc, hold_traces)
            if hold_succ:
                store.save_task_memory(tm_acc)
                print(f"[evolve] hold knowledge distilled: {hold_succ[:8]}", flush=True)
            best = {
                "spec": spec.to_dict(),
                "holdout_sr": float(hold_metrics["sr"]),
                "train_sr": float(train_metrics["sr"]),
                "holdout": hold_metrics,
                "train": train_metrics,
            }
            store.snapshot(f"best_v{spec.version}")
        else:
            # Rollback harness knobs, but KEEP train PASS+FAIL knowledge.
            spec = parent_spec
            store.save_spec(spec)
            tm_keep, keep_notes = distill_knowledge_from_traces(parent_tm, traces)
            store.save_task_memory(tm_keep)
            print(
                f"[evolve] rejected gen={g}; rolled back knobs to v{spec.version}; "
                f"kept {len(keep_notes)} knowledge updates",
                flush=True,
            )

    store.save_metrics(
        {
            "best": best,
            "history": history,
            "policy": "harness-only + LLM meta-planner (no model weight updates)",
            "meta_llm": planner_model,
        }
    )
    return {"best": best, "history": history}


def evaluate_only(
    store: ArtifactStore,
    *,
    num_questions: int = 20,
    start_id: int = 0,
    tag: str = "eval",
    llm: Optional[str] = None,
    force_inner_llm_planner: bool = False,
) -> Dict[str, Any]:
    """Run one eval under the saved harness.

    By default, preserve ``planner_heuristic_only`` from harness.json (paper Memory
    Agent uses heuristic=True). Pass force_inner_llm_planner=True only for ablations
    that intentionally turn the inner LLM planner on.
    """
    spec = store.load_spec()
    if llm:
        spec.llm = llm
    if force_inner_llm_planner:
        spec.planner_heuristic_only = False
    store.save_spec(spec)
    result = run_harness_eval(
        store,
        spec,
        tag=tag,
        num_questions=num_questions,
        start_id=start_id,
    )
    store.save_metrics(result["metrics"])
    store.append_history({"event": "eval", **result["metrics"]})
    return result
