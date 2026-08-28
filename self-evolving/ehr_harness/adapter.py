"""Run EhrAgent under a harness spec (subprocess; frozen model weights)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import ArtifactStore
from .mutate import apply_overlays_to_task_memory, distill_knowledge_from_traces
from .spec import HarnessSpec
from .traces import collect_traces, summarize_traces


DEFAULT_EHRAGENT = Path(os.environ.get("EHRAGENT_ROOT", "/home/czk/EhrAgent")).resolve()
DEFAULT_DATA_ROOT = Path(
    os.environ.get("EHRAGENT_DATA_ROOT", "/home/czk/EhrAgent/ehrsql-ehragent")
).resolve()


def resolve_python(ehragent_root: Path) -> str:
    candidates = [
        ehragent_root / ".venv" / "bin" / "python",
        ehragent_root.parent / ".venv" / "bin" / "python",
        Path("/raid/czk/EhrAgent/.venv/bin/python"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return sys.executable


def prepare_run_dir(store: ArtifactStore, spec: HarnessSpec, tag: str) -> Path:
    run_dir = store.paths.runs_dir / tag
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / str(spec.num_shots)).mkdir(parents=True, exist_ok=True)

    tm = apply_overlays_to_task_memory(spec, store.load_task_memory())
    tm_path = run_dir / ".task_memory.json"
    tm_path.write_text(json.dumps(tm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    spec.save(run_dir / "harness.json")
    return run_dir


def ehragent_num_questions_arg(start_id: int, num_questions: int) -> int:
    """EhrAgent main.py uses ``range(start_id, num_questions)`` (end index, not count).

    Harness APIs take a *count*; convert here so holdout slices are non-empty.
    ``num_questions == -1`` still means "through end of benchmark".
    """
    if num_questions < 0:
        return -1
    return int(start_id) + int(num_questions)


def build_command(
    *,
    python_bin: str,
    ehragent_dir: Path,
    spec: HarnessSpec,
    data_path: Path,
    logs_path: Path,
    num_questions: int,
    start_id: int,
    seed: int,
    quiet: bool,
) -> List[str]:
    # EhrAgent: run_indices = range(start_id, num_questions)  → num_questions is END index
    end_or_count = ehragent_num_questions_arg(start_id, num_questions)
    cmd = [
        python_bin,
        str(ehragent_dir / "main.py"),
        *spec.cli_flags(),
        "--dataset",
        "mimic_iii",
        "--data_path",
        str(data_path),
        "--logs_path",
        str(logs_path),
        "--num_questions",
        str(end_or_count),
        "--start_id",
        str(start_id),
        "--seed",
        str(seed),
        "--no_shuffle",
        "--memory_trace",
    ]
    if quiet:
        cmd.append("--quiet")
    return cmd


def run_harness_eval(
    store: ArtifactStore,
    spec: HarnessSpec,
    *,
    tag: str,
    num_questions: int = 20,
    start_id: int = 0,
    seed: int = 42,
    ehragent_root: Path = DEFAULT_EHRAGENT,
    data_root: Path = DEFAULT_DATA_ROOT,
    quiet: bool = True,
    timeout_s: Optional[int] = None,
) -> Dict[str, Any]:
    ehragent_dir = ehragent_root / "ehragent"
    data_path = data_root / "mimic_iii" / "valid_preprocessed.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing benchmark: {data_path}")
    if not (ehragent_dir / "main.py").is_file():
        raise FileNotFoundError(f"Missing EhrAgent main: {ehragent_dir / 'main.py'}")

    run_dir = prepare_run_dir(store, spec, tag)
    python_bin = resolve_python(ehragent_root)
    end_arg = ehragent_num_questions_arg(start_id, num_questions)
    if num_questions > 0 and end_arg <= start_id:
        raise ValueError(
            f"Empty EhrAgent slice: start_id={start_id}, count={num_questions} → end={end_arg}"
        )
    cmd = build_command(
        python_bin=python_bin,
        ehragent_dir=ehragent_dir,
        spec=spec,
        data_path=data_path,
        logs_path=run_dir,
        num_questions=num_questions,
        start_id=start_id,
        seed=seed,
        quiet=quiet,
    )

    env = os.environ.copy()
    env.setdefault("EHRAGENT_DATA_ROOT", str(data_root))
    # Make sure Autogen/openai see the same gateway as EhrAgent .env
    dotenv = ehragent_dir / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip("'").strip('"'))

    run_log = run_dir / "run.log"
    print(f"[harness] tag={tag} version={spec.version}", flush=True)
    print(
        f"[harness] slice start_id={start_id} count={num_questions} "
        f"→ EhrAgent --num_questions {end_arg} (end index)",
        flush=True,
    )
    print(f"[harness] cmd: {' '.join(cmd)}", flush=True)
    with run_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(ehragent_dir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )

    # Sync runtime task-memory writes, but keep harness overlay sidecar keys
    produced_tm = run_dir / ".task_memory.json"
    if produced_tm.is_file():
        produced = json.loads(produced_tm.read_text(encoding="utf-8"))
        prior = store.load_task_memory()
        # Merge prior verified skills/experiences so earlier PASSes survive into later runs.
        for bucket in ("skills", "experiences", "pitfalls"):
            merged = dict(prior.get(bucket) or {})
            incoming = produced.get(bucket) or {}
            if isinstance(incoming, dict):
                for k, v in incoming.items():
                    if not isinstance(v, dict):
                        merged[k] = v
                        continue
                    old = merged.get(k) if isinstance(merged.get(k), dict) else {}
                    if int(v.get("successes", 0) or 0) >= int(old.get("successes", 0) or 0):
                        merged[k] = {**old, **v}
                    else:
                        merged[k] = {**v, **old}
            produced[bucket] = merged
        seeded = apply_overlays_to_task_memory(spec, prior)
        for k in (
            "_harness_constraint_overlays",
            "_harness_family_overlays",
            "_harness_retrieval_budget",
            "_harness_ltm_code_max_lines",
        ):
            if k in seeded:
                produced[k] = seeded[k]
        store.save_task_memory(produced)

    logs_q = run_dir / str(spec.num_shots)
    traces = collect_traces(logs_q, data_path=data_path, run_log=run_log)
    # Bilateral knowledge: PASS → skills; FAIL → structured diagnostics (not raw Error overlays).
    tm_now = store.load_task_memory()
    tm_distilled, distill_notes = distill_knowledge_from_traces(tm_now, traces)
    if distill_notes:
        store.save_task_memory(tm_distilled)
        print(
            f"[harness] distilled {len(distill_notes)} knowledge updates into task_memory",
            flush=True,
        )
    metrics = summarize_traces(traces)
    metrics.update(
        {
            "tag": tag,
            "harness_version": spec.version,
            "exit_code": proc.returncode,
            "num_questions_requested": num_questions,
            "start_id": start_id,
            "run_dir": str(run_dir),
            "llm": spec.llm,
            "knobs": {
                "num_shots": spec.num_shots,
                "compress_prompt": spec.compress_prompt,
                "planner_heuristic_only": spec.planner_heuristic_only,
                "ltm_code_max_lines": spec.ltm_code_max_lines,
            },
        }
    )
    if proc.returncode != 0:
        metrics["sr"] = metrics.get("sr", 0.0)
        metrics["failed_run"] = True
        print(f"[harness] WARNING: EhrAgent exit_code={proc.returncode}; see {run_log}", flush=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "traces.json").write_text(
        json.dumps([t.to_dict() for t in traces], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics, "traces": traces, "run_dir": run_dir, "exit_code": proc.returncode}
