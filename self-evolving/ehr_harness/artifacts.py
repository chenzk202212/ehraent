"""Versioned harness artifact store (JSON only — no checkpoints / weights)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .spec import HarnessSpec


@dataclass
class ArtifactPaths:
    root: Path

    @property
    def harness_json(self) -> Path:
        return self.root / "harness.json"

    @property
    def task_memory(self) -> Path:
        return self.root / "task_memory.json"

    @property
    def history(self) -> Path:
        return self.root / "history.jsonl"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.paths = ArtifactPaths(Path(root).resolve())
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)

    def load_spec(self) -> HarnessSpec:
        if self.paths.harness_json.is_file():
            return HarnessSpec.load(self.paths.harness_json)
        return HarnessSpec()

    def save_spec(self, spec: HarnessSpec) -> None:
        spec.save(self.paths.harness_json)

    def ensure_task_memory(self) -> Path:
        if not self.paths.task_memory.is_file():
            self.paths.task_memory.write_text(
                json.dumps(
                    {
                        "version": 0,
                        "pitfalls": {},
                        "table_hints": {},
                        "experiences": {},
                        "skills": {},
                        "active_states": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return self.paths.task_memory

    def load_task_memory(self) -> Dict[str, Any]:
        path = self.ensure_task_memory()
        return json.loads(path.read_text(encoding="utf-8"))

    def save_task_memory(self, data: Dict[str, Any]) -> None:
        data = dict(data)
        data["version"] = int(data.get("version", 0) or 0) + 1
        self.paths.task_memory.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def append_history(self, record: Dict[str, Any]) -> None:
        record = {
            **record,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self.paths.history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        self.paths.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    def snapshot(self, tag: str) -> Path:
        dest = self.paths.runs_dir / tag
        dest.mkdir(parents=True, exist_ok=True)
        for src in (self.paths.harness_json, self.paths.task_memory, self.paths.metrics):
            if src.is_file():
                shutil.copy2(src, dest / src.name)
        return dest

    def list_history(self) -> List[Dict[str, Any]]:
        if not self.paths.history.is_file():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.paths.history.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
