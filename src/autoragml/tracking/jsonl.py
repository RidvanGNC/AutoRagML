"""JsonlTracker — bağımlılıksız, ağsız JSON-lines deney takibi (ADR 0019).

`<tracking_dir>/events.jsonl` — her satır bir olay; `end_run` `summary.json` yazar
(düz params + son metrikler). Deterministik dump (`sort_keys`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoragml.logging import get_logger

logger = get_logger(__name__)


class JsonlTracker:
    """JSON-lines dosyasına ekleyen yerel takipçi."""

    def __init__(self, tracking_dir: str | Path) -> None:
        self._dir = Path(tracking_dir)
        self._events = self._dir / "events.jsonl"
        self._params: dict[str, Any] = {}
        self._last_metrics: dict[str, float] = {}
        self._run_id: str | None = None

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(UTC).isoformat(), "kind": kind, **payload}
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._events.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def start_run(self, run_id: str, *, project: str, config: dict[str, Any]) -> None:
        self._run_id = run_id
        self._write("start", {"run_id": run_id, "project": project, "config": config})

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)
        self._write("params", {"params": params})

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self._last_metrics.update(metrics)
        self._write("metrics", {"metrics": metrics, "step": step})

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        self._write("artifact", {"path": str(path), "name": name or Path(path).name})

    def end_run(self, *, status: str = "ok") -> None:
        self._write("end", {"status": status})
        summary = {
            "run_id": self._run_id,
            "status": status,
            "params": self._params,
            "metrics": self._last_metrics,
        }
        text = json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2)
        (self._dir / "summary.json").write_text(text + "\n", encoding="utf-8")
