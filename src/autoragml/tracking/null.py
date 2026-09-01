"""NullTracker — `backend == none` için tam no-op (ADR 0019)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class NullTracker:
    """Hiçbir şey yapmaz. Takip kapalıyken kullanılır."""

    def start_run(self, run_id: str, *, project: str, config: dict[str, Any]) -> None:
        return

    def log_params(self, params: dict[str, Any]) -> None:
        return

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        return

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        return

    def end_run(self, *, status: str = "ok") -> None:
        return
