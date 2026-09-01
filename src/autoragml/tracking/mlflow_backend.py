"""MlflowTracker — opsiyonel MLflow backend (`[tracking]` extra, ADR 0019).

Lazy import: mlflow yoksa `resolve_tracker` `ConfigError` verir (fail-fast).
`uri_env` verilirse tracking URI ortamdan çözülür — uzak sunucu = kullanıcının açık opt-in'i.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoragml.logging import get_logger

logger = get_logger(__name__)

_FLAT_SEP = "."
_MAX_PARAM_LEN = 250  # mlflow param değeri sınırı


def _flatten(prefix: str, obj: Any, out: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{_FLAT_SEP}{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = str(list(obj))[:_MAX_PARAM_LEN]
    else:
        out[prefix] = str(obj)[:_MAX_PARAM_LEN]


class MlflowTracker:
    """MLflow'a eşleyen takipçi. `mlflow` import edilebilir olmalı."""

    def __init__(self, *, tracking_uri: str | None = None, experiment: str = "autoragml") -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._active = False

    def start_run(self, run_id: str, *, project: str, config: dict[str, Any]) -> None:
        self._mlflow.start_run(run_name=f"{project}:{run_id}")
        self._active = True
        flat: dict[str, str] = {}
        _flatten("cfg", config, flat)
        self._mlflow.log_params(flat)

    def log_params(self, params: dict[str, Any]) -> None:
        flat: dict[str, str] = {}
        _flatten("", params, flat)
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def end_run(self, *, status: str = "ok") -> None:
        if self._active:
            self._mlflow.end_run(status="FINISHED" if status == "ok" else "FAILED")
            self._active = False
