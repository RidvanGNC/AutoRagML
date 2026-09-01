"""tracking — opsiyonel deney takibi (ADR 0019).

`resolve_tracker(config, run_dir) -> Tracker`. Varsayılan `JsonlTracker` (bağımlılıksız,
ağsız); `none` → `NullTracker`; `mlflow` → `MlflowTracker` (`[tracking]` extra — yoksa
`ConfigError`, sessizce jsonl'a düşmez).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from autoragml.config.settings import Settings
from autoragml.contracts.enums import TrackingBackend
from autoragml.contracts.run_config import RunConfig
from autoragml.exceptions import ConfigError
from autoragml.tracking.base import Tracker
from autoragml.tracking.jsonl import JsonlTracker
from autoragml.tracking.null import NullTracker

__all__ = ["JsonlTracker", "NullTracker", "Tracker", "resolve_tracker"]


def resolve_tracker(
    config: RunConfig, *, run_dir: str | Path, settings: Settings | None = None
) -> Tracker:
    """`config.tracking.backend` → uygun `Tracker`."""
    backend = config.tracking.backend
    if backend is TrackingBackend.NONE:
        return NullTracker()
    if backend is TrackingBackend.JSONL:
        return JsonlTracker(Path(run_dir) / "tracking")
    if backend is TrackingBackend.MLFLOW:
        if importlib.util.find_spec("mlflow") is None:
            msg = "tracking.backend=mlflow için MLflow gerekli — `pip install autoragml[tracking]`"
            raise ConfigError(msg)
        from autoragml.tracking.mlflow_backend import MlflowTracker

        uri = (settings or Settings()).get(config.tracking.uri_env) if config.tracking.uri_env else None
        return MlflowTracker(tracking_uri=uri, experiment=config.project_name)
    msg = f"bilinmeyen tracking backend: {backend!r}"  # pragma: no cover
    raise ConfigError(msg)
