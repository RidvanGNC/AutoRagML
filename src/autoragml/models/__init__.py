"""models — YAML katalogdan `Candidate` üretir + estimator kurar (ADR 0012/0013).

- `models/catalog/*.yaml` (pakete gömülü) + `RunConfig.model_catalog_override` deep-merge
- `resolve_candidates(config, task)` → modalite + görev uyumlu `[Candidate]` (eksik dep → atla)
- `build_estimator(candidate, task, params)` → fit edilmemiş estimator
- Entry-points `autoragml.models` ikincil
"""

from __future__ import annotations

from autoragml.models.estimator import EstimatorBuildError, build_estimator, resolve_class_path
from autoragml.models.registry import (
    ModelCatalogError,
    build_candidates,
    load_catalog,
    resolve_candidates,
)

__all__ = [
    "EstimatorBuildError",
    "ModelCatalogError",
    "build_candidates",
    "build_estimator",
    "load_catalog",
    "resolve_candidates",
    "resolve_class_path",
]
