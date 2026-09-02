"""models — YAML katalogdan `Candidate` üretir + estimator kurar (ADR 0012/0013).

- `models/catalog/*.yaml` (pakete gömülü) + `RunConfig.model_catalog_override` deep-merge
- `resolve_candidates(config, task)` → modalite + görev uyumlu `[Candidate]` (eksik dep → atla)
- `build_estimator(candidate, task, params)` → fit edilmemiş estimator
- Entry-points `autoragml.models` ikincil
"""

from __future__ import annotations

from typing import Any

from autoragml.contracts.candidate import Candidate
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
    "apply_model_hints",
    "build_candidates",
    "build_estimator",
    "load_catalog",
    "resolve_candidates",
    "resolve_class_path",
]


def apply_model_hints(
    candidates: list[Candidate], hints: dict[str, dict[str, Any]]
) -> list[Candidate]:
    """Plan'ın model ipuçlarını (ör. Tweedie objective) eşleşen adayların default_params'ına merge et (ADR 0024)."""
    if not hints:
        return candidates
    out: list[Candidate] = []
    for cand in candidates:
        hint = hints.get(cand.key) or hints.get(cand.family)
        if hint:
            out.append(cand.model_copy(update={"default_params": {**cand.default_params, **hint}}))
        else:
            out.append(cand)
    return out
