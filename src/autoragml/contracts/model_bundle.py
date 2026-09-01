"""ModelBundle — şampiyon modelin taşınabilir paketi. DONDU (ADR 0011 + 0014 + 0015).

`pipeline` canlı fitted nesne (FittedTransform'lar + estimator + postprocessors);
serialize edilmez, `persistence` ayrı yazar. Geri kalan alanlar saf metadata.
Şampiyon tüm train'de refit edilir; ES modelleri için `best_iteration` sabittir.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import Provenance


class BundleMetadata(Contract):
    """ModelBundle'ın serialize edilebilir metadata'sı."""

    feature_cols: list[str]
    feature_set_hash: str
    target_col: str
    model_key: str
    scenario: str = "scenario_1"
    best_iteration: int | None = None
    provenance_fitted_on: Provenance = Provenance.TRAIN
    config_snapshot_ref: str | None = None
    adaptive_plan_summary: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    postprocess_summary: dict[str, Any] = Field(default_factory=dict)  # uygulanan düzeltme adımları (ADR 0017)
    ensemble: dict[str, Any] = Field(default_factory=dict)  # üye key → ağırlık (ADR 0021); tek modelde boş


class ModelBundle(Contract):
    """Fitted şampiyon + metadata + metrikler."""

    metadata: BundleMetadata
    metrics_oof: dict[str, float] = Field(default_factory=dict)
    metrics_holdout: dict[str, float] = Field(default_factory=dict)  # bir kez skorlanır
    artifact_path: str | None = None

    # Canlı fitted pipeline — serialize edilmez.
    pipeline: Any = Field(default=None, exclude=True, repr=False)
