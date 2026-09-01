"""EnsembleSpec — Caruana greedy selection sonucu. DONDU (ADR 0021).

`ensembling.build_weighted_ensemble` üretir. Üye key'leri + ağırlıklar; ağırlık toplamı ~1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoragml.contracts._base import FrozenContract


class EnsembleSpec(FrozenContract):
    """Ağırlıklı ensemble tarifi (OOF üzerinde seçilmiş)."""

    member_keys: list[str] = Field(min_length=1)
    weights: list[float] = Field(min_length=1)  # member_keys ile hizalı
    method: Literal["ges", "bagged_ges"]
    n_bags: int = Field(default=0, ge=0)
    oof_metric: float
    base_model_count: int = Field(ge=1)
