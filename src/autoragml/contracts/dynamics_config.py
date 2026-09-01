"""DynamicsConfig — `dynamics/planner` ayarları (ADR 0007 + 0010 + 0015).

`RunConfig.dynamics` altında taşınır.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoragml.contracts._base import Contract


class DynamicsConfig(Contract):
    """Veriye-özel strateji planlayıcısının ayarları."""

    # Yapısal seçim
    structure: Literal["auto", "pooled", "per_group_champion"] = "auto"
    per_group_min_series: int = Field(default=2, ge=1)
    per_group_max_series: int = Field(default=5000, ge=1)
    per_group_min_history_multiplier: float = Field(default=2.0, gt=0.0)

    # candidate_ops seçenekleri (HPO uzayında seçilir — ADR 0015)
    numeric_transform_choices: list[str] = Field(
        default_factory=lambda: ["none", "log1p", "yeo_johnson", "quantile"]
    )
    target_transform_choices: list[str] = Field(
        default_factory=lambda: ["none", "log1p", "yeo_johnson"]
    )

    # Kodlama (committed)
    high_cardinality_encoding: Literal["target_encode", "hashing"] = "target_encode"
    low_cardinality_encoding: Literal["onehot", "ordinal"] = "onehot"
    max_onehot_cardinality: int = Field(default=20, ge=2)

    # Custom recipe'ler (isimle; registry'de çözülür — ADR 0015)
    recipes: list[str] = Field(default_factory=list)

    # Sızıntı şüphelilerini otomatik düşür (varsayılan HAYIR — ADR 0011: yalnız uyar)
    drop_leakage_suspects: bool = False
