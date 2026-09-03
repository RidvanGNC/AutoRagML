"""Explanation — `explain()` öznitelik atıfı çıktısı (ADR 0037). Salt-veri sözleşme."""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract


class FeatureScore(Contract):
    """Tek öznitelik için global önem."""

    feature: str
    importance: float  # ≥ 0; SHAP: mean(|value|), permutation: ortalama |Δŷ| / skor düşüşü
    std: float | None = None


class Explanation(Contract):
    """Şampiyon öznitelik atıfı (ADR 0037).

    `method`: `"shap_tree"` | `"shap_linear"` | `"shap"` | `"permutation"` | `"unavailable"`.
    `per_sample`: örnek-başı SHAP değerleri (n×p) — yalnız SHAP + `per_sample=True`.
    """

    method: str
    feature_names: list[str] = Field(default_factory=list)
    global_importance: list[FeatureScore] = Field(default_factory=list)  # önem azalan sıralı
    base_value: float | None = None
    per_sample: list[list[float]] | None = None
    notes: list[str] = Field(default_factory=list)
