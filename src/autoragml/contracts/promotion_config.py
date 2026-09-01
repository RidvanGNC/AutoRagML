"""PromotionConfig — mutlak eşik kapısı (ADR 0014, DemandSensing `promotion_rules`).

`RunConfig.promotion` altında taşınır. Şampiyon bu eşikleri geçemezse `PromotionResult.passed=False`
(seçim yine yapılır — kapı bilgilendirir, engellemez).
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract


class PromotionConfig(Contract):
    """Şampiyon adayın geçmesi beklenen mutlak eşikler."""

    smape_max: float | None = Field(default=35.0, ge=0.0)
    abs_bias_max: float | None = None
    rmse_max: float | None = None
    min_folds: int = Field(default=2, ge=1)
    require_leakage_pass: bool = True
