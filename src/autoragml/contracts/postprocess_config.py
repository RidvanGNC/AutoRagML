"""PostprocessConfig — ham tahmin düzeltme zinciri yapılandırması. DONDU (ADR 0017).

`RunConfig.postprocess` altında taşınır. **Varsayılanı tam no-op.** Fit gereken tek
adım `calibrate` — yalnız `champ_report.oof` üzerinden (ADR 0011). Sıra:
`calibrate → clip → round → business_rule`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from autoragml.contracts._base import Contract


class ClipConfig(Contract):
    """Alt/üst sınır kırpma. `auto_nonneg` + opsiyonel `auto_upper`."""

    lower: float | None = None
    upper: float | None = None
    auto_nonneg: bool = True  # profile hedef min ≥ 0 ve lower None ise → lower=0 (regresyon/forecasting)
    auto_upper_multiplier: float | None = None  # None → auto_upper kapalı; ör. 50.0
    auto_upper_percentile: float = Field(default=99.0, gt=0.0, lt=100.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> ClipConfig:
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            msg = "clip.lower, clip.upper'dan küçük olmalı"
            raise ValueError(msg)
        return self


class RoundConfig(Contract):
    """Yuvarlama. `threshold` modu DemandSensing eşikli yukarı-yuvarlama desenidir."""

    mode: Literal["off", "nearest", "threshold", "ceil", "floor"] = "off"
    decimals: int = Field(default=0, ge=0)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)  # yalnız mode="threshold"


class CalibrateConfig(Contract):
    """OOF üzerinde bias/ölçek düzeltme. `linear`/isotonic → v1.1."""

    method: Literal["off", "additive_bias", "multiplicative"] = "off"
    ratio_bounds: tuple[float, float] = (0.2, 5.0)  # multiplicative güvenlik bandı

    @model_validator(mode="after")
    def _check_ratio(self) -> CalibrateConfig:
        lo, hi = self.ratio_bounds
        if not 0.0 < lo < hi:
            msg = "calibrate.ratio_bounds: 0 < low < high olmalı"
            raise ValueError(msg)
        return self


class ConformalConfig(Contract):
    """Split-conformal tahmin aralığı — **v1.1** (sözleşme rezerve)."""

    enabled: bool = False
    coverage: float = Field(default=0.9, gt=0.0, lt=1.0)
    per_group: bool = False


class PostprocessConfig(Contract):
    """Tahmin düzeltme zinciri. Varsayılan: `enabled` ama tüm adımlar kapalı = no-op."""

    enabled: bool = True
    clip: ClipConfig = Field(default_factory=ClipConfig)
    round: RoundConfig = Field(default_factory=RoundConfig)
    calibrate: CalibrateConfig = Field(default_factory=CalibrateConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    apply_in_validation: bool = False  # v1: serving-only

    @model_validator(mode="after")
    def _v1_guard(self) -> PostprocessConfig:
        if self.conformal.enabled:
            msg = "postprocess.conformal.enabled v1'de desteklenmiyor (v1.1 — ADR 0017)"
            raise ValueError(msg)
        if self.apply_in_validation:
            msg = "postprocess.apply_in_validation v1'de desteklenmiyor (serving-only — ADR 0017)"
            raise ValueError(msg)
        return self
