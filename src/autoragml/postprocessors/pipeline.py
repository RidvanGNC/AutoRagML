"""Postprocessor + FittedPostprocessor — fit-ayrımlı düzeltme zinciri (ADR 0017 + 0044).

Sıra: `calibrate → clip → conformal genişliği (ADR 0044) → round → business_rule`. `Postprocessor.fit`
yalnız `engines/champion.refit_champion` içinde, `champ_report.oof` üzerinden çağrılır.
`FittedPostprocessor` immutable (`__slots__`); `business_rule` interfaces tarafından enjekte edilir.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from autoragml.contracts.postprocess_config import PostprocessConfig
from autoragml.logging import get_logger
from autoragml.postprocessors.conformal import ConformalFit, fit_conformal
from autoragml.postprocessors.steps import (
    apply_round,
    calibrate_params,
    resolve_clip_lower,
    resolve_clip_upper,
)

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
BusinessRule = Callable[[_Arr], npt.ArrayLike]


class FittedPostprocessor:
    """Öğrenilmiş/çözülmüş düzeltme adımları — saf `apply`."""

    __slots__ = (
        "_bias",
        "_business_rule",
        "_conformal",
        "_lower",
        "_ratio",
        "_round_decimals",
        "_round_mode",
        "_round_threshold",
        "_summary",
        "_upper",
    )

    def __init__(
        self,
        *,
        bias: float | None,
        ratio: float | None,
        lower: float | None,
        upper: float | None,
        round_mode: str,
        round_decimals: int,
        round_threshold: float,
        summary: dict[str, Any],
        business_rule: BusinessRule | None = None,
        conformal: ConformalFit | None = None,
    ) -> None:
        self._bias = bias
        self._ratio = ratio
        self._lower = lower
        self._upper = upper
        self._round_mode = round_mode
        self._round_decimals = round_decimals
        self._round_threshold = round_threshold
        self._summary = summary
        self._business_rule = business_rule
        self._conformal = conformal

    @property
    def is_noop(self) -> bool:
        return (
            self._bias is None
            and self._ratio is None
            and self._lower is None
            and self._upper is None
            and self._round_mode == "off"
            and self._business_rule is None
            and self._conformal is None
        )

    @property
    def has_conformal(self) -> bool:
        """ADR 0044: `predict_interval` için conformal genişlik fit edildi mi."""
        return self._conformal is not None

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self._summary)

    def with_business_rule(self, fn: BusinessRule | None) -> FittedPostprocessor:
        """`interfaces` katmanı iş kuralı hook'unu enjekte eder → yeni immutable nesne."""
        return FittedPostprocessor(
            bias=self._bias,
            ratio=self._ratio,
            lower=self._lower,
            upper=self._upper,
            round_mode=self._round_mode,
            round_decimals=self._round_decimals,
            round_threshold=self._round_threshold,
            summary={**self._summary, "business_rule": fn is not None},
            business_rule=fn,
            conformal=self._conformal,
        )

    def _point(self, y_pred: npt.ArrayLike) -> _Arr:
        """Bias/ölçek + clip — round/business_rule'dan ÖNCEki nokta (conformal merkezi de bu)."""
        out = np.array(y_pred, dtype=np.float64, copy=True).ravel()
        if self._bias is not None:
            out = out - self._bias
        elif self._ratio is not None:
            out = out * self._ratio
        if self._lower is not None or self._upper is not None:
            out = np.clip(out, self._lower, self._upper)
        return out

    def apply(self, y_pred: npt.ArrayLike) -> _Arr:
        """Ham nokta tahmini → düzeltilmiş tahmin (ADR 0017 sırası)."""
        out = self._point(y_pred)
        out = apply_round(out, self._round_mode, self._round_decimals, self._round_threshold)
        if self._business_rule is not None:
            out = np.asarray(self._business_rule(out), dtype=np.float64).ravel()
        return out

    def interval(
        self,
        y_pred: npt.ArrayLike,
        *,
        group: npt.NDArray[np.object_] | None = None,
        coverage: float | None = None,
    ) -> tuple[_Arr, _Arr]:
        """Nokta tahmin ± split-conformal genişlik → `(lower, upper)` (ADR 0044).

        Conformal fit edilmemişse (`postprocess.conformal.enabled=False` veya yetersiz OOF)
        `(point, point)` döner — sıfır-genişlik, sessizce yanlış aralık üretmez.
        """
        point = self._point(y_pred)
        if self._conformal is None:
            rounded = apply_round(point, self._round_mode, self._round_decimals, self._round_threshold)
            return rounded, rounded
        width = self._conformal.width_for(point.size, group, coverage)
        lower, upper = point - width, point + width
        if self._lower is not None or self._upper is not None:
            lower = np.clip(lower, self._lower, self._upper)
            upper = np.clip(upper, self._lower, self._upper)
        lower = apply_round(lower, self._round_mode, self._round_decimals, self._round_threshold)
        upper = apply_round(upper, self._round_mode, self._round_decimals, self._round_threshold)
        return lower, upper


class Postprocessor:
    """Fit edilmemiş düzeltme zinciri — `build_postprocessor` üretir."""

    __slots__ = ("_cfg", "_is_regression", "_target_min")

    def __init__(
        self, cfg: PostprocessConfig, *, is_regression: bool, target_min: float | None
    ) -> None:
        self._cfg = cfg
        self._is_regression = is_regression
        self._target_min = target_min

    @property
    def is_active(self) -> bool:
        """Herhangi bir adım *çözülebilir* mi (fit'ten önce kaba kontrol)."""
        if not self._cfg.enabled:
            return False
        c = self._cfg.clip
        return bool(
            c.lower is not None
            or c.upper is not None
            or c.auto_nonneg
            or c.auto_upper_multiplier is not None
            or self._cfg.round.mode != "off"
            or self._cfg.calibrate.method != "off"
            or self._cfg.conformal.enabled
        )

    def fit(
        self,
        y_true: _Arr | None = None,
        y_pred: _Arr | None = None,
        *,
        group: npt.NDArray[np.object_] | None = None,
    ) -> FittedPostprocessor:
        """OOF'tan kalibrasyon + clip sınırlarını + conformal genişliğini (ADR 0044) çöz."""
        cfg = self._cfg
        bias, ratio, warn = calibrate_params(cfg.calibrate, y_true, y_pred)
        if warn:
            logger.warning("[postprocess] %s", warn)

        lower = resolve_clip_lower(
            cfg.clip.lower,
            auto_nonneg=cfg.clip.auto_nonneg,
            is_regression=self._is_regression,
            target_min=self._target_min,
        )
        upper = resolve_clip_upper(
            cfg.clip.upper,
            multiplier=cfg.clip.auto_upper_multiplier,
            percentile=cfg.clip.auto_upper_percentile,
            y_true=y_true,
        )

        conformal = None
        if cfg.conformal.enabled and y_true is not None and y_pred is not None:
            # ADR 0044: kalibrasyon seti = bu OOF (calibrate ile aynı kaynak, ADR 0011 leakage-safe).
            conformal = fit_conformal(
                y_true, y_pred, coverage=cfg.conformal.coverage,
                group=group if cfg.conformal.per_group else None,
                per_group=cfg.conformal.per_group,
            )
            if conformal is None:
                logger.warning("[postprocess] conformal: yetersiz OOF örneklemi — aralık üretilmeyecek")

        summary: dict[str, Any] = {
            "calibrate": {"method": cfg.calibrate.method, "bias": bias, "ratio": ratio},
            "clip": {"lower": lower, "upper": upper},
            "round": cfg.round.mode,
            "conformal": {
                "enabled": cfg.conformal.enabled, "coverage": cfg.conformal.coverage,
                "per_group": cfg.conformal.per_group, "fitted": conformal is not None,
            },
        }
        return FittedPostprocessor(
            bias=bias,
            ratio=ratio,
            lower=lower,
            upper=upper,
            round_mode=cfg.round.mode,
            round_decimals=cfg.round.decimals,
            round_threshold=cfg.round.threshold,
            summary=summary,
            conformal=conformal,
        )
