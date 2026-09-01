"""Postprocess adım yardımcıları — saf numpy (ADR 0017).

`calibrate` parametreleri OOF'tan hesaplanır; `clip`/`round` stateless.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from autoragml.contracts.postprocess_config import CalibrateConfig

_Arr = npt.NDArray[np.float64]


def resolve_clip_lower(
    explicit: float | None, *, auto_nonneg: bool, is_regression: bool, target_min: float | None
) -> float | None:
    """Kullanıcı verdiyse onunki; yoksa hedef ispatlı negatif-değilse 0.0."""
    if explicit is not None:
        return explicit
    if auto_nonneg and is_regression and target_min is not None and target_min >= 0.0:
        return 0.0
    return None


def resolve_clip_upper(
    explicit: float | None,
    *,
    multiplier: float | None,
    percentile: float,
    y_true: _Arr | None,
) -> float | None:
    """Kullanıcı verdiyse onunki; yoksa `multiplier` set + OOF varsa `pXX·mult`."""
    if explicit is not None:
        return explicit
    if multiplier is None or y_true is None or y_true.size == 0:
        return None
    finite = y_true[np.isfinite(y_true)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile) * multiplier)


def calibrate_params(
    cfg: CalibrateConfig, y_true: _Arr | None, y_pred: _Arr | None
) -> tuple[float | None, float | None, str | None]:
    """(bias, ratio, warning) — biri dolu diğeri None; OOF eksikse warning döner."""
    if cfg.method == "off":
        return None, None, None
    if y_true is None or y_pred is None or y_true.size == 0 or y_pred.size == 0:
        return None, None, f"calibrate.method={cfg.method} ama OOF yok → kalibrasyon atlandı"

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return None, None, "calibrate: OOF'ta sonlu değer yok → atlandı"

    if cfg.method == "additive_bias":
        return float(np.mean(yp - yt)), None, None

    # multiplicative
    denom = float(np.sum(yp))
    if denom == 0.0:
        return None, None, "calibrate=multiplicative ama Σy_pred=0 → atlandı"
    lo, hi = cfg.ratio_bounds
    ratio = float(np.clip(np.sum(yt) / denom, lo, hi))
    return None, ratio, None


def apply_round(a: _Arr, mode: str, decimals: int, threshold: float) -> _Arr:
    """Yuvarlama modları. `threshold`: kesir ≥ eşik → yukarı (tamsayı)."""
    if mode == "off":
        return a
    if mode == "nearest":
        return np.round(a, decimals)
    if mode == "ceil":
        return np.ceil(a)
    if mode == "floor":
        return np.floor(a)
    if mode == "threshold":
        base = np.floor(a)
        return base + (a - base >= threshold).astype(np.float64)
    return a
