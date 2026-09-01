"""ADI / CV² + SBC talep sınıflandırması (ADR 0010).

DemandSensing `intermittent_features` deseninden portlandı. **Ölçüm** — router değil:
sınıf `dynamics`'e ipucu verir, model ailesini kısıtlamaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.contracts.enums import ClassificationScheme, IntermittencyClass

_SBC = ClassificationScheme.SBC


def compute_adi_cv2(series: pd.Series) -> tuple[float, float, float, int]:
    """Döner: (adi, cv2, zero_ratio, non_zero_count). Tamamen sıfır seri → (n, inf, ., 0)."""
    y = pd.to_numeric(series, errors="coerce").fillna(0.0)
    total_n = int(len(y))
    if total_n == 0:
        return 0.0, 0.0, 0.0, 0
    non_zero = y[y > 0.0]
    non_zero_count = int(len(non_zero))
    zero_ratio = float((y <= 0.0).mean())
    if non_zero_count == 0:
        return float(total_n), float("inf"), zero_ratio, 0
    adi = float(total_n / non_zero_count)
    mean_nz = float(non_zero.mean())
    if mean_nz <= 0.0:
        cv2 = float("inf")
    else:
        cv = float(non_zero.std(ddof=0) / mean_nz)
        cv2 = cv * cv
    return adi, cv2, zero_ratio, non_zero_count


def classify_sbc(*, adi: float, cv2: float, adi_threshold: float, cv2_threshold: float) -> IntermittencyClass:
    """Syntetos-Boylan-Croston 4-sınıf (lineer eşik)."""
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return IntermittencyClass.LUMPY
    smooth_adi = adi < adi_threshold
    smooth_cv2 = cv2 < cv2_threshold
    if smooth_adi and smooth_cv2:
        return IntermittencyClass.SMOOTH
    if not smooth_adi and smooth_cv2:
        return IntermittencyClass.INTERMITTENT
    if smooth_adi and not smooth_cv2:
        return IntermittencyClass.ERRATIC
    return IntermittencyClass.LUMPY


def classify_demand(
    y: pd.Series,
    *,
    scheme: ClassificationScheme,
    adi_threshold: float,
    cv2_threshold: float,
    min_history_periods: int,
    min_non_zero_points: int,
) -> tuple[IntermittencyClass, float, float, float, int]:
    """Bir seriyi sınıflandır. Döner: (sınıf, adi, cv2, zero_ratio, non_zero_count).

    `kh` şeması v1'de SBC'ye düşer (uyarı çağıran tarafta verilir).
    """
    adi, cv2, zero_ratio, nz = compute_adi_cv2(y)
    if len(y) < min_history_periods or nz < min_non_zero_points:
        return IntermittencyClass.INSUFFICIENT, adi, cv2, zero_ratio, nz
    _ = scheme  # kh: v1'de SBC ile aynı (Kostenko-Hyndman non-lineer sınır — takip)
    label = classify_sbc(
        adi=adi, cv2=cv2, adi_threshold=adi_threshold, cv2_threshold=cv2_threshold
    )
    return label, adi, cv2, zero_ratio, nz
