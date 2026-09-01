"""Zaman serisi tanısı — `TimeSeriesProfile` (ADR 0010).

`pandas.infer_freq` + freq→periyot sözlüğü + numpy ACF ile mevsimsellik doğrulaması
(statsmodels yok). Trend gücü OLS R². Per-series ADI/CV² intermittency **ölçüm**.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.analyzers.intermittent import classify_demand
from autoragml.contracts.analyzer_config import TimeSeriesAnalyzerConfig
from autoragml.contracts.data_profile import SeasonalityHint, SeriesProfile, TimeSeriesProfile
from autoragml.contracts.enums import ClassificationScheme, PerSeriesDetail
from autoragml.logging import get_logger

logger = get_logger(__name__)

# Pandas frekans kodu → aday mevsimsel periyotlar (Nixtla deseni).
_FREQ_SEASONS: dict[str, list[int]] = {
    "H": [24, 168],
    "h": [24, 168],
    "D": [7, 30, 365],
    "B": [5, 21],
    "W": [52],
    "W-MON": [52],
    "W-SUN": [52],
    "M": [12],
    "MS": [12],
    "ME": [12],
    "Q": [4],
    "QS": [4],
    "Y": [1],
    "YS": [1],
}
_SAMPLE_SERIES = 200
_MIN_ACF_STRENGTH = 0.2


def _infer_freq(timestamps: pd.Series) -> tuple[str | None, float]:
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(ts) < 3:
        return None, 0.0
    try:
        inferred = pd.infer_freq(pd.DatetimeIndex(ts))
    except (ValueError, TypeError):
        inferred = None
    diffs = ts.diff().dropna()
    if diffs.empty:
        return inferred, 0.0
    modal = diffs.mode()
    if modal.empty:
        return inferred, 0.0
    confidence = float((diffs == modal.iloc[0]).mean())
    if inferred is None:
        days = modal.iloc[0].days
        if days == 7:
            inferred = f"W-{ts.iloc[0].strftime('%a').upper()}"
        else:
            inferred = {1: "D", 28: "MS", 30: "MS", 31: "MS", 90: "QS", 365: "YS"}.get(days)
    return inferred, confidence


def _seasonal_candidates(freq: str | None, max_period: int) -> list[int]:
    if freq is None:
        return []
    key = freq.split("-")[0] if freq not in _FREQ_SEASONS else freq
    return [p for p in _FREQ_SEASONS.get(freq, _FREQ_SEASONS.get(key, [])) if 2 <= p <= max_period]


def _acf_strength(values: npt.NDArray[np.float64], lag: int) -> float:
    """Lag-k örneklem otokorelasyonunun mutlak değeri."""
    if len(values) <= lag + 3:
        return 0.0
    a = values[:-lag]
    b = values[lag:]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _trend_strength(values: npt.NDArray[np.float64]) -> float:
    n = len(values)
    if n < 4 or float(np.var(values)) == 0.0:
        return 0.0
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, values, 1)
    fitted = slope * t + intercept
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    return max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _series_gaps(timestamps: pd.Series, freq: str | None) -> int:
    if freq is None:
        return 0
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(ts) < 2:
        return 0
    try:
        full = pd.date_range(ts.min(), ts.max(), freq=freq)
    except (ValueError, TypeError):
        return 0
    return int(len(full) - len(ts))


def diagnose_timeseries(
    frame: pd.DataFrame,
    *,
    target: str,
    time_col: str,
    group_col: str | None,
    config: TimeSeriesAnalyzerConfig,
) -> tuple[TimeSeriesProfile, list[str]]:
    """Zaman serisi profili + uyarılar."""
    warnings: list[str] = []
    if config.classification_scheme is ClassificationScheme.KH:
        warnings.append("classification_scheme=kh v1'de SBC'ye düşüyor (Kostenko-Hyndman — takip).")

    work = frame[[c for c in {time_col, group_col, target} if c is not None]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])

    global_freq, freq_conf = _infer_freq(work[time_col])
    span = None
    if not work.empty:
        span = (work[time_col].min().isoformat(), work[time_col].max().isoformat())

    if group_col is None:
        groups: list[tuple[str, pd.DataFrame]] = [("__single__", work)]
    else:
        groups = [(str(g), d) for g, d in work.groupby(group_col, observed=True)]

    detail = config.per_series_detail
    sampled_groups = groups
    if detail is PerSeriesDetail.SAMPLED and len(groups) > _SAMPLE_SERIES:
        sampled_groups = groups[:_SAMPLE_SERIES]
        warnings.append(f"per_series_detail=sampled: {len(groups)} grubun ilk {_SAMPLE_SERIES}'i profillendi.")

    per_series: list[SeriesProfile] = []
    class_counts: dict[str, int] = {}
    regular_flags: list[bool] = []
    gaps: dict[str, int] = {}
    if detail is not PerSeriesDetail.SUMMARY_ONLY:
        for gid, gdf in sampled_groups:
            gdf = gdf.sort_values(time_col)
            y = pd.to_numeric(gdf[target], errors="coerce").fillna(0.0)
            label, adi, cv2, zero_ratio, nz = classify_demand(
                y,
                scheme=config.classification_scheme,
                adi_threshold=config.adi_threshold,
                cv2_threshold=config.cv2_threshold,
                min_history_periods=config.min_history_periods,
                min_non_zero_points=config.min_non_zero_points,
            )
            recent = y.tail(config.recent_window_periods)
            recent_label, *_ = classify_demand(
                recent,
                scheme=config.classification_scheme,
                adi_threshold=config.adi_threshold,
                cv2_threshold=config.cv2_threshold,
                min_history_periods=min(config.min_history_periods, len(recent)),
                min_non_zero_points=1,
            )
            g_freq, _ = _infer_freq(gdf[time_col])
            n_gaps = _series_gaps(gdf[time_col], g_freq)
            regular_flags.append(n_gaps == 0)
            if group_col is not None and n_gaps > 0:
                gaps[str(gid)] = n_gaps
            per_series.append(
                SeriesProfile(
                    group=str(gid),
                    n_obs=int(len(y)),
                    n_nonzero=int(nz),
                    zero_ratio=float(zero_ratio),
                    adi=float(adi) if np.isfinite(adi) else None,
                    cv2=float(cv2) if np.isfinite(cv2) else None,
                    history_weeks=int(len(y)),
                    intermittency_class=label,
                    intermittency_class_recent=recent_label,
                    class_changed_over_time=bool(recent_label != label),
                )
            )
            class_counts[label.value] = class_counts.get(label.value, 0) + 1

    # Mevsimsellik + trend: zaman damgası bazında toplulaştırılmış (toplam) seri
    # üzerinden — tek bir keyfi grup yerine tüm talebin sinyali.
    agg = (
        pd.to_numeric(work[target], errors="coerce")
        .groupby(work[time_col])
        .sum()
        .sort_index()
    )
    ref_values: npt.NDArray[np.float64] = np.asarray(agg.to_numpy(), dtype=np.float64)
    seasonality: list[SeasonalityHint] = []
    for period in _seasonal_candidates(global_freq, config.max_seasonality_period):
        strength = _acf_strength(ref_values, period)
        if strength >= _MIN_ACF_STRENGTH:
            seasonality.append(SeasonalityHint(period=period, strength=min(1.0, strength)))
    trend_strength = _trend_strength(ref_values) if ref_values.size else None

    profile = TimeSeriesProfile(
        freq=global_freq,
        freq_confidence=freq_conf,
        span=span,
        regular=bool(regular_flags) and all(regular_flags),
        gaps=gaps,
        seasonality=seasonality,
        trend_strength=trend_strength,
        stationarity_pvalue=None,  # statsmodels opsiyonel — v1'de doldurulmuyor
        per_series=per_series,
        intermittency_summary=class_counts,
        classification_scheme=config.classification_scheme,
        per_series_detail=detail,
    )
    return profile, warnings
