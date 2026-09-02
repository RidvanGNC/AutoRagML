"""Reduction feature engineering — hedeften lag/rolling/fark + takvim özellikleri (ADR 0004 + 0025).

**Leakage-safe by construction:** hedef-türevi her özellik `shift(≥ horizon)` tabanlı →
h-adım direkt tahminde bir test satırı yalnız train dönemindeki `y`'yi görür. Takvim
özellikleri `time_col`'dan doğrudan (gelecek tarihler için de bilinir — sızıntı yok).
Recursive/iteratif çok-adım → ADR 0026.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from autoragml.contracts.task_spec import TaskSpec
from autoragml.logging import get_logger

logger = get_logger(__name__)

_LAG_MULTIPLES = (0, 1, 2, 3)  # horizon + k
_ROLL_WINDOWS = (4, 8, 13)
_EWM_SPANS = (4, 12)
_SEASONAL_LAG_K = (0, 1, 2)  # H + k·s
_SEASONAL_ROLL_K = (0, 1, 2, 3)


def _calendar_features(out: pd.DataFrame, time_col: str, target: str) -> list[str]:
    """`time_col`'dan takvim özellikleri + döngüsel kodlama (sızıntı yok — tarih bilinir)."""
    dt = pd.to_datetime(out[time_col], errors="coerce").dt
    parts: dict[str, pd.Series] = {
        f"{target}_cal_month": dt.month.astype("float64"),
        f"{target}_cal_quarter": dt.quarter.astype("float64"),
        f"{target}_cal_dayofweek": dt.dayofweek.astype("float64"),
        f"{target}_cal_dayofyear": dt.dayofyear.astype("float64"),
        f"{target}_cal_weekofyear": dt.isocalendar().week.astype("float64"),
        f"{target}_cal_is_month_start": dt.is_month_start.astype("float64"),
        f"{target}_cal_is_month_end": dt.is_month_end.astype("float64"),
        f"{target}_cal_is_quarter_start": dt.is_quarter_start.astype("float64"),
    }
    # döngüsel: lineer/MLP için sin/cos, ağaçlar ham int'i kullanır
    month = dt.month.to_numpy(dtype=np.float64)
    dow = dt.dayofweek.to_numpy(dtype=np.float64)
    parts[f"{target}_cal_month_sin"] = pd.Series(np.sin(2 * np.pi * month / 12.0), index=out.index)
    parts[f"{target}_cal_month_cos"] = pd.Series(np.cos(2 * np.pi * month / 12.0), index=out.index)
    parts[f"{target}_cal_dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 7.0), index=out.index)
    parts[f"{target}_cal_dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 7.0), index=out.index)
    for name, series in parts.items():
        out[name] = series
    return list(parts)


def build_reduction_features(
    frame: pd.DataFrame,
    task: TaskSpec,
    *,
    horizon: int,
    season: int = 1,
    add_calendar: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Long-format TS frame'ine zengin hedef-türevi + takvim özellikleri ekle → (frame, yeni_kolonlar)."""
    target = task.targets[0]
    time_col = task.time_col
    group_col = task.group_col
    if time_col is None or time_col not in frame.columns:
        return frame, []

    out = frame.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    sort_cols = [c for c in (group_col, time_col) if c and c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    y = pd.to_numeric(out[target], errors="coerce")
    grouper = out[group_col] if group_col and group_col in out.columns else pd.Series(0, index=out.index)
    grouped = y.groupby(grouper)
    base = grouped.shift(horizon)  # h-adım güvenli taban
    new_cols: list[str] = []

    # --- lag'ler: horizon + {0,1,2,3} ---
    for k in _LAG_MULTIPLES:
        col = f"{target}_lag_{horizon + k}"
        out[col] = grouped.shift(horizon + k)
        new_cols.append(col)

    # --- mevsim-hizalı lag'ler: H = ceil(h/s)·s ---
    s = max(int(season), 1)
    seasonal = s >= 2 and s <= 400
    if seasonal:
        h_season = int(math.ceil(horizon / s) * s)
        for k in _SEASONAL_LAG_K:
            lag = h_season + k * s
            col = f"{target}_slag_{lag}"
            out[col] = grouped.shift(lag)
            new_cols.append(col)
        # seasonal target differencing referansı (ADR 0026): y_{t-s} — s ≥ h iken train aktüeli
        out[f"{target}_sdiff_ref"] = grouped.shift(h_season)
        new_cols.append(f"{target}_sdiff_ref")

    # --- rolling (base = shift(horizon)): mean/std/min/max ---
    windows = (*_ROLL_WINDOWS, s) if seasonal else _ROLL_WINDOWS
    for w in dict.fromkeys(windows):  # tekrarsız
        bg = base.groupby(grouper)
        out[f"{target}_rollmean_{w}"] = bg.transform(lambda x, _w=w: x.rolling(_w, min_periods=1).mean())
        out[f"{target}_rollstd_{w}"] = bg.transform(lambda x, _w=w: x.rolling(_w, min_periods=1).std())
        out[f"{target}_rollmin_{w}"] = bg.transform(lambda x, _w=w: x.rolling(_w, min_periods=1).min())
        out[f"{target}_rollmax_{w}"] = bg.transform(lambda x, _w=w: x.rolling(_w, min_periods=1).max())
        new_cols += [f"{target}_roll{stat}_{w}" for stat in ("mean", "std", "min", "max")]

    # --- mevsimsel rolling: aynı-mevsim geçmiş dönemlerin ortalaması ---
    if seasonal:
        seas_lags = pd.concat(
            [grouped.shift(h_season + k * s) for k in _SEASONAL_ROLL_K], axis=1
        )
        out[f"{target}_seasonal_rollmean"] = seas_lags.mean(axis=1)
        out[f"{target}_seasonal_rollstd"] = seas_lags.std(axis=1)
        new_cols += [f"{target}_seasonal_rollmean", f"{target}_seasonal_rollstd"]

    # --- ewm ---
    for span in _EWM_SPANS:
        out[f"{target}_ewm_{span}"] = base.groupby(grouper).transform(
            lambda x, _sp=span: x.ewm(span=_sp, min_periods=1).mean()
        )
        new_cols.append(f"{target}_ewm_{span}")

    # --- fark (difference) özellikleri ---
    out[f"{target}_diff1_lag_{horizon}"] = grouped.shift(horizon) - grouped.shift(horizon + 1)
    new_cols.append(f"{target}_diff1_lag_{horizon}")
    if seasonal:
        out[f"{target}_diffs_lag_{horizon}"] = grouped.shift(horizon) - grouped.shift(horizon + s)
        new_cols.append(f"{target}_diffs_lag_{horizon}")

    # --- zaman-içi konum ---
    out[f"{target}_step_index"] = grouped.cumcount().astype("float64")
    new_cols.append(f"{target}_step_index")

    # --- takvim ---
    if add_calendar:
        new_cols += _calendar_features(out, time_col, target)

    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(
        "[reduction] %d özellik (shift≥%d, s=%d%s, leakage-safe).",
        len(new_cols), horizon, s, ", takvim" if add_calendar else "",
    )
    return out, new_cols
