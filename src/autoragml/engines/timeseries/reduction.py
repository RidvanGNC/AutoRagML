"""Reduction feature engineering — hedeften lag/rolling/fark + takvim özellikleri (ADR 0004 + 0025).

**Leakage-safe by construction:** hedef-türevi her özellik `shift(≥ horizon)` tabanlı →
h-adım direkt tahminde bir test satırı yalnız train dönemindeki `y`'yi görür. Takvim
özellikleri `time_col`'dan doğrudan (gelecek tarihler için de bilinir — sızıntı yok).
Recursive/iteratif çok-adım → ADR 0026.
"""

from __future__ import annotations

import math
from typing import Literal

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
    strategy: Literal["direct", "recursive"] = "direct",
    max_lag: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Long-format TS frame'ine zengin hedef-türevi + takvim özellikleri ekle → (frame, yeni_kolonlar).

    `direct` (ADR 0004): `shift(horizon)` tabanı, h-adım direkt. `recursive` (ADR 0026):
    `shift(1)` tabanı, lag 1..max_lag; model 1-adım eğitilir, serving `FittedRecursivePipeline`.
    """
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
    s = max(int(season), 1)
    seasonal = s >= 2 and s <= 400

    recursive = strategy == "recursive"
    base_shift = 1 if recursive else horizon
    if recursive:
        k_max = max_lag or max(horizon, 3 * s if seasonal else horizon, 12)
        lag_list = list(range(1, k_max + 1))
    else:
        lag_list = [horizon + k for k in _LAG_MULTIPLES]
    base = grouped.shift(base_shift)
    new_cols: list[str] = []

    for lag in lag_list:
        col = f"{target}_lag_{lag}"
        out[col] = grouped.shift(lag)
        new_cols.append(col)

    # --- mevsim-hizalı lag'ler ---
    h_season = int(math.ceil(base_shift / s) * s) if recursive else int(math.ceil(horizon / s) * s)
    if seasonal:
        for k in _SEASONAL_LAG_K:
            lag = h_season + k * s
            col = f"{target}_slag_{lag}"
            if col not in out.columns:
                out[col] = grouped.shift(lag)
                new_cols.append(col)
        if not recursive:  # seasonal target differencing (ADR 0026-A) yalnız direct
            out[f"{target}_sdiff_ref"] = grouped.shift(h_season)
            new_cols.append(f"{target}_sdiff_ref")

    # --- rolling (base = shift(base_shift)): mean/std/min/max ---
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
    out[f"{target}_diff1_lag_{base_shift}"] = grouped.shift(base_shift) - grouped.shift(base_shift + 1)
    new_cols.append(f"{target}_diff1_lag_{base_shift}")
    if seasonal:
        out[f"{target}_diffs_lag_{base_shift}"] = grouped.shift(base_shift) - grouped.shift(base_shift + s)
        new_cols.append(f"{target}_diffs_lag_{base_shift}")

    # --- zaman-içi konum ---
    out[f"{target}_step_index"] = grouped.cumcount().astype("float64")
    new_cols.append(f"{target}_step_index")

    # --- takvim ---
    if add_calendar:
        new_cols += _calendar_features(out, time_col, target)

    new_cols = list(dict.fromkeys(new_cols))  # tekrarsız (recursive'de lag ↔ slag çakışabilir)
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(
        "[reduction] %d özellik (%s, shift≥%d, s=%d%s, leakage-safe).",
        len(new_cols), strategy, base_shift, s, ", takvim" if add_calendar else "",
    )
    return out, new_cols
