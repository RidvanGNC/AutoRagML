"""Reduction feature engineering — hedeften lag/rolling özellikleri (ADR 0004).

**Leakage-safe:** tüm hedef-türevi özellikler `shift(horizon)` ile üretilir → h-adım
direkt tahminde bir test satırı yalnız train dönemindeki `y`'yi görür (rolling-origin
splitter test bloğunu tam `horizon` dönem tutar). Recursive/iteratif çok-adım v1.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.contracts.task_spec import TaskSpec
from autoragml.logging import get_logger

logger = get_logger(__name__)

_LAG_MULTIPLES = (0, 1, 2, 3)  # horizon, 2*... değil — horizon + k
_ROLL_WINDOWS = (4, 8, 13)
_EWM_SPANS = (4, 12)


def build_reduction_features(
    frame: pd.DataFrame, task: TaskSpec, *, horizon: int
) -> tuple[pd.DataFrame, list[str]]:
    """Long-format TS frame'ine hedef-türevi özellikler ekle. Döner: (frame, yeni_kolonlar)."""
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

    # h-adım güvenli taban: her şey en az `horizon` kaydırılır.
    base = grouped.shift(horizon)
    new_cols: list[str] = []

    for k in _LAG_MULTIPLES:
        col = f"{target}_lag_{horizon + k}"
        out[col] = grouped.shift(horizon + k)
        new_cols.append(col)

    for w in _ROLL_WINDOWS:
        rmean = base.groupby(grouper).transform(lambda s, _w=w: s.rolling(_w, min_periods=1).mean())
        rstd = base.groupby(grouper).transform(lambda s, _w=w: s.rolling(_w, min_periods=1).std())
        out[f"{target}_rollmean_{w}"] = rmean
        out[f"{target}_rollstd_{w}"] = rstd
        new_cols += [f"{target}_rollmean_{w}", f"{target}_rollstd_{w}"]

    for span in _EWM_SPANS:
        ewm = base.groupby(grouper).transform(lambda s, _s=span: s.ewm(span=_s, min_periods=1).mean())
        out[f"{target}_ewm_{span}"] = ewm
        new_cols.append(f"{target}_ewm_{span}")

    # zaman-içi konum (trend proxy)
    out[f"{target}_step_index"] = grouped.cumcount().astype("float64")
    new_cols.append(f"{target}_step_index")

    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    logger.info("[reduction] %d hedef-türevi özellik eklendi (shift>=%d, leakage-safe).", len(new_cols), horizon)
    return out, new_cols
