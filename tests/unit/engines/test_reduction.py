"""engines.timeseries.reduction — leakage-safe lag/rolling özellikleri (ADR 0004)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.contracts.enums import Modality, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.timeseries.reduction import build_reduction_features

_TASK = TaskSpec(
    task=Task.FORECASTING, modality=Modality.TIMESERIES, targets=["y"], time_col="ds", group_col="g", horizon=4
)


def _panel() -> pd.DataFrame:
    weeks = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    return pd.DataFrame(
        {
            "g": np.repeat(["A", "B"], 60),
            "ds": np.tile(weeks, 2),
            "y": np.concatenate([np.arange(60.0), np.arange(60.0) * 2]),
        }
    )


def test_reduction_adds_shifted_features() -> None:
    out, cols = build_reduction_features(_panel(), _TASK, horizon=4)
    assert "y_lag_4" in cols and "y_rollmean_4" in cols and "y_ewm_4" in cols
    # A grubu, y = step index. lag_4 satır t → y(t-4) = t-4 (t>=4).
    a = out[out["g"] == "A"].sort_values("ds").reset_index(drop=True)
    assert np.isnan(a["y_lag_4"].iloc[0])  # ilk 4 satır warmup
    assert a["y_lag_4"].iloc[10] == 6.0  # y(6) = 6


def test_reduction_leakage_safe_min_shift() -> None:
    out, _ = build_reduction_features(_panel(), _TASK, horizon=4)
    a = out[out["g"] == "A"].sort_values("ds").reset_index(drop=True)
    # her lag kolonu en az horizon kaydırılmış: satır t'de lag <= y(t-4)
    for t in range(10, 40):
        for col in [c for c in out.columns if c.startswith("y_lag_")]:
            val = a[col].iloc[t]
            if not np.isnan(val):
                assert val <= t - 4 + 1e-9  # y(t-4) veya daha eski


def test_reduction_group_isolation() -> None:
    out, _ = build_reduction_features(_panel(), _TASK, horizon=4)
    b = out[out["g"] == "B"].sort_values("ds").reset_index(drop=True)
    # B: y = 2*step. lag_4 satır 10 → y(6) = 12 (A'nın 6 değil)
    assert b["y_lag_4"].iloc[10] == 12.0


def test_reduction_noop_without_time_col() -> None:
    task = TaskSpec(task=Task.REGRESSION, modality=Modality.TABULAR, targets=["y"])
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [3.0, 4.0]})
    out, cols = build_reduction_features(df, task, horizon=4)
    assert cols == []
    assert list(out.columns) == ["y", "x"]
