"""engines.timeseries.reduction — zengin leakage-safe özellikler (ADR 0004 + 0025)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.contracts.enums import Modality, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.timeseries.reduction import build_reduction_features

_TASK = TaskSpec(
    task=Task.FORECASTING, modality=Modality.TIMESERIES, targets=["y"], time_col="ds", group_col="g", horizon=4
)


def _panel(n: int = 60) -> pd.DataFrame:
    weeks = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    return pd.DataFrame(
        {
            "g": np.repeat(["A", "B"], n),
            "ds": np.tile(weeks, 2),
            "y": np.concatenate([np.arange(float(n)), np.arange(float(n)) * 2]),
        }
    )


def test_reduction_adds_shifted_features() -> None:
    out, cols = build_reduction_features(_panel(), _TASK, horizon=4)
    assert "y_lag_4" in cols and "y_rollmean_4" in cols and "y_ewm_4" in cols
    a = out[out["g"] == "A"].sort_values("ds").reset_index(drop=True)
    assert np.isnan(a["y_lag_4"].iloc[0])
    assert a["y_lag_4"].iloc[10] == 6.0


def test_reduction_all_target_features_leakage_safe() -> None:
    """ADR 0025: her hedef-türevi kolon (lag/slag/roll/ewm/diff/seasonal) `shift ≥ horizon`."""
    out, cols = build_reduction_features(_panel(120), _TASK, horizon=4, season=12)
    a = out[out["g"] == "A"].sort_values("ds").reset_index(drop=True)  # y = step index
    derived = [
        c for c in cols
        if not c.startswith("y_cal_") and c not in {"y_step_index"}
    ]
    assert {"y_slag_12", "y_seasonal_rollmean", "y_diff1_lag_4"} <= set(cols)
    for t in range(30, 100):
        for col in derived:
            val = a[col].iloc[t]
            if np.isnan(val):
                continue
            # y(τ) = τ; rolling/diff τ-değerlerinin kombinasyonu ama hepsi ≤ y(t-4) = t-4
            assert val <= t - 4 + 1e-6, f"{col}@{t} = {val} > {t - 4}"


def test_reduction_calendar_features_no_shift() -> None:
    out, cols = build_reduction_features(_panel(), _TASK, horizon=4)
    cal = [c for c in cols if c.startswith("y_cal_")]
    assert {"y_cal_month", "y_cal_dayofweek", "y_cal_month_sin", "y_cal_dow_cos"} <= set(cal)
    assert not out[cal].isna().any().any()  # takvim ilk satırda bile dolu (warmup yok)
    assert out["y_cal_month_sin"].abs().max() <= 1.0


def test_reduction_seasonal_features_only_when_season_ge_2() -> None:
    _, no_season = build_reduction_features(_panel(), _TASK, horizon=4, season=1)
    _, with_season = build_reduction_features(_panel(120), _TASK, horizon=4, season=12)
    assert not any(c.startswith("y_slag_") for c in no_season)
    assert any(c.startswith("y_slag_") for c in with_season)
    assert "y_diffs_lag_4" in with_season and "y_diffs_lag_4" not in no_season


def test_reduction_group_isolation() -> None:
    out, _ = build_reduction_features(_panel(), _TASK, horizon=4)
    b = out[out["g"] == "B"].sort_values("ds").reset_index(drop=True)
    assert b["y_lag_4"].iloc[10] == 12.0  # B: y=2*step → y(6)=12


def test_reduction_sdiff_ref_leakage_safe() -> None:
    """ADR 0026: `y_sdiff_ref` = shift(H≥horizon) — horizon satırında train aktüeli."""
    out, cols = build_reduction_features(_panel(120), _TASK, horizon=4, season=12)
    assert "y_sdiff_ref" in cols
    a = out[out["g"] == "A"].sort_values("ds").reset_index(drop=True)  # y = step index
    for t in range(20, 100):
        v = a["y_sdiff_ref"].iloc[t]
        if not np.isnan(v):
            assert v <= t - 4 + 1e-9  # y(t-12) ≤ y(t-4)


def test_reduction_recursive_strategy_shift1_lags() -> None:
    """ADR 0026 B: recursive → `shift(1)` tabanı, lag 1..k_max; sdiff_ref üretilmez."""
    out, cols = build_reduction_features(_panel(120), _TASK, horizon=4, season=12, strategy="recursive")
    assert "y_lag_1" in cols and "y_lag_2" in cols
    assert "y_sdiff_ref" not in cols  # recursive'de seasonal target differencing yok
    assert "y_diff1_lag_1" in cols  # fark özelliği shift(1) tabanlı
    a = out[out["g"] == "A"].sort_values("ds").reset_index(drop=True)  # y = step index
    assert np.isnan(a["y_lag_1"].iloc[0])
    assert a["y_lag_1"].iloc[10] == 9.0  # bir önceki dönem
    # leakage: her hedef-türevi kolon ≤ y(t-1) = t-1
    derived = [c for c in cols if not c.startswith("y_cal_") and c != "y_step_index"]
    for t in range(30, 100):
        for col in derived:
            v = a[col].iloc[t]
            if not np.isnan(v):
                assert v <= t - 1 + 1e-6, f"{col}@{t} = {v} > {t - 1}"


def test_reduction_noop_without_time_col() -> None:
    task = TaskSpec(task=Task.REGRESSION, modality=Modality.TABULAR, targets=["y"])
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [3.0, 4.0]})
    out, cols = build_reduction_features(df, task, horizon=4)
    assert cols == []
    assert list(out.columns) == ["y", "x"]
