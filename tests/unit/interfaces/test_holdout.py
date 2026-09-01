"""interfaces.holdout — nihai holdout carve + skorlama (ADR 0020)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import Modality, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.interfaces.holdout import split_holdout

_REG = TaskSpec(task=Task.REGRESSION, modality=Modality.TABULAR, targets=["y"])
_FC = TaskSpec(
    task=Task.FORECASTING, modality=Modality.TIMESERIES, targets=["y"],
    time_col="ds", group_col="g", horizon=4,
)


def _cfg(**ov):
    return resolve_run_config(target="y", overrides=ov).config


def test_random_holdout_disjoint_and_sized() -> None:
    df = pd.DataFrame({"y": np.arange(400.0), "x": np.arange(400.0)})
    split = split_holdout(df, _cfg(), _REG)
    assert split is not None and not split.is_timeseries
    assert split.n_holdout == 80  # 0.2 * 400
    assert len(split.train) == 320
    # train + holdout ayrık, birlikte tüm satırlar
    both = pd.concat([split.train, split.scoring_frame])["y"].sort_values().to_numpy()
    np.testing.assert_array_equal(both, np.arange(400.0))


def test_holdout_skipped_when_too_small() -> None:
    df = pd.DataFrame({"y": np.arange(80.0), "x": np.arange(80.0)})
    assert split_holdout(df, _cfg(), _REG) is None  # 80 < ceil(100/0.8)+1


def test_ts_holdout_is_last_horizon_periods() -> None:
    weeks = pd.date_range("2022-01-03", periods=120, freq="W-MON")
    rows = [{"g": g, "ds": wk, "y": float(i)} for g in ("A", "B") for i, wk in enumerate(weeks)]
    df = pd.DataFrame(rows)
    split = split_holdout(df, _cfg(time_col="ds", group_col="g"), _FC)
    assert split is not None and split.is_timeseries
    assert split.holdout_mask is not None
    # son 4 hafta × 2 grup = 8 satır
    assert split.n_holdout == 8
    hold_weeks = pd.to_datetime(split.scoring_frame.loc[split.holdout_mask, "ds"]).nunique()
    assert hold_weeks == 4
    # train holdout dönemlerini içermez
    assert pd.to_datetime(split.train["ds"]).max() < pd.to_datetime(
        split.scoring_frame.loc[split.holdout_mask, "ds"]
    ).min()
