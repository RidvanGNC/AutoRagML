"""analyzers.analyze — uçtan uca DataProfile + TaskSpec (ADR 0010)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.enums import Modality, Task
from autoragml.exceptions import DataLoadError
from autoragml.io import load_dataset


def _cfg(target: str = "y", **over: object):
    return resolve_run_config(target=target, overrides=over or None).config


def test_tabular_regression_end_to_end() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=300),
            "num": rng.normal(size=300),
            "cat": rng.choice(list("abcd"), 300),
            "row_id": np.arange(300),
        }
    )
    cfg = _cfg()
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    assert task.task is Task.REGRESSION
    assert task.modality is Modality.TABULAR
    assert profile.n_rows == 300
    assert {c.name for c in profile.columns} == {"y", "num", "cat", "row_id"}
    assert profile.timeseries is None
    assert profile.target_profile.name == "y"


def test_timeseries_end_to_end() -> None:
    weeks = pd.date_range("2025-01-06", periods=90, freq="W-MON")
    rows = []
    rng = np.random.default_rng(1)
    for g in ["A", "B", "C"]:
        for i, wk in enumerate(weeks):
            rows.append({"grp": g, "ds": wk, "y": 50 + 10 * np.sin(i / 52 * 6.28) + rng.normal(0, 4)})
    df = pd.DataFrame(rows)
    cfg = _cfg(time_col="ds", group_col="grp")
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    assert task.task is Task.FORECASTING
    assert task.modality is Modality.TIMESERIES
    assert profile.timeseries is not None
    assert profile.timeseries.freq in {"W-MON", "W"}
    assert len(profile.timeseries.per_series) == 3


def test_missing_target_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DataLoadError, match="Hedef kolon"):
        analyze(load_dataset(df, _cfg(target="a")), _cfg(target="nonexistent"))


def test_lazy_source_profiles_with_sampling(tmp_path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(size=2000), "x": rng.normal(size=2000)})
    p = tmp_path / "big.parquet"
    df.to_parquet(p)
    cfg = resolve_run_config(
        target="y",
        overrides={"io": {"eager_max_bytes": 1}, "analyzers": {"profiling_sample_rows": 1000}},
    ).config
    ds = load_dataset(p, cfg)
    profile, _ = analyze(ds, cfg)
    assert profile.n_rows == 2000  # tam sayım korunur
    assert profile.confidence <= 0.75  # örneklem → düşük güven
