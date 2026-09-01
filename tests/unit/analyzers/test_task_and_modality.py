"""analyzers.modality + task_inference (ADR 0010)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers.modality import detect_modality
from autoragml.analyzers.profiling import build_column_profiles
from autoragml.analyzers.task_inference import infer_task
from autoragml.config import resolve_run_config
from autoragml.contracts.analyzer_config import ThresholdConfig
from autoragml.contracts.enums import Layout, Modality, Task

THR = ThresholdConfig()


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides=over or None).config


def _target_profile(df: pd.DataFrame):
    profs = build_column_profiles(df, target="y", thr=THR, sampled=False)
    return next(p for p in profs if p.name == "y")


def test_modality_time_col_forces_timeseries() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0], "ds": pd.to_datetime(["2026-01-01", "2026-01-08"])})
    mod, _ = detect_modality(df, _cfg(time_col="ds"), layout=Layout.LONG)
    assert mod is Modality.TIMESERIES


def test_modality_plain_tabular() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
    mod, _ = detect_modality(df, _cfg(), layout=Layout.NA)
    assert mod is Modality.TABULAR


def test_modality_hint_conflict_warns() -> None:
    df = pd.DataFrame({"y": [1.0], "ds": pd.to_datetime(["2026-01-01"])})
    mod, warnings = detect_modality(df, _cfg(time_col="ds", modality_hint="tabular"), layout=Layout.LONG)
    assert mod is Modality.TIMESERIES
    assert any("timeseries olarak" in w for w in warnings)


def test_task_regression() -> None:
    df = pd.DataFrame({"y": np.random.default_rng(0).normal(size=100), "x": range(100)})
    spec = infer_task(df, _cfg(), modality=Modality.TABULAR, target_profile=_target_profile(df))
    assert spec.task is Task.REGRESSION


def test_task_binary() -> None:
    df = pd.DataFrame({"y": [0, 1] * 50, "x": range(100)})
    spec = infer_task(df, _cfg(), modality=Modality.TABULAR, target_profile=_target_profile(df))
    assert spec.task is Task.BINARY_CLASSIFICATION


def test_task_multiclass_int_warns() -> None:
    df = pd.DataFrame({"y": ([0, 1, 2, 3, 4] * 20), "x": range(100)})
    spec = infer_task(df, _cfg(), modality=Modality.TABULAR, target_profile=_target_profile(df))
    assert spec.task is Task.MULTICLASS_CLASSIFICATION
    assert spec.inference_warnings


def test_task_forecasting_from_modality() -> None:
    df = pd.DataFrame(
        {"y": np.random.default_rng(0).normal(size=60), "ds": pd.date_range("2026-01-01", periods=60, freq="D")}
    )
    spec = infer_task(
        df, _cfg(time_col="ds"), modality=Modality.TIMESERIES, target_profile=_target_profile(df)
    )
    assert spec.task is Task.FORECASTING
    assert spec.time_col == "ds"


def test_task_hint_regression_overrides_timeseries() -> None:
    df = pd.DataFrame(
        {"y": np.random.default_rng(0).normal(size=60), "ds": pd.date_range("2026-01-01", periods=60, freq="D")}
    )
    cfg = _cfg(time_col="ds", task_hint="regression")
    spec = infer_task(df, cfg, modality=Modality.TIMESERIES, target_profile=_target_profile(df))
    assert spec.task is Task.REGRESSION
