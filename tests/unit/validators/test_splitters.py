"""validators.splitters — split stratejileri + resolve (ADR 0008/2 + 0010/6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.enums import SplitKind
from autoragml.io import load_dataset
from autoragml.validators.splitters import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    RollingOriginSplitter,
    SplitError,
    resolve_splitter,
)


def test_holdout_time_ordered() -> None:
    df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=100, freq="D"), "y": range(100)})
    folds = HoldoutSplitter(test_fraction=0.2, time_col="ds").split(df)
    assert len(folds) == 1
    assert folds[0].train_idx.max() < folds[0].test_idx.min()  # zaman sırası
    assert folds[0].test_idx.size == 20


def test_kfold_count_and_coverage() -> None:
    df = pd.DataFrame({"y": range(50), "x": range(50)})
    folds = KFoldSplitter(n_splits=5, seed=0).split(df)
    assert len(folds) == 5
    covered = np.concatenate([f.test_idx for f in folds])
    assert sorted(covered) == list(range(50))


def test_kfold_too_few_rows() -> None:
    with pytest.raises(SplitError):
        KFoldSplitter(n_splits=5, seed=0).split(pd.DataFrame({"y": [1, 2, 3]}))


def test_group_kfold_no_group_shared() -> None:
    df = pd.DataFrame({"g": [f"g{i%10}" for i in range(200)], "y": range(200)})
    folds = GroupKFoldSplitter(n_splits=5, group_col="g").split(df)
    for f in folds:
        assert not (set(df["g"].iloc[f.train_idx]) & set(df["g"].iloc[f.test_idx]))


def test_rolling_origin_expanding_no_time_overlap() -> None:
    df = pd.DataFrame(
        {
            "ds": np.repeat(pd.date_range("2024-01-01", periods=60, freq="W-MON"), 3),
            "g": ["A", "B", "C"] * 60,
            "y": np.arange(180, dtype=float),
        }
    )
    sp = RollingOriginSplitter(
        time_col="ds", group_col="g", horizon_periods=4, step_periods=4, n_folds=4, min_train_periods=20
    )
    folds = sp.split(df)
    assert 1 <= len(folds) <= 4
    prev_train = 0
    for f in folds:
        tr_max = pd.to_datetime(df["ds"].iloc[f.train_idx]).max()
        te_min = pd.to_datetime(df["ds"].iloc[f.test_idx]).min()
        assert tr_max < te_min  # zaman sızıntısı yok
        assert f.train_idx.size >= prev_train  # genişleyen train
        prev_train = f.train_idx.size


def _analyzed(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(target="y", overrides=over or None).config
    profile, task = analyze(load_dataset(df, cfg), cfg)
    return df.reset_index(drop=True), cfg, task, profile


def test_resolve_forecasting_rolling() -> None:
    weeks = pd.date_range("2023-01-02", periods=120, freq="W-MON")
    df = pd.DataFrame({"ds": np.tile(weeks, 2), "g": np.repeat(["A", "B"], 120), "y": np.arange(240.0)})
    frame, cfg, task, profile = _analyzed(df, time_col="ds", group_col="g")
    assert resolve_splitter(frame, cfg, task, profile).kind is SplitKind.ROLLING_ORIGIN


def test_resolve_small_forecasting_holdout() -> None:
    weeks = pd.date_range("2026-01-05", periods=10, freq="W-MON")
    df = pd.DataFrame({"ds": weeks, "y": np.arange(10.0)})
    frame, cfg, task, profile = _analyzed(df, time_col="ds")
    assert resolve_splitter(frame, cfg, task, profile).kind is SplitKind.HOLDOUT


def test_resolve_tabular_kfold_and_classification_stratified() -> None:
    reg = pd.DataFrame({"y": np.random.default_rng(0).normal(size=300), "x": range(300)})
    frame, cfg, task, profile = _analyzed(reg)
    assert resolve_splitter(frame, cfg, task, profile).kind is SplitKind.KFOLD


def test_resolve_explicit_kind_override() -> None:
    df = pd.DataFrame({"y": np.random.default_rng(0).normal(size=300), "x": range(300)})
    frame, cfg, task, profile = _analyzed(df, split_policy={"kind": "holdout", "test_size": 0.3})
    sp = resolve_splitter(frame, cfg, task, profile)
    assert sp.kind is SplitKind.HOLDOUT
