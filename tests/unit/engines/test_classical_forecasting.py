"""engines.timeseries.classical — native StatsForecast yolu (ADR 0023)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.engines.timeseries.classical import (
    is_classical,
    refit_classical,
    run_classical_reports,
)
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _panel(n_series: int = 6, n_periods: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_series):
        base = 80 + g * 25
        for i, ds in enumerate(pd.date_range("2016-01-01", periods=n_periods, freq="MS")):
            rows.append(
                {"unique_id": f"s{g}", "ds": ds, "y": base + 18 * np.sin(i / 12 * 6.283) + rng.normal(0, 3)}
            )
    return pd.DataFrame(rows)


def _prep(df: pd.DataFrame):
    cfg = resolve_run_config(
        target="y",
        overrides={
            "hpo_level": "none", "time_col": "ds", "group_col": "unique_id",
            "split_policy": {"horizon": 6},
        },
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    return cfg, ds, profile, task


def test_classical_reports_produced_for_seasonal_panel() -> None:
    df = _panel()
    cfg, ds, profile, task = _prep(df)
    classical = [c for c in resolve_candidates(cfg, task) if is_classical(c)]
    assert {c.key for c in classical} >= {"auto_ets", "auto_arima", "croston"}

    reports = run_classical_reports(materialize_frame(ds), profile, task, cfg, classical)
    by_key = {r.candidate_key: r for r in reports}
    assert "auto_ets" in by_key
    ets = by_key["auto_ets"]
    assert ets.oof_metrics["smape"] < 20  # mevsimsel seri → ETS iyi
    assert len(ets.folds) >= 2
    assert ets.oof is not None and ets.oof.y_true.shape == ets.oof.y_pred.shape


def test_classical_refit_predicts_future_horizon() -> None:
    df = _panel()
    cfg, ds, profile, task = _prep(df)
    ets = next(c for c in resolve_candidates(cfg, task) if c.key == "auto_ets")

    # son 6 dönemi holdout: train'de fit, tam frame'de predict, holdout hizası
    train = df.groupby("unique_id", group_keys=False).apply(lambda g: g.iloc[:-6])
    forecaster = refit_classical(ets, train, profile, task, cfg)

    preds = forecaster.predict(df.sort_values(["unique_id", "ds"]).reset_index(drop=True))
    full = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    hold = full.groupby("unique_id", sort=False).cumcount(ascending=False) < 6
    assert not np.isnan(preds[hold.to_numpy()]).any()
    # holdout tahminleri gerçeklere makul yakın (mevsimsel seri)
    err = np.abs(preds[hold.to_numpy()] - full.loc[hold, "y"].to_numpy())
    assert float(np.median(err)) < 25


def test_no_classical_candidates_returns_empty() -> None:
    df = _panel()
    cfg, ds, profile, task = _prep(df)
    assert run_classical_reports(materialize_frame(ds), profile, task, cfg, []) == []
