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
    refit_classical_ensemble,
    run_classical_reports,
)
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _panel(n_series: int = 4, n_periods: int = 54) -> pd.DataFrame:
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
            "split_policy": {"horizon": 4},
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

    reports, extra = run_classical_reports(materialize_frame(ds), profile, task, cfg, classical)
    by_key = {r.candidate_key: r for r in reports}
    assert "auto_ets" in by_key
    ets = by_key["auto_ets"]
    assert ets.oof_metrics["smape"] < 20  # mevsimsel seri → ETS iyi
    assert len(ets.folds) >= 2
    assert ets.oof is not None and ets.oof.y_true.shape == ets.oof.y_pred.shape
    # EAT ansamblı (ADR 0024) — ≥2 klasik model varsa
    assert "classical_ensemble" in by_key
    assert extra and extra[0].key == "classical_ensemble" and extra[0].ensemble_members


def test_classical_refit_predicts_future_horizon() -> None:
    df = _panel()
    cfg, ds, profile, task = _prep(df)
    ets = next(c for c in resolve_candidates(cfg, task) if c.key == "auto_ets")

    # son 4 dönemi holdout: train'de fit, tam frame'de predict, holdout hizası
    train = df.groupby("unique_id", group_keys=False).apply(lambda g: g.iloc[:-4])
    forecaster = refit_classical(ets, train, profile, task, cfg)

    full = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    preds = forecaster.predict(full)
    hold = (full.groupby("unique_id", sort=False).cumcount(ascending=False) < 4).to_numpy()
    assert not np.isnan(preds[hold]).any()
    err = np.abs(preds[hold] - full.loc[hold, "y"].to_numpy())
    assert float(np.median(err)) < 25


def test_classical_ensemble_refit() -> None:
    df = _panel()
    cfg, ds, profile, task = _prep(df)
    classical = [c for c in resolve_candidates(cfg, task) if is_classical(c)]
    _, extra = run_classical_reports(materialize_frame(ds), profile, task, cfg, classical)
    ens_cand = next(c for c in extra if c.key == "classical_ensemble")

    bundle_forecaster = refit_classical_ensemble(ens_cand, classical, df, profile, task, cfg)
    preds = bundle_forecaster.predict(df.sort_values(["unique_id", "ds"]).reset_index(drop=True))
    assert preds.shape == (len(df),)
    hold = (
        df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False).cumcount(ascending=False) < 4
    ).to_numpy()
    assert not np.isnan(preds[hold]).any()


def test_no_classical_candidates_returns_empty() -> None:
    df = _panel()
    cfg, ds, profile, task = _prep(df)
    assert run_classical_reports(materialize_frame(ds), profile, task, cfg, []) == ([], [])
