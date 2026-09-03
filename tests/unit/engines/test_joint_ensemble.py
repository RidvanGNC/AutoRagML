"""engines.timeseries.joint_ensemble — klasik + reduction ortak forecasting ensemble (ADR 0035/P2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.engines.timeseries.classical import is_classical, run_classical_reports
from autoragml.engines.timeseries.joint_ensemble import (
    JOINT_ENSEMBLE_KEY,
    FittedJointForecaster,
    build_joint_forecast_ensemble,
)
from autoragml.engines.timeseries.reduction import build_reduction_features
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates
from autoragml.validators import run_validation_suite

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")


def _panel(n_series: int = 5, n_periods: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for g in range(n_series):
        base = 90 + g * 20
        for i, ds in enumerate(pd.date_range("2016-01-01", periods=n_periods, freq="MS")):
            trend = 0.4 * i
            rows.append({
                "unique_id": f"s{g}", "ds": ds,
                "y": base + trend + 20 * np.sin(i / 12 * 6.283) + rng.normal(0, 2.5),
            })
    return pd.DataFrame(rows)


def _prep():
    df = _panel()
    cfg = resolve_run_config(
        target="y",
        overrides={"hpo_level": "none", "time_col": "ds", "group_col": "unique_id",
                   "split_policy": {"horizon": 4}},
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    horizon, season = 4, 12
    aug, new_cols = build_reduction_features(frame, task, horizon=horizon, season=season)
    from autoragml.analyzers.profiling import build_column_profiles
    extra = build_column_profiles(aug[new_cols], target="y", thr=cfg.analyzers.thresholds, sampled=False)
    profile = profile.model_copy(update={"columns": [*profile.columns, *extra]})
    plan = build_plan(profile, task, cfg)
    return df, cfg, profile, task, frame, aug, plan


def test_joint_ensemble_builds_across_families() -> None:
    df, cfg, profile, task, frame, aug, plan = _prep()
    cands = resolve_candidates(cfg, task)
    classical = [c for c in cands if is_classical(c)]
    reduction = [c for c in cands if not is_classical(c) and c.family in {"gbdt", "forest", "linear"}]

    _, _, cv_grid = run_classical_reports(frame, profile, task, cfg, classical)
    assert cv_grid is not None

    joint = build_joint_forecast_ensemble(frame, profile, task, cfg, plan, cv_grid, reduction)
    if joint is None:
        pytest.skip("joint GES tek üyeye indi (küçük sentetik panel) — mekanizma yine de test_none'da")
    report, cand = joint
    assert cand.key == JOINT_ENSEMBLE_KEY and cand.family == "ensemble"
    kinds = set((cand.default_params["member_kinds"]).values())
    assert "reduction" in kinds  # en az bir reduction üye → ortak ızgara çalıştı
    assert report.oof is not None and report.oof.y_pred.shape == report.oof.y_true.shape
    assert np.isfinite(report.oof.y_pred).all()


def test_joint_builds_on_heterogeneous_panel() -> None:
    """Heterojen uzunluk/başlangıç — cutoff'lar seriye göre değişir; _win bazlı gruplama şart."""
    rng = np.random.default_rng(7)
    rows = []
    for g in range(18):
        length = int(rng.integers(48, 84))
        start = pd.Timestamp("2015-01-01") + pd.DateOffset(months=int(rng.integers(0, 20)))
        for i, ds in enumerate(pd.date_range(start, periods=length, freq="MS")):
            rows.append({"unique_id": f"s{g}", "ds": ds,
                         "y": 55 + g * 4 + 14 * np.sin(i / 12 * 6.28) + 0.3 * i + rng.normal(0, 2)})
    df = pd.DataFrame(rows)
    cfg = resolve_run_config(
        target="y", overrides={"hpo_level": "none", "time_col": "ds", "group_col": "unique_id",
                               "split_policy": {"horizon": 6}},
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    aug, new_cols = build_reduction_features(frame, task, horizon=6, season=12)
    from autoragml.analyzers.profiling import build_column_profiles
    extra = build_column_profiles(aug[new_cols], target="y", thr=cfg.analyzers.thresholds, sampled=False)
    profile = profile.model_copy(update={"columns": [*profile.columns, *extra]})
    plan = build_plan(profile, task, cfg)
    cands = resolve_candidates(cfg, task)
    classical = [c for c in cands if is_classical(c)]
    reduction = [c for c in cands if not is_classical(c) and c.family in {"gbdt", "forest", "linear"}]

    _, _, cv_grid = run_classical_reports(frame, profile, task, cfg, classical)
    assert cv_grid is not None
    joint = build_joint_forecast_ensemble(frame, profile, task, cfg, plan, cv_grid, reduction)
    assert joint is not None, "heterojen panelde joint kurulmalı (_win bazlı gruplama)"
    _report, cand = joint
    kinds = cand.default_params["member_kinds"]
    assert "reduction" in set(kinds.values()) and "classical" in set(kinds.values())


def test_joint_none_without_reduction() -> None:
    df, cfg, profile, task, frame, aug, plan = _prep()
    classical = [c for c in resolve_candidates(cfg, task) if is_classical(c)]
    _, _, cv_grid = run_classical_reports(frame, profile, task, cfg, classical)
    assert cv_grid is not None
    assert build_joint_forecast_ensemble(frame, profile, task, cfg, plan, cv_grid, []) is None


def test_joint_champion_refit_and_serving() -> None:
    df, cfg, profile, task, frame, aug, plan = _prep()
    cands = resolve_candidates(cfg, task)
    classical = [c for c in cands if is_classical(c)]
    reduction = [c for c in cands if not is_classical(c) and c.family in {"gbdt", "forest", "linear"}]

    cl_reports, cl_extra, cv_grid = run_classical_reports(frame, profile, task, cfg, classical)
    red_reports = run_validation_suite(reduction, aug, plan, profile, task, cfg)
    joint = build_joint_forecast_ensemble(frame, profile, task, cfg, plan, cv_grid, reduction)
    if joint is None:
        pytest.skip("joint tek üyeye indi")
    j_report, j_cand = joint

    from autoragml.engines.champion import _joint_bundle
    from autoragml.validators import DefaultTuner

    reports = [*cl_reports, *red_reports, j_report]
    candidates = [*classical, *cl_extra, *reduction, j_cand]
    from autoragml.scoring import score_reports
    selection = score_reports(reports, candidates, cfg, task, profile)

    import functools

    from autoragml.engines.timeseries.core_engine import _reduce_only
    pre = functools.partial(_reduce_only, task=task, horizon=4, season=12)
    bundle = _joint_bundle(
        j_cand, candidates, selection, reports, aug.reset_index(drop=True),
        plan, profile, task, cfg, DefaultTuner(), pre,
    )
    assert isinstance(bundle.pipeline, FittedJointForecaster)
    preds = bundle.pipeline.predict(frame)
    assert preds.shape == (len(frame),) and np.isfinite(preds).all()
