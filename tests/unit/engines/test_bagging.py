"""engines — k-fold bagged şampiyon refit (ADR 0022)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.engines.champion import refit_champion
from autoragml.engines.ensemble_pipeline import FittedEnsemblePipeline
from autoragml.engines.model_pipeline import FittedModelPipeline
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates
from autoragml.scoring import score_reports
from autoragml.validators import run_validation_suite


def _reg_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 4))
    y = x @ np.array([1.3, -1.7, 0.6, 0.2]) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(4)}})


def _cls_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 4))
    logit = x @ np.array([1.5, -1.0, 0.5, 0.0])
    y = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(4)}})


def _champion(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(target="y", overrides={"hpo_level": "none", **over}).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    plan = build_plan(profile, task, cfg)
    candidates = resolve_candidates(cfg, task)
    reports = run_validation_suite(candidates, frame, plan, profile, task, cfg)
    selection = score_reports(reports, candidates, cfg, task, profile)
    bundle = refit_champion(selection, candidates, reports, frame, plan, profile, task, cfg)
    return bundle, df


def test_regression_champion_is_bagged() -> None:
    bundle, df = _champion(_reg_df())
    assert isinstance(bundle.pipeline, FittedEnsemblePipeline)
    assert bundle.metadata.ensemble["bagged"] is True
    assert bundle.metadata.ensemble["folds"] == 5
    assert len(bundle.pipeline.members) == 5
    pred = bundle.pipeline.predict(_reg_df(30))
    assert pred.shape == (30,)
    assert not np.isnan(pred).any()


def test_bagging_disabled_gives_single_model() -> None:
    bundle, _ = _champion(_reg_df(), bagging={"enabled": False})
    assert isinstance(bundle.pipeline, FittedModelPipeline)
    assert bundle.metadata.ensemble == {}


def test_bagging_folds_configurable() -> None:
    bundle, _ = _champion(_reg_df(), bagging={"folds": 3})
    assert bundle.metadata.ensemble["folds"] == 3
    assert len(bundle.pipeline.members) == 3  # type: ignore[union-attr]


def test_classification_not_bagged_predictions_discrete() -> None:
    bundle, df = _champion(_cls_df(), task_hint="binary_classification")
    assert isinstance(bundle.pipeline, FittedModelPipeline)  # v1: sınıflandırma bag'lenmez
    pred = bundle.pipeline.predict(_cls_df(50))
    assert set(np.unique(pred)).issubset({0.0, 1.0})


def test_small_data_falls_back_to_single() -> None:
    bundle, _ = _champion(_reg_df(60))  # < min_rows_for_cv/(...) → holdout splitter → 1 fold
    assert isinstance(bundle.pipeline, FittedModelPipeline)
