"""models.estimator — class_path çözümü + estimator kurulumu (ADR 0012/0013)."""

from __future__ import annotations

import numpy as np
import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import Task
from autoragml.models import build_candidates, build_estimator, resolve_class_path
from autoragml.models.estimator import EstimatorBuildError


def _cands():
    return {c.key: c for c in build_candidates(resolve_run_config(target="y").config)}


def test_resolve_class_path_str_and_dict() -> None:
    assert resolve_class_path("a.b.C", Task.REGRESSION) == "a.b.C"
    cp = {"regression": "x.Reg", "classification": "x.Clf"}
    assert resolve_class_path(cp, Task.REGRESSION) == "x.Reg"
    assert resolve_class_path(cp, Task.FORECASTING) == "x.Reg"  # reduction
    assert resolve_class_path(cp, Task.BINARY_CLASSIFICATION) == "x.Clf"


def test_resolve_class_path_missing_family() -> None:
    with pytest.raises(EstimatorBuildError, match="ailesi"):
        resolve_class_path({"regression": "x.Reg"}, Task.BINARY_CLASSIFICATION)


def test_build_estimator_merges_params() -> None:
    est = build_estimator(_cands()["random_forest"], Task.REGRESSION, {"n_estimators": 42})
    assert est.n_estimators == 42
    assert est.random_state == 42  # default korunur


def test_build_estimator_wrap_pipeline() -> None:
    est = build_estimator(_cands()["ridge"], Task.REGRESSION)
    from sklearn.pipeline import Pipeline

    assert isinstance(est, Pipeline)
    assert est.steps[-1][0] == "model"


def test_build_estimator_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 120)
    est = build_estimator(_cands()["lightgbm"], Task.REGRESSION, {"n_estimators": 20})
    est.fit(x, y)
    assert est.predict(x[:5]).shape == (5,)


def test_build_estimator_invalid_param_raises() -> None:
    with pytest.raises(EstimatorBuildError, match="geçersiz"):
        build_estimator(_cands()["ridge"], Task.REGRESSION, {"not_a_real_param": 1})
