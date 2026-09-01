"""scoring.metrics (ADR 0014)."""

from __future__ import annotations

import numpy as np

from autoragml.contracts.enums import Task
from autoragml.scoring.metrics import (
    bias,
    compute_metrics,
    csl,
    default_primary_metric,
    lower_is_better,
    rmse,
    smape,
    wmape,
)


def test_regression_metric_values() -> None:
    yt = np.array([10.0, 20.0, 30.0, 40.0])
    yp = np.array([12.0, 18.0, 33.0, 39.0])
    assert rmse(yt, yp) > 0
    assert abs(bias(yt, yp) - np.mean(yp - yt)) < 1e-9
    assert 0 <= smape(yt, yp) <= 200
    assert wmape(yt, yp) >= 0
    assert csl([1.0, 2.0], [3.0, 3.0]) == 100.0  # her tahmin talebi karşılıyor


def test_zero_safe() -> None:
    assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert wmape([0.0], [5.0]) == 0.0


def test_compute_metrics_by_task() -> None:
    reg = compute_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], Task.REGRESSION)
    assert {"smape", "rmse", "bias", "abs_bias", "csl"} <= set(reg)
    clf = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], Task.BINARY_CLASSIFICATION)
    assert {"accuracy", "f1_macro", "balanced_accuracy"} <= set(clf)


def test_primary_and_direction() -> None:
    assert default_primary_metric(Task.FORECASTING) == "smape"
    assert default_primary_metric(Task.REGRESSION) == "rmse"
    assert default_primary_metric(Task.BINARY_CLASSIFICATION) == "f1_macro"
    assert lower_is_better("rmse") and not lower_is_better("csl")
