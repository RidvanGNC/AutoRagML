"""ensembling.greedy — Caruana greedy selection matematiği (ADR 0021)."""

from __future__ import annotations

import numpy as np

from autoragml.ensembling.greedy import bagged_greedy_selection, greedy_selection


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _kwargs():
    return {"metric_fn": _rmse, "lower_is_better": True, "max_models": 30, "sorted_init_k": 1}


def test_weights_sum_to_one_and_nonnegative() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    preds = np.column_stack([y + rng.normal(0, s, 200) for s in (0.2, 0.5, 1.0)])
    w = greedy_selection(preds, y, **_kwargs())
    assert w.shape == (3,)
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-9


def test_best_single_model_dominates() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    preds = np.column_stack([
        y + rng.normal(0, 0.05, 300),  # çok iyi
        y + rng.normal(0, 1.5, 300),   # kötü
        y + rng.normal(0, 1.5, 300),   # kötü
    ])
    w = greedy_selection(preds, y, **_kwargs())
    assert np.argmax(w) == 0
    assert w[0] > 0.6


def test_blending_two_complementary_models() -> None:
    n = 400
    rng = np.random.default_rng(2)
    y = rng.normal(size=n)
    e1 = rng.normal(0, 0.6, n)
    preds = np.column_stack([y + e1, y - e1])  # hataları zıt → ortalama çok daha iyi
    w = greedy_selection(preds, y, **_kwargs())
    blend = preds @ w
    assert _rmse(y, blend) < min(_rmse(y, preds[:, 0]), _rmse(y, preds[:, 1]))
    assert 0.3 < w[0] < 0.7  # dengeli


def test_use_best_ignores_late_garbage() -> None:
    n = 200
    rng = np.random.default_rng(3)
    y = rng.normal(size=n)
    preds = np.column_stack([y + rng.normal(0, 0.1, n), rng.normal(0, 5, n)])
    w = greedy_selection(preds, y, max_models=50, sorted_init_k=1, metric_fn=_rmse, lower_is_better=True)
    assert w[0] > 0.9  # çöp model neredeyse hiç seçilmez


def test_bagged_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=250)
    preds = np.column_stack([y + rng.normal(0, s, 250) for s in (0.2, 0.4, 0.6, 0.8, 1.0)])
    kw = {"metric_fn": _rmse, "lower_is_better": True, "max_models": 20, "sorted_init_k": 1,
          "n_bags": 15, "bag_fraction": 0.6, "seed": 7}
    w1 = bagged_greedy_selection(preds, y, **kw)
    w2 = bagged_greedy_selection(preds, y, **kw)
    np.testing.assert_allclose(w1, w2)
    assert abs(w1.sum() - 1.0) < 1e-9
