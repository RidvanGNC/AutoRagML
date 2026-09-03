"""Caruana greedy ensemble selection — saf numpy (ADR 0021).

Caruana et al. 2004 "Ensemble Selection from Libraries of Models" + 2006 (bagging).
Referans implementasyonlar: AutoGluon (Apache-2.0), auto-sklearn (BSD-3) — bu **temiz**
implementasyon makaleden; kod kopyalanmadı.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

_Arr = npt.NDArray[np.float64]
MetricFn = Callable[[_Arr, _Arr], float]


def _oriented(metric_fn: MetricFn, y_true: _Arr, y_pred: _Arr, *, lower_is_better: bool) -> float:
    """Minimize yönlü skor; non-finite → +inf."""
    value = metric_fn(y_true, y_pred)
    if not np.isfinite(value):
        return float("inf")
    return float(value) if lower_is_better else -float(value)


def greedy_selection(
    predictions: _Arr,
    y_true: _Arr,
    *,
    metric_fn: MetricFn,
    lower_is_better: bool,
    max_models: int,
    sorted_init_k: int = 1,
) -> _Arr:
    """OOF tahmin matrisi (n×m) → ağırlık vektörü (m,), toplamı 1.

    Deterministik: sorted-init + tie-break en düşük indekse gider.
    `use_best`: greedy boyunca görülen en iyi ensemble durumuna geri döner.
    """
    n, m = predictions.shape
    if m == 1:
        return np.array([1.0])

    single = np.array(
        [_oriented(metric_fn, y_true, predictions[:, j], lower_is_better=lower_is_better) for j in range(m)]
    )
    order = np.argsort(single, kind="stable")
    picks: list[int] = [int(j) for j in order[: max(0, sorted_init_k)]]
    ens_sum = predictions[:, picks].sum(axis=1) if picks else np.zeros(n)

    best_score = float("inf")
    best_len = len(picks)
    if picks:
        best_score = _oriented(
            metric_fn, y_true, ens_sum / len(picks), lower_is_better=lower_is_better
        )

    for _ in range(max_models):
        k = len(picks)
        candidate_mean = (ens_sum[:, None] + predictions) / (k + 1)
        scores = np.array(
            [
                _oriented(metric_fn, y_true, candidate_mean[:, j], lower_is_better=lower_is_better)
                for j in range(m)
            ]
        )
        scores = np.round(scores, 6)
        smin = scores.min()
        tied = np.flatnonzero(scores == smin)
        in_ens = [int(t) for t in tied if t in picks]
        chosen = in_ens[0] if in_ens else int(tied[0])  # tied sıralı → en düşük indeks

        picks.append(chosen)
        ens_sum = ens_sum + predictions[:, chosen]
        cur = _oriented(metric_fn, y_true, ens_sum / len(picks), lower_is_better=lower_is_better)
        if cur < best_score - 1e-12:
            best_score = cur
            best_len = len(picks)

    picks = picks[:best_len] if best_len > 0 else picks[:1]
    counts = np.bincount(picks, minlength=m).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def greedy_selection_proba(
    proba_stack: _Arr,
    y_true: _Arr,
    *,
    metric_fn: MetricFn,
    lower_is_better: bool,
    max_models: int,
    sorted_init_k: int = 1,
) -> _Arr:
    """Olasılık GES (ADR 0036) — `proba_stack` (m, n, C) → ağırlık (m,), toplamı 1.

    `greedy_selection` ile aynı Caruana döngüsü; ortalama **olasılık matrisi** üstünde.
    `metric_fn(y_true, mean_proba)` — mean_proba (n, C).
    """
    m = proba_stack.shape[0]
    if m == 1:
        return np.array([1.0])

    def score(mean_proba: _Arr) -> float:
        v = metric_fn(y_true, mean_proba)
        return float("inf") if not np.isfinite(v) else (float(v) if lower_is_better else -float(v))

    single = np.array([score(proba_stack[j]) for j in range(m)])
    order = np.argsort(single, kind="stable")
    picks: list[int] = [int(j) for j in order[: max(1, sorted_init_k)]]
    ens_sum = proba_stack[picks].sum(axis=0)
    best_score = score(ens_sum / len(picks))
    best_len = len(picks)

    for _ in range(max_models):
        k = len(picks)
        scores = np.round(
            np.array([score((ens_sum + proba_stack[j]) / (k + 1)) for j in range(m)]), 6
        )
        tied = np.flatnonzero(scores == scores.min())
        in_ens = [int(t) for t in tied if t in picks]
        chosen = in_ens[0] if in_ens else int(tied[0])
        picks.append(chosen)
        ens_sum = ens_sum + proba_stack[chosen]
        cur = score(ens_sum / len(picks))
        if cur < best_score - 1e-12:
            best_score, best_len = cur, len(picks)

    picks = picks[:best_len] if best_len > 0 else picks[:1]
    counts = np.bincount(picks, minlength=m).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def bagged_greedy_selection(
    predictions: _Arr,
    y_true: _Arr,
    *,
    metric_fn: MetricFn,
    lower_is_better: bool,
    max_models: int,
    sorted_init_k: int,
    n_bags: int,
    bag_fraction: float,
    seed: int,
) -> _Arr:
    """Model kütüphanesinin rastgele alt-kümelerinde tekrarlı GES → ortalama ağırlık (Caruana 2006)."""
    n, m = predictions.shape
    if m <= 2:
        return greedy_selection(
            predictions, y_true, metric_fn=metric_fn, lower_is_better=lower_is_better,
            max_models=max_models, sorted_init_k=sorted_init_k,
        )
    rng = np.random.default_rng(seed)
    bag_size = max(2, int(round(m * bag_fraction)))
    acc = np.zeros(m, dtype=np.float64)
    for _ in range(n_bags):
        idx = np.sort(rng.choice(m, size=min(bag_size, m), replace=False))
        w_bag = greedy_selection(
            predictions[:, idx], y_true, metric_fn=metric_fn, lower_is_better=lower_is_better,
            max_models=max_models, sorted_init_k=min(sorted_init_k, len(idx)),
        )
        acc[idx] += w_bag
    total = acc.sum()
    return acc / total if total > 0 else acc
