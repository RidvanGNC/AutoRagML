"""Çoklu karşılaştırma testleri — MCB + Diebold-Mariano (forecasting, opsiyonel; ADR 0014).

- **MCB** (Multiple Comparisons with the Best): fold başına model sıralaması → ortalama rank.
- **Diebold-Mariano**: şampiyon vs diğer her model için eşit tahmin doğruluğu testi
  (Harvey-Leybourne-Newbold küçük-örnek düzeltmesi, h=1).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from autoragml.contracts.scoreboard import ComparisonTests
from autoragml.contracts.validation import ValidationReport
from autoragml.scoring.metrics import lower_is_better

_MIN_FOLDS = 3


def _fold_metric(report: ValidationReport, fold_id: int, metric: str) -> float | None:
    for fr in report.folds:
        if fr.fold_id == fold_id:
            return fr.metrics.get(metric)
    return None


def mcb_ranks(reports: list[ValidationReport], metric: str) -> dict[str, float]:
    """Fold başına sıralama → model başına ortalama rank (küçük = iyi)."""
    lower = lower_is_better(metric)
    fold_ids = sorted({fr.fold_id for r in reports for fr in r.folds})
    ranks: dict[str, list[int]] = {r.candidate_key: [] for r in reports}
    for fid in fold_ids:
        vals = [
            (r.candidate_key, v)
            for r in reports
            if (v := _fold_metric(r, fid, metric)) is not None and np.isfinite(v)
        ]
        if len(vals) < 2:
            continue
        vals.sort(key=lambda t: t[1] if lower else -t[1])
        for position, (key, _) in enumerate(vals, start=1):
            ranks[key].append(position)
    return {k: float(np.mean(v)) for k, v in ranks.items() if v}


def diebold_mariano(
    champ: ValidationReport, other: ValidationReport, metric: str
) -> float | None:
    """p-değeri: H0 = eşit tahmin doğruluğu. `None` = yetersiz fold."""
    lower = lower_is_better(metric)
    diffs: list[float] = []
    for fr_c in champ.folds:
        lc = fr_c.metrics.get(metric)
        lo = _fold_metric(other, fr_c.fold_id, metric)
        if lc is None or lo is None:
            continue
        loss_c, loss_o = (lc, lo) if lower else (-lc, -lo)
        diffs.append(loss_o - loss_c)  # > 0 → şampiyon daha düşük kayıp

    n = len(diffs)
    if n < _MIN_FOLDS:
        return None
    d = np.asarray(diffs, dtype=float)
    s = float(d.std(ddof=1))
    if s == 0.0:
        return 1.0 if float(d.mean()) == 0.0 else 0.0
    dm = float(d.mean()) / (s / np.sqrt(n))
    hln = np.sqrt(max(1e-9, (n - 1) / n))  # h=1 düzeltmesi
    return float(2.0 * stats.t.sf(abs(dm * hln), df=n - 1))


def build_comparison_tests(
    reports: list[ValidationReport], champion_key: str, metric: str
) -> ComparisonTests | None:
    """Forecasting + yeterli fold varsa MCB + DM; değilse `None`."""
    if not any(len(r.folds) >= _MIN_FOLDS for r in reports):
        return None
    champ = next((r for r in reports if r.candidate_key == champion_key), None)
    if champ is None:
        return None
    dm = {
        r.candidate_key: p
        for r in reports
        if r.candidate_key != champion_key
        and (p := diebold_mariano(champ, r, metric)) is not None
    }
    return ComparisonTests(mcb_ranks=mcb_ranks(reports, metric), dm_pvalues=dm)
