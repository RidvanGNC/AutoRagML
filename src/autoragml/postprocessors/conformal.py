"""Split-conformal tahmin aralığı (ADR 0044).

Kalibrasyon seti = **mevcut şampiyon OOF'u** (`calibrate` ile aynı kaynak, ADR 0011 leakage-safe) —
ayrı bir split/ekstra fit gerekmez. Genişlik, finite-sample düzeltmeli mutlak-residual kantili
(standart split-conformal formülü, örn. MAPIE): `q = ceil((n+1)·coverage) / n` order statistic.

`coverage` fit-zamanında sabitlenmez — ham |residual| dizisi saklanır, `interval()` her çağrıda
istenen `coverage` için kantili hesaplar (varsayılan `config.postprocess.conformal.coverage`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_Arr = npt.NDArray[np.float64]


def _quantile_width(sorted_abs_resid: _Arr, coverage: float) -> float:
    """Finite-sample düzeltmeli split-conformal kantili — `ceil((n+1)·coverage)/n` order statistic."""
    n = sorted_abs_resid.size
    if n == 0:
        return 0.0
    q_level = min(float(np.ceil((n + 1) * coverage) / n), 1.0)
    return float(np.quantile(sorted_abs_resid, q_level, method="higher"))


@dataclass(frozen=True, slots=True)
class ConformalFit:
    """Fit edilmiş conformal genişlik kaynağı — saf, joblib-picklable (yalnız numpy dizileri)."""

    default_coverage: float
    global_residuals: _Arr  # sıralı |residual|, OOF
    group_residuals: dict[str, _Arr] | None  # grup → sıralı |residual| (yalnız min_group_oof≥ olanlar)

    def width_for(self, n: int, group: object | None, coverage: float | None) -> _Arr | float:
        """`n` satırlık genişlik: skaler (grup yok/yetersiz) veya satır-başı dizi."""
        cov = coverage if coverage is not None else self.default_coverage
        global_w = _quantile_width(self.global_residuals, cov)
        if group is None or not self.group_residuals:
            return global_w
        out = np.full(n, global_w, dtype=np.float64)
        g = np.asarray(group, dtype=object).astype(str)
        for gv, resid in self.group_residuals.items():
            mask = g == gv
            if mask.any():
                out[mask] = _quantile_width(resid, cov)
        return out


def fit_conformal(
    y_true: _Arr,
    y_pred: _Arr,
    *,
    coverage: float,
    group: npt.NDArray[np.object_] | None = None,
    per_group: bool = False,
    min_group_oof: int = 10,
) -> ConformalFit | None:
    """OOF residual'larından `ConformalFit`. Yetersiz örneklem (n<2) → `None` (aralık üretilmez)."""
    resid = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    finite = np.isfinite(resid)
    resid = resid[finite]
    if resid.size < 2:
        return None

    group_residuals: dict[str, _Arr] | None = None
    if per_group and group is not None:
        g = np.asarray(group, dtype=object)[finite]
        group_residuals = {}
        for gv in np.unique(g.astype(str)):
            gr = resid[g.astype(str) == gv]
            # ADR 0044: küçük grup örneklemi → güvenilmez kantil, o grup global'e düşer
            # (ConformalFit.width_for'da group_residuals'ta olmayan grup = global_w).
            if gr.size >= min_group_oof:
                group_residuals[gv] = np.sort(gr)
        if not group_residuals:
            group_residuals = None

    return ConformalFit(
        default_coverage=coverage,
        global_residuals=np.sort(resid),
        group_residuals=group_residuals,
    )
