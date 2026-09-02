"""2. dalga — panel forecasting koşum + değerlendirme (ADR 0015/0022).

Her seri için **son `horizon` dönem** harici holdout. `AutoRagML().fit(train)` →
`predict(tüm sıralı eval frame)` (reduction lag'leri train'den hesaplanır) → holdout
maskesiyle seç → **seasonal-naive** baseline ile sMAPE karşılaştırması.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from scripts.benchmarks.datasets import BenchmarkDataset
from scripts.benchmarks.evaluate import Outcome, evaluate

_MIN_TRAIN_MULT = 3  # seri en az season_length*3 + horizon train dönemi taşımalı


def _prep_panel(
    df: pd.DataFrame, ds: BenchmarkDataset
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(train_df, eval_df, holdout_mask) — eval_df kept-serilerin tüm geçmişi, sıralı."""
    tc, gc, h = ds.time_col, ds.group_col, ds.horizon
    assert tc is not None and gc is not None
    d = df.sort_values([gc, tc]).reset_index(drop=True)

    tail = d.groupby(gc, sort=False).cumcount(ascending=False)
    is_holdout_all = tail < h
    train_all = d[~is_holdout_all]
    counts = train_all.groupby(gc)[tc].count()
    keep = set(counts[counts >= ds.season_length * _MIN_TRAIN_MULT + h].index)

    eval_df = d[d[gc].isin(keep)].sort_values([gc, tc]).reset_index(drop=True)
    hold_mask = (eval_df.groupby(gc, sort=False).cumcount(ascending=False) < h).to_numpy()
    train_df = eval_df.loc[~hold_mask].reset_index(drop=True)
    return train_df, eval_df, hold_mask


def _seasonal_naive(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, hold_mask: np.ndarray, ds: BenchmarkDataset, target: str
) -> np.ndarray:
    """Her holdout satırı için: son `season_length` train gözleminin döngüsel tekrarı."""
    gc, s = ds.group_col, ds.season_length
    assert gc is not None
    last_by_series = {
        g: pd.to_numeric(grp[target], errors="coerce").to_numpy(dtype=np.float64)[-s:]
        for g, grp in train_df.groupby(gc, sort=False)
    }
    hold = eval_df.loc[hold_mask]
    pos = hold.groupby(gc, sort=False).cumcount().to_numpy()
    out = np.empty(len(hold), dtype=np.float64)
    for i, (g, p) in enumerate(zip(hold[gc].to_numpy(), pos, strict=True)):
        hist = last_by_series.get(g)
        out[i] = float(hist[p % len(hist)]) if hist is not None and len(hist) else 0.0
    return out


def run_forecasting(
    ds: BenchmarkDataset, hpo: str, out_dir: str, *, forecast_reduction: str = "direct"
) -> Outcome:
    """Panel forecasting: son-horizon holdout + seasonal-naive karşılaştırması."""
    from autoragml import AutoRagML

    t0 = time.perf_counter()
    df, target = ds.loader()
    train_df, eval_df, hold_mask = _prep_panel(df, ds)
    n_series = eval_df[ds.group_col].nunique()
    print(
        f"    {n_series} seri · {len(train_df)} train / {int(hold_mask.sum())} holdout satırı · "
        f"h={ds.horizon} · s={ds.season_length} · reduction={forecast_reduction}"
    )

    overrides: dict[str, object] = {"split_policy": {"horizon": ds.horizon}}
    if ds.primary_metric:
        overrides["primary_metric"] = ds.primary_metric
    if forecast_reduction != "direct":
        overrides["forecast_reduction"] = forecast_reduction
    model = AutoRagML(hpo_level=hpo, output_dir=out_dir, project_name=ds.name)
    result = model.fit(
        train_df, target=target, time_col=ds.time_col, group_col=ds.group_col, **overrides
    )

    preds = np.asarray(result.predict(eval_df), dtype=np.float64)
    y_pred = preds[hold_mask]
    y_true = pd.to_numeric(eval_df.loc[hold_mask, target], errors="coerce").to_numpy(dtype=np.float64)
    naive = _seasonal_naive(train_df, eval_df, hold_mask, ds, target)

    runtime = time.perf_counter() - t0
    return evaluate(ds.name, result, y_true, y_pred, naive, runtime_s=runtime, target_encoded=False)
