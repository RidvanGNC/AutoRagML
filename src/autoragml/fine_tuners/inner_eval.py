"""İç resample kurulumu + tek konfigürasyon değerlendirmesi (ADR 0013).

`RandomSearchTuner` ve `OptunaTuner` bu modülü paylaşır. Dış test'e **hiç dokunmaz** —
yalnız dış-fold train frame'i (`frame`) üzerinde çalışır (ADR 0010/6, ADR 0014/1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import Task
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.models import build_estimator
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.scoring.metrics import compute_metrics, default_primary_metric, lower_is_better
from autoragml.validators.frame_ops import (
    inner_holdout_split,
    reserved_columns,
    split_xy,
    target_transform_choice,
)
from autoragml.validators.splitters import KFoldSplitter, RollingOriginSplitter

IdxPair = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]


def resolve_primary_metric(config: RunConfig, task: TaskSpec) -> str:
    return config.primary_metric or default_primary_metric(task.task)


def build_inner_splits(
    frame: pd.DataFrame, task: TaskSpec, config: RunConfig, n_folds: int
) -> list[IdxPair]:
    """Dış-fold train'i içinde HPO değerlendirme split'leri (holdout veya küçük CV)."""
    is_ts = task.task is Task.FORECASTING and task.time_col is not None
    if n_folds <= 1:
        tr, va = inner_holdout_split(
            len(frame), config.validation.holdout_fraction, time_ordered=is_ts, seed=config.seed
        )
        return [(tr, va)]
    if is_ts:
        assert task.time_col is not None
        horizon = task.horizon or 4
        n_periods = frame[task.time_col].nunique()
        min_train = max(horizon, n_periods // (n_folds + 1))
        try:
            splitter = RollingOriginSplitter(
                time_col=task.time_col,
                group_col=task.group_col,
                horizon_periods=horizon,
                step_periods=horizon,
                n_folds=n_folds,
                min_train_periods=min_train,
            )
            folds = splitter.split(frame)
            return [(f.train_idx, f.test_idx) for f in folds]
        except Exception:  # noqa: BLE001 - yetersiz veri -> tek holdout'a düş
            tr, va = inner_holdout_split(len(frame), config.validation.holdout_fraction, time_ordered=True)
            return [(tr, va)]
    stratify = task.targets[0] if task.task.value.endswith("classification") else None
    try:
        folds = KFoldSplitter(n_splits=n_folds, seed=config.seed, stratify_col=stratify).split(frame)
        return [(f.train_idx, f.test_idx) for f in folds]
    except Exception:  # noqa: BLE001
        tr, va = inner_holdout_split(
            len(frame), config.validation.holdout_fraction, time_ordered=False, seed=config.seed
        )
        return [(tr, va)]


def evaluate_trial(
    candidate: Candidate,
    frame: pd.DataFrame,
    plan: AdaptivePlan,
    task: TaskSpec,
    ctx: PlanContext,
    config: RunConfig,
    params: dict[str, Any],
    candidate_choices: dict[str, str],
    inner_splits: list[IdxPair],
    *,
    fidelity_value: float | None = None,
) -> float:
    """Bir (params, candidate_choices) konfigürasyonunu iç split'lerde skorla.

    Döner: **küçük daha iyi** yönde bir değer (higher-is-better metrikler negatiflenir).
    """
    metric = resolve_primary_metric(config, task)
    reserved = reserved_columns(task)
    target = task.targets[0]
    full_params = dict(params)
    if fidelity_value is not None and candidate.fidelity:
        full_params[candidate.fidelity] = int(round(fidelity_value))

    scores: list[float] = []
    for tr_idx, va_idx in inner_splits:
        train = frame.iloc[tr_idx].reset_index(drop=True)
        val = frame.iloc[va_idx].reset_index(drop=True)

        pipe = FeaturePipeline.from_plan(plan, candidate_choices)
        fitted, train_t = pipe.fit_transform(train, ctx)
        val_t = fitted.apply(val)

        x_tr, y_tr = split_xy(train_t, reserved, target)
        x_va, y_va = split_xy(val_t, reserved, target)
        x_va = x_va.reindex(columns=x_tr.columns, fill_value=0.0)

        tt = TargetTransform(target_transform_choice(plan, candidate_choices)).fit(y_tr)
        est = build_estimator(candidate, task.task, full_params)
        est.fit(x_tr, tt.forward(y_tr))
        pred = tt.inverse(np.asarray(est.predict(x_va), dtype=np.float64))
        m = compute_metrics(y_va, pred, task.task)
        scores.append(m.get(metric, next(iter(m.values()))))

    mean_score = float(np.mean(scores))
    return mean_score if lower_is_better(metric) else -mean_score
