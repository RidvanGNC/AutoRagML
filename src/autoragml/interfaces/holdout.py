"""Nihai holdout — orchestrator carve + tek-seferlik skorlama (ADR 0020).

`validators` CV'sinden ayrı. Tabular: seed'li rastgele; TS: son `horizon` dönem
(global cutoff, `[group, time]` stable sıralı). `shift(horizon)` reduction özellikleri
bu genişlikte leakage-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.model_pipeline import FittedModelPipeline
from autoragml.logging import get_logger
from autoragml.scoring.metrics import compute_metrics

logger = get_logger(__name__)

_BoolArr = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class HoldoutSplit:
    """Nihai holdout ayrımı."""

    train: pd.DataFrame  # engine yalnız bunu görür
    scoring_frame: pd.DataFrame  # holdout skorlaması (TS: tüm sıralı frame; tabular: holdout df)
    holdout_mask: _BoolArr | None  # scoring_frame üzerinde bool (TS); tabular → None
    is_timeseries: bool
    n_holdout: int


def _min_rows_needed(config: RunConfig) -> int:
    frac = config.validation.holdout_fraction
    return math.ceil(config.validation.min_rows_for_cv / (1.0 - frac)) + 1


def split_holdout(frame: pd.DataFrame, config: RunConfig, task: TaskSpec) -> HoldoutSplit | None:
    """Nihai holdout'u carve et. Yeterli veri yoksa `None` (WARNING)."""
    n = len(frame)
    if n < _min_rows_needed(config):
        logger.warning(
            "[holdout] %d satır < %d — nihai holdout atlandı; skorlar OOF'ta kalır",
            n,
            _min_rows_needed(config),
        )
        return None

    is_ts = task.task.value == "forecasting" and bool(task.time_col) and task.time_col in frame.columns
    if is_ts:
        return _ts_holdout(frame, config, task)
    return _random_holdout(frame, config)


def _random_holdout(frame: pd.DataFrame, config: RunConfig) -> HoldoutSplit:
    n = len(frame)
    n_holdout = max(1, round(n * config.validation.holdout_fraction))
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n)
    hold_pos = np.sort(perm[:n_holdout])
    train_pos = np.sort(perm[n_holdout:])
    work = frame.reset_index(drop=True)
    return HoldoutSplit(
        train=work.iloc[train_pos].reset_index(drop=True),
        scoring_frame=work.iloc[hold_pos].reset_index(drop=True),
        holdout_mask=None,
        is_timeseries=False,
        n_holdout=int(n_holdout),
    )


def _ts_holdout(frame: pd.DataFrame, config: RunConfig, task: TaskSpec) -> HoldoutSplit:
    time_col = task.time_col
    assert time_col is not None
    group_col = task.group_col if task.group_col and task.group_col in frame.columns else None
    sort_cols = [c for c in (group_col, time_col) if c]

    work = frame.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    distinct = np.sort(pd.Series(work[time_col].dropna().unique()))
    horizon = task.horizon or (config.split_policy.horizon if config.split_policy else None) or 4
    k = min(int(horizon), max(1, len(distinct) - 1))
    cutoff = distinct[-k]
    mask = (work[time_col].to_numpy() >= cutoff)

    if mask.sum() == 0 or mask.sum() >= len(work):
        logger.warning("[holdout] TS holdout dejenere — atlandı")
        # fallback: rastgele
        return _random_holdout(frame, config)

    return HoldoutSplit(
        train=work.loc[~mask].reset_index(drop=True),
        scoring_frame=work,
        holdout_mask=np.asarray(mask, dtype=bool),
        is_timeseries=True,
        n_holdout=int(mask.sum()),
    )


def score_holdout(
    pipeline: FittedModelPipeline, split: HoldoutSplit, task: TaskSpec
) -> dict[str, float]:
    """Şampiyon pipeline'ını holdout'ta **bir kez** skorla."""
    target = task.targets[0]
    preds = np.asarray(pipeline.predict(split.scoring_frame), dtype=np.float64)
    y_true = pd.to_numeric(split.scoring_frame[target], errors="coerce").to_numpy(dtype=np.float64)
    if split.holdout_mask is not None:
        preds = preds[split.holdout_mask]
        y_true = y_true[split.holdout_mask]
    return compute_metrics(y_true, preds, task.task)
