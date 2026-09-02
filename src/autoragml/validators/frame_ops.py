"""Fold-frame yardımcıları — `runner` ve `fine_tuners` tarafından paylaşılır.

Saf dönüşüm + tek bir fit yardımcısı (`fit_estimator`, fold-içi iç-val early stopping).
Split sınırı yönetimi burada değil — çağıran (runner/tuner) hangi frame'in train
olduğuna karar verir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import SemanticRole, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.logging import get_logger

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]


@dataclass
class OOFArrays:
    """Out-of-fold tahminler — `scoring` class-weighted skoru + guardrail için."""

    y_true: _Arr
    y_pred: _Arr
    group: npt.NDArray[np.object_] | None = None


def prediction_health(y_true: _Arr, y_pred: _Arr) -> dict[str, float]:
    """Tahmin sağlık istatistikleri (ADR 0014 guardrail girdisi)."""
    finite = np.isfinite(y_pred)
    pred_abs_max = float(np.max(np.abs(y_pred[finite]))) if finite.any() else 0.0
    true_abs_max = float(np.max(np.abs(y_true[np.isfinite(y_true)]))) if np.isfinite(y_true).any() else 0.0
    return {
        "n_non_finite": float((~finite).sum()),
        "n_negative": float((y_pred[finite] < 0).sum()),
        "pred_abs_max": pred_abs_max,
        "true_abs_max": true_abs_max,
        "pred_scale_ratio": pred_abs_max / true_abs_max if true_abs_max > 0 else 0.0,
    }


def reserved_columns(task: TaskSpec) -> set[str]:
    """Özellik olmayan kolonlar: hedef, zaman, grup."""
    return {c for c in (task.time_col, task.group_col, *task.targets) if c}


def column_roles(profile: DataProfile) -> dict[str, SemanticRole]:
    return {c.name: c.semantic_role for c in profile.columns}


def split_xy(frame: pd.DataFrame, reserved: set[str], target: str) -> tuple[pd.DataFrame, _Arr]:
    """Transform edilmiş frame'den X (yalnız sayısal) + y ayır."""
    y = np.asarray(pd.to_numeric(frame[target], errors="coerce"), dtype=np.float64)
    drop = [c for c in reserved if c in frame.columns]
    x = frame.drop(columns=drop)
    numeric = x.select_dtypes(include=["number", "bool"])
    if numeric.shape[1] < x.shape[1]:
        dropped = sorted(set(x.columns) - set(numeric.columns))
        logger.warning("sayısal olmayan kolonlar X'ten düşürüldü: %s", dropped)
    return numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0), y


def target_transform_choice(plan: AdaptivePlan, candidate_choices: dict[str, str]) -> str:
    """`candidate_ops`'taki `target` grubunun seçili (veya varsayılan) değeri."""
    for group in plan.candidate_ops:
        if group.group_name == "target":
            return candidate_choices.get("target", group.default)
    return "none"


def sdiff_ref_col(target: str) -> str:
    """seasonal_difference referans kolonu (reduction üretir — ADR 0026)."""
    return f"{target}_sdiff_ref"


def sdiff_ref(frame: pd.DataFrame, target: str, choice: str) -> _Arr | None:
    """`choice == "seasonal_difference"` ise `{target}_sdiff_ref` dizisi, değilse `None`."""
    col = sdiff_ref_col(target)
    if choice != "seasonal_difference" or col not in frame.columns:
        return None
    return np.asarray(pd.to_numeric(frame[col], errors="coerce"), dtype=np.float64)


def inner_holdout_split(
    n: int, frac: float, *, time_ordered: bool, seed: int = 0
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Fold-içi train/val ayrımı. TS'de son parça; değilse rastgele."""
    n_val = max(1, int(round(n * frac)))
    if time_ordered:
        return np.arange(n - n_val), np.arange(n - n_val, n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]


def fit_estimator(
    est: Any,
    candidate: Candidate,
    x: pd.DataFrame,
    y: _Arr,
    config: RunConfig,
    task: TaskSpec,
) -> int | None:
    """Fit + (destekleniyorsa) fold-içi iç-val early stopping. Döner: best_iteration."""
    module = type(est).__module__
    if not candidate.supports_early_stopping or len(x) < 30:
        est.fit(x, y)
        return None

    rounds = candidate.early_stopping_rounds or 50
    frac = config.validation.early_stopping_fraction
    is_ts = task.task is Task.FORECASTING
    tr_idx, val_idx = inner_holdout_split(len(x), frac, time_ordered=is_ts, seed=config.seed)
    x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    if module.startswith("lightgbm"):
        import lightgbm as lgb

        est.fit(
            x_tr,
            y_tr,
            eval_X=x_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(rounds, verbose=False)],
        )
        return int(getattr(est, "best_iteration_", 0)) or None
    if module.startswith("sklearn.ensemble") and hasattr(est, "set_params"):
        est.set_params(early_stopping=True, validation_fraction=frac, n_iter_no_change=rounds)
        est.fit(x, y)
        return int(getattr(est, "n_iter_", 0)) or None
    est.fit(x, y)
    return None
