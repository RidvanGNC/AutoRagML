"""Sahte ValidationReport / DataProfile üreticileri."""

from __future__ import annotations

import numpy as np

from autoragml.contracts.data_profile import ColumnProfile, ColumnStats, DataProfile, TargetSummary
from autoragml.contracts.enums import RawDtype, SemanticRole, SplitKind
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.validators.frame_ops import OOFArrays


def make_report(
    key: str,
    *,
    smape: float,
    se: float = 1.0,
    n_folds: int = 4,
    flags_metrics: dict[str, float] | None = None,
    leakage_fail: bool = False,
    n_negative: float = 0.0,
    frac_negative: float = 0.0,
    best_iteration: int | None = 200,
) -> ValidationReport:
    rng = np.random.default_rng(hash(key) % 2**32)
    fold_smapes = smape + rng.normal(0, se * np.sqrt(n_folds), n_folds)
    folds = [
        FoldReport(
            fold_id=i,
            n_train=100,
            n_test=25,
            metrics={"smape": float(v), "rmse": float(v / 2), "abs_bias": 1.0},
            best_iteration=best_iteration,
        )
        for i, v in enumerate(fold_smapes, start=1)
    ]
    metrics = {"smape": smape, "rmse": smape / 2, "abs_bias": 1.0, "wmape": smape * 0.9}
    metrics.update(flags_metrics or {})
    return ValidationReport(
        candidate_key=key,
        split_kind=SplitKind.ROLLING_ORIGIN,
        folds=folds,
        oof_metrics=metrics,
        oof_metric_se={"smape": se, "rmse": se / 2},
        prediction_health={"n_negative": n_negative, "frac_negative": frac_negative,
                           "n_non_finite": 0.0, "pred_abs_max": 120.0,
                           "true_abs_max": 100.0, "pred_scale_ratio": 1.2},
        leakage=LeakageReport(status="FAIL" if leakage_fail else "PASS"),
        realized_seconds=1.0,
        oof=OOFArrays(y_true=np.array([10.0, 20.0]), y_pred=np.array([11.0, 19.0])),
    )


def make_profile(target_min: float = 0.0) -> DataProfile:
    tp = ColumnProfile(
        name="y",
        raw_dtype=RawDtype.FLOAT,
        semantic_role=SemanticRole.TARGET,
        stats=ColumnStats(n_unique=50, missing_ratio=0.0, min=target_min, max=100.0),
    )
    return DataProfile(
        columns=[tp],
        n_rows=500,
        n_cols=1,
        target_profile=tp,
        target_summary=TargetSummary(),
    )
