"""validators — split sınırını yöneten TEK yer (ADR 0010/6 + 0011).

- `resolve_splitter(frame, config, task, profile)` → split_policy (kısmi) + görev tabanlı
- `run_validation(candidate, ...)` → nested CV → `ValidationReport`
  (HPO + candidate_ops seçimi iç resample'da `Tuner` ile; dış fold yalnız skorlar)
- `run_validation_suite(candidates, ...)` → aynı split ile liste
- `leakage_checks` — overlap / preprocessing / multi_test → BLOCK
- `FeaturePipeline` / `TargetTransform` fold içinde fit; provenance == train
"""

from __future__ import annotations

from autoragml.validators.frame_ops import (
    column_roles,
    fit_estimator,
    inner_holdout_split,
    reserved_columns,
    split_xy,
    target_transform_choice,
)
from autoragml.validators.leakage_checks import check_fold_leakage, merge_leakage
from autoragml.validators.runner import (
    DefaultTuner,
    Tuner,
    TunerOutcome,
    run_validation,
    run_validation_suite,
)
from autoragml.validators.splitters import (
    Fold,
    SplitError,
    Splitter,
    resolve_splitter,
)

__all__ = [
    "DefaultTuner",
    "Fold",
    "SplitError",
    "Splitter",
    "Tuner",
    "TunerOutcome",
    "check_fold_leakage",
    "column_roles",
    "fit_estimator",
    "inner_holdout_split",
    "merge_leakage",
    "reserved_columns",
    "resolve_splitter",
    "run_validation",
    "run_validation_suite",
    "split_xy",
    "target_transform_choice",
]
