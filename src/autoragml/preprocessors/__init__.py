"""preprocessors — leakage-safe by construction (ADR 0011).

- `FeaturePipeline.from_plan(plan, candidate_choices)` → fit edilebilir dönüşüm zinciri
- `fit`/`fit_transform` yalnız train frame'inde (fold içinde, `validators` çağırır)
- `TargetTransform` — hedef `y` üzerinde forward/inverse (engine, estimator etrafında)
- Katalog transformları: drop · date_expand · impute · encode (onehot/ordinal/target_encode/hashing)
  · scale · log1p · yeo_johnson · quantile
- `target_encode` → sklearn iç cross-fitting (train'e cross-fit, test'e full-train)
"""

from __future__ import annotations

from autoragml.preprocessors.catalog import (
    PreprocessError,
    build_encode,
    build_numeric_transform,
    build_op,
)
from autoragml.preprocessors.pipeline import FeaturePipeline, FittedFeaturePipeline
from autoragml.preprocessors.target import FittedTargetTransform, TargetTransform

__all__ = [
    "FeaturePipeline",
    "FittedFeaturePipeline",
    "FittedTargetTransform",
    "PreprocessError",
    "TargetTransform",
    "build_encode",
    "build_numeric_transform",
    "build_op",
]
