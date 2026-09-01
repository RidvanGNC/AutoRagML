"""ValidationConfig — `validators` ayarları (ADR 0010/6 + 0013).

`RunConfig.validation` altında taşınır.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract


class ValidationConfig(Contract):
    """Split + nested CV + early stopping ayarları."""

    min_rows_for_cv: int = Field(default=100, ge=10)
    default_kfold_splits: int = Field(default=5, ge=2)
    default_rolling_folds: int = Field(default=4, ge=2)
    holdout_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)

    # Early stopping (fold-içi iç-val, ADR 0013)
    early_stopping_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)

    # Rolling-origin varsayılanları (split_policy verilmezse)
    default_rolling_step: int | None = None  # None → horizon
    default_min_train_periods: int = Field(default=0, ge=0)  # 0 → n_periods // (folds+1)
