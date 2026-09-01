"""postprocessors — ham tahmin düzeltme zinciri (ADR 0017).

`build_postprocessor(cfg, profile, task) -> Postprocessor` → `.fit(y_true, y_pred)` →
`FittedPostprocessor.apply(y_pred)`. `ModelBundle.pipeline` içine gömülür; serving'de çalışır.
Sıra: calibrate → clip → round → business_rule. Fit yalnız OOF üzerinden (ADR 0011).
"""

from __future__ import annotations

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import Task
from autoragml.contracts.postprocess_config import PostprocessConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.postprocessors.pipeline import FittedPostprocessor, Postprocessor

_REGRESSION_TASKS = {
    Task.REGRESSION,
    Task.FORECASTING,
    Task.QUANTILE_REGRESSION,
    Task.ORDINAL_REGRESSION,
}

__all__ = ["FittedPostprocessor", "Postprocessor", "build_postprocessor"]


def build_postprocessor(
    cfg: PostprocessConfig, profile: DataProfile, task: TaskSpec
) -> Postprocessor:
    """`PostprocessConfig` + hedef profili → fit edilmemiş `Postprocessor`."""
    target_min = profile.target_profile.stats.min
    is_regression = task.task in _REGRESSION_TASKS
    return Postprocessor(cfg, is_regression=is_regression, target_min=target_min)
