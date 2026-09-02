"""TimeSeriesCoreEngine — reduction + native classical çekirdek engine (ADR 0004 + 0015 + 0023).

Akış: reduction FE (leakage-safe, shift≥horizon) → ortak akış. Reduction adayları
tabular pipeline'dan, **klasik adaylar (statsforecast) native `StatsForecast` yolundan**
(`run_classical=True`) geçer; şampiyon her iki aileden olabilir.
"""

from __future__ import annotations

import functools

import pandas as pd

from autoragml.analyzers.profiling import build_column_profiles
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.core import run_core_pipeline
from autoragml.engines.timeseries.reduction import build_reduction_features
from autoragml.io import materialize_frame
from autoragml.logging import get_logger
from autoragml.validators import Tuner

logger = get_logger(__name__)


def _reduce_only(frame: pd.DataFrame, task: TaskSpec, horizon: int) -> pd.DataFrame:
    return build_reduction_features(frame, task, horizon=horizon)[0]


class TimeSeriesCoreEngine:
    """Reduction-frame + tabular model havuzu + rolling-origin CV."""

    key = "timeseries_core"

    def run(
        self,
        dataset: Dataset,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
        *,
        tuner: Tuner | None = None,
    ) -> EngineResult:
        frame = materialize_frame(dataset)
        horizon = task.horizon or (config.split_policy.horizon if config.split_policy else None) or 4

        augmented, new_cols = build_reduction_features(frame, task, horizon=int(horizon))
        messages: list[str] = []
        if new_cols:
            extra = build_column_profiles(
                augmented[new_cols],
                target=task.targets[0],
                thr=config.analyzers.thresholds,
                sampled=False,
            )
            profile = profile.model_copy(update={"columns": [*profile.columns, *extra]})
            messages.append(f"reduction: {len(new_cols)} hedef-türevi özellik (shift≥{int(horizon)}).")

        pre_transform = (
            functools.partial(_reduce_only, task=task, horizon=int(horizon)) if new_cols else None
        )
        return run_core_pipeline(
            self.key, augmented, profile, task, config,
            tuner=tuner, messages=messages, pre_transform=pre_transform,
            run_classical=config.classical_forecasting,
        )
