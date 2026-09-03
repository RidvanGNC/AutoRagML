"""TimeSeriesCoreEngine — reduction + native classical çekirdek engine (ADR 0004 + 0015 + 0023).

Akış: reduction FE (leakage-safe, shift≥horizon) → ortak akış. Reduction adayları
tabular pipeline'dan, **klasik adaylar (statsforecast) native `StatsForecast` yolundan**
(`run_classical=True`) geçer; şampiyon her iki aileden olabilir.

`plan.structure == "per_group_champion"` + `plan.segments` → segment başına pooled akış
(ADR 0028); serving `FittedSegmentedPipeline` yönlendirir.
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
from autoragml.dynamics import build_plan
from autoragml.engines.core import run_core_pipeline
from autoragml.engines.segmented import run_segmented
from autoragml.engines.timeseries.classical import _resolve_freq, _season_length
from autoragml.engines.timeseries.reduction import build_reduction_features
from autoragml.io import materialize_frame
from autoragml.logging import get_logger
from autoragml.validators import Tuner

logger = get_logger(__name__)


def _reduce_only(frame: pd.DataFrame, task: TaskSpec, horizon: int, season: int) -> pd.DataFrame:
    return build_reduction_features(frame, task, horizon=horizon, season=season)[0]


def _subset_profile(profile: DataProfile, ids: set[str]) -> DataProfile:
    """Segment serilerine daraltılmış profil (ADR 0028) — per_series + intermittency_summary."""
    ts = profile.timeseries
    if ts is None:
        return profile
    per = [sp for sp in ts.per_series if sp.group in ids]
    summary: dict[str, int] = {}
    for sp in per:
        summary[sp.intermittency_class.value] = summary.get(sp.intermittency_class.value, 0) + 1
    return profile.model_copy(
        update={"timeseries": ts.model_copy(update={"per_series": per, "intermittency_summary": summary})}
    )


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
        horizon = int(
            task.horizon or (config.split_policy.horizon if config.split_policy else None) or 4
        )
        season = int(_season_length(profile, _resolve_freq(profile)))

        plan = build_plan(profile, task, config)
        if plan.segments:
            logger.info("[engine] segmented: %d segment", len(plan.segments))
            return run_segmented(
                self.key,
                lambda sf, sp: self._run_pooled(sf, config, sp, task, tuner, horizon, season),
                frame, profile, task, plan,
                subset_profile=_subset_profile,
            )
        return self._run_pooled(frame, config, profile, task, tuner, horizon, season)

    def _run_pooled(
        self,
        frame: pd.DataFrame,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
        tuner: Tuner | None,
        horizon: int,
        season: int,
    ) -> EngineResult:
        if config.forecast_reduction == "recursive":
            return self._run_recursive(frame, config, profile, task, tuner, horizon, season)

        augmented, new_cols = build_reduction_features(
            frame, task, horizon=horizon, season=season
        )
        messages: list[str] = []
        if new_cols:
            extra = build_column_profiles(
                augmented[new_cols],
                target=task.targets[0],
                thr=config.analyzers.thresholds,
                sampled=False,
            )
            profile = profile.model_copy(update={"columns": [*profile.columns, *extra]})
            messages.append(f"reduction: {len(new_cols)} hedef-türevi özellik (shift≥{horizon}).")

        pre_transform = (
            functools.partial(_reduce_only, task=task, horizon=horizon, season=season)
            if new_cols
            else None
        )
        return run_core_pipeline(
            self.key, augmented, profile, task, config,
            tuner=tuner, messages=messages, pre_transform=pre_transform,
            run_classical=config.classical_forecasting,
            run_neural_ts=config.neural_enabled != "off",  # ADR 0032 — kapı neural_gate/reports'ta
        )

    def _run_recursive(
        self,
        frame: pd.DataFrame,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
        tuner: Tuner | None,
        horizon: int,
        season: int,
    ) -> EngineResult:
        """Recursive multi-step yolu (ADR 0026 B). Özellikler `run_recursive_reports` içinde
        fold-başına kurulur; burada yalnız profili zenginleştirip planın impute/scale
        adımlarının lag kolonlarını kapsamasını sağlarız."""
        aug, new_cols = build_reduction_features(
            frame, task, horizon=1, season=season, strategy="recursive"
        )
        messages = [f"forecast_reduction=recursive (h={horizon}, s={season})"]
        if new_cols:
            extra = build_column_profiles(
                aug[new_cols], target=task.targets[0], thr=config.analyzers.thresholds, sampled=False
            )
            profile = profile.model_copy(update={"columns": [*profile.columns, *extra]})
            messages.append(f"recursive reduction: {len(new_cols)} özellik (shift≥1).")
        return run_core_pipeline(
            self.key, frame, profile, task, config,
            tuner=tuner, messages=messages, pre_transform=None,
            run_classical=config.classical_forecasting,
            recursive=True, recursive_season=season,
        )
