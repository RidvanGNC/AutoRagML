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
from autoragml.contracts.model_bundle import ModelBundle
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.dynamics import build_plan
from autoragml.engines.core import run_core_pipeline
from autoragml.engines.segmented import run_segmented
from autoragml.engines.timeseries.classical import _resolve_freq, _season_length
from autoragml.engines.timeseries.hierarchical import (
    FittedHierarchicalForecaster,
    build_hierarchy,
    hierarchical_available,
)
from autoragml.engines.timeseries.reduction import build_reduction_features
from autoragml.exceptions import EngineError
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

        if config.hierarchy_cols:  # ADR 0045 — segmentasyon/recursive ile birleşmez, hep pooled
            return self._run_hierarchical(frame, config, profile, task, tuner, horizon, season)

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
            raw_frame=frame,  # ADR 0035/P2 — joint_ensemble ham frame'e cutoff başına refit yapar
            run_classical=config.classical_forecasting,
            run_neural_ts=config.neural_enabled != "off",  # ADR 0032 — kapı neural_gate/reports'ta
            run_foundation_ts=config.foundation_enabled != "off",  # ADR 0033 — kapı foundation_gate'te
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
            run_neural_ts=config.neural_enabled != "off",  # ADR 0032
            run_foundation_ts=config.foundation_enabled != "off",  # ADR 0033
            recursive=True, recursive_season=season,
        )

    def _run_hierarchical(
        self,
        frame: pd.DataFrame,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
        tuner: Tuner | None,
        horizon: int,
        season: int,
    ) -> EngineResult:
        """Hiyerarşik reconciliation (ADR 0045): bottom paneli agrega düğümlerle genişlet →
        normal pooled akış (segmentasyon/recursive ile birleşmez, v1) → şampiyonu
        `FittedHierarchicalForecaster` ile sar (MinTrace/wls_struct, serving-zamanında)."""
        if not hierarchical_available():
            msg = "hierarchy_cols verildi ama `hierarchicalforecast` kurulu değil ([hierarchical] extra)"
            raise EngineError(msg)
        assert config.hierarchy_cols and task.group_col  # RunConfig._post_checks garanti eder
        hspec = build_hierarchy(
            frame, hierarchy_cols=config.hierarchy_cols, group_col=task.group_col,
            time_col=task.time_col or "ds", target_col=task.targets[0],
        )
        agg_frame = hspec.agg_frame
        # `aggregate()` yalnız [group_col, time_col, target_col] tutar (diğer ham kolonlar düşer) —
        # orijinal (agregasyon-öncesi) profil onları hâlâ feature sanıp committed_ops üretebilir;
        # FeaturePipeline bu genişletilmiş panelde onları bulamaz → aday çöker. Profili budayarak
        # yalnız gerçekten var olan 3 kolona daralt (ADR 0045).
        keep = {task.group_col, task.time_col, task.targets[0]}
        pruned_profile = profile.model_copy(
            update={"columns": [c for c in profile.columns if c.name in keep]}
        )
        messages = [
            f"hierarchical: {len(hspec.node_order)} düğüm ({len(hspec.bottom_ids)} bottom + "
            f"{len(hspec.node_order) - len(hspec.bottom_ids)} agrega), MinTrace(wls_struct)"
        ]
        result = self._run_pooled(agg_frame, config, pruned_profile, task, tuner, horizon, season)
        result = result.model_copy(update={"messages": [*messages, *result.messages]})

        rec_method = config.hierarchy_reconcile_method  # ADR 0047

        def _wrap(bundle: ModelBundle) -> ModelBundle:
            wrapped = FittedHierarchicalForecaster(
                inner=bundle.pipeline, hspec=hspec, reconcile_method=rec_method
            )
            return bundle.model_copy(update={"pipeline": wrapped})

        inner_finalize = result.finalize
        wrapped_champion = _wrap(result.champion)

        def _finalize_wrapped(full_frame: pd.DataFrame) -> ModelBundle:
            if inner_finalize is None:  # pragma: no cover - champion_refit_full=False ise
                return wrapped_champion
            full_hspec = build_hierarchy(
                full_frame, hierarchy_cols=config.hierarchy_cols or [], group_col=hspec.group_col,
                time_col=hspec.time_col, target_col=hspec.target_col,
            )
            refit: ModelBundle = inner_finalize(full_hspec.agg_frame)
            return refit.model_copy(
                update={
                    "pipeline": FittedHierarchicalForecaster(
                        inner=refit.pipeline, hspec=full_hspec, reconcile_method=rec_method
                    )
                }
            )

        return result.model_copy(update={"champion": wrapped_champion, "finalize": _finalize_wrapped})
