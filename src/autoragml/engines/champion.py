"""Şampiyon refit — seçilen adayı **tüm training verisi** üzerinde yeniden fit et (ADR 0013/0014).

ES modelleri için `n_estimators` = validation fold'larındaki `best_iteration` medyanı
(ADR 0013 — full data'da ES yapamayız). `ModelBundle` üretir.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.scoreboard import SelectionResult
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import ValidationReport
from autoragml.engines.model_pipeline import FittedModelPipeline
from autoragml.exceptions import EngineError
from autoragml.models import build_estimator
from autoragml.postprocessors import build_postprocessor
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.validators import DefaultTuner, Tuner
from autoragml.validators.frame_ops import (
    column_roles,
    fit_estimator,
    reserved_columns,
    split_xy,
    target_transform_choice,
)


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(cols)).encode()).hexdigest()[:16]


def _median_best_iteration(report: ValidationReport | None) -> int | None:
    if report is None:
        return None
    iters = [fr.best_iteration for fr in report.folds if fr.best_iteration is not None]
    return int(np.median(iters)) if iters else None


def refit_champion(
    selection: SelectionResult,
    candidates: list[Candidate],
    reports: list[ValidationReport],
    frame: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    *,
    tuner: Tuner | None = None,
    pre_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> ModelBundle:
    """Şampiyonu tüm veride yeniden fit et → `ModelBundle`."""
    key = selection.champion.model_key
    candidate = next((c for c in candidates if c.key == key), None)
    if candidate is None:
        msg = f"Şampiyon aday bulunamadı: {key!r}"
        raise EngineError(msg)

    tuner = tuner or DefaultTuner()
    work = frame.reset_index(drop=True)
    target = task.targets[0]
    reserved = reserved_columns(task)

    ctx = PlanContext(
        target=target,
        task=task.task,
        column_roles=column_roles(profile),
        group_col=task.group_col,
        time_col=task.time_col,
        seed=config.seed,
    )
    outcome = tuner.tune(candidate, work, plan, task, ctx, config)

    pipe = FeaturePipeline.from_plan(plan, outcome.candidate_choices)
    fitted_pipe, frame_t = pipe.fit_transform(work, ctx)
    x, y = split_xy(frame_t, reserved, target)

    tt = TargetTransform(target_transform_choice(plan, outcome.candidate_choices)).fit(y)

    champ_report = next((r for r in reports if r.candidate_key == key), None)
    median_iter = _median_best_iteration(champ_report)
    params = dict(outcome.best_params)
    if median_iter and candidate.fidelity:
        params[candidate.fidelity] = median_iter

    estimator = build_estimator(candidate, task.task, params)
    best_iter: int | None
    if median_iter and candidate.fidelity:
        estimator.fit(x, tt.forward(y))  # sabit iterasyon → ES yok
        best_iter = median_iter
    else:
        best_iter = fit_estimator(estimator, candidate, x, tt.forward(y), config, task)

    postproc = build_postprocessor(config.postprocess, profile, task)
    fitted_post = None
    postprocess_summary: dict[str, object] = {}
    if postproc.is_active:
        oof = getattr(champ_report, "oof", None)
        yt = getattr(oof, "y_true", None)
        yp = getattr(oof, "y_pred", None)
        candidate_post = postproc.fit(yt, yp)
        if not candidate_post.is_noop:
            fitted_post = candidate_post
            postprocess_summary = candidate_post.summary

    model_pipeline = FittedModelPipeline(
        feature_pipeline=fitted_pipe,
        estimator=estimator,
        target_transform=tt,
        feature_cols=list(x.columns),
        reserved=reserved,
        pre_transform=pre_transform,
        postprocessor=fitted_post,
    )

    champ_row = next((r for r in selection.scoreboard.rows if r.model_key == key), None)
    metadata = BundleMetadata(
        feature_cols=list(x.columns),
        feature_set_hash=_feature_hash(list(x.columns)),
        target_col=target,
        model_key=key,
        scenario=selection.champion.scenario,
        best_iteration=best_iter,
        adaptive_plan_summary={
            "structure": plan.structure,
            "committed_ops": len(plan.committed_ops),
            "candidate_choices": outcome.candidate_choices,
        },
        params=params,
        postprocess_summary=postprocess_summary,
    )
    return ModelBundle(
        metadata=metadata,
        metrics_oof=dict(champ_row.all_metrics_mean) if champ_row else {},
        pipeline=model_pipeline,
    )
