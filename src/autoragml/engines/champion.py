"""Şampiyon refit — seçilen adayı **tüm training verisi** üzerinde yeniden fit et (ADR 0013/0014/0021).

ES modelleri için `n_estimators` = validation fold'larındaki `best_iteration` medyanı
(ADR 0013 — full data'da ES yapamayız). Şampiyon `weighted_ensemble` ise her üye
postprocess'siz refit edilir + `FittedEnsemblePipeline`. `ModelBundle` üretir.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

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
from autoragml.engines.ensemble_pipeline import FittedEnsemblePipeline
from autoragml.engines.model_pipeline import FittedModelPipeline
from autoragml.ensembling import ENSEMBLE_KEY
from autoragml.exceptions import EngineError
from autoragml.models import build_estimator
from autoragml.postprocessors import FittedPostprocessor, build_postprocessor
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.validators import DefaultTuner, Tuner
from autoragml.validators.frame_ops import (
    column_roles,
    fit_estimator,
    reserved_columns,
    split_xy,
    target_transform_choice,
)

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(cols)).encode()).hexdigest()[:16]


def _median_best_iteration(report: ValidationReport | None) -> int | None:
    if report is None:
        return None
    iters = [fr.best_iteration for fr in report.folds if fr.best_iteration is not None]
    return int(np.median(iters)) if iters else None


@dataclass
class _FitResult:
    pipeline: FittedModelPipeline
    params: dict[str, object]
    best_iter: int | None
    candidate_choices: dict[str, str]
    feature_cols: list[str]
    postprocess_summary: dict[str, object] = field(default_factory=dict)


def _fit_pipeline(
    candidate: Candidate,
    report: ValidationReport | None,
    work: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    tuner: Tuner,
    *,
    with_postproc: bool,
    pre_transform: Transform | None,
) -> _FitResult:
    """Tek bir adayı tüm train'de fit → `FittedModelPipeline` (+ opsiyonel postprocess)."""
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

    median_iter = _median_best_iteration(report)
    params = dict(outcome.best_params)
    if median_iter and candidate.fidelity:
        params[candidate.fidelity] = median_iter

    estimator = build_estimator(candidate, task.task, params)
    if median_iter and candidate.fidelity:
        estimator.fit(x, tt.forward(y))
        best_iter: int | None = median_iter
    else:
        best_iter = fit_estimator(estimator, candidate, x, tt.forward(y), config, task)

    fitted_post: FittedPostprocessor | None = None
    postprocess_summary: dict[str, object] = {}
    if with_postproc:
        postproc = build_postprocessor(config.postprocess, profile, task)
        if postproc.is_active:
            oof = getattr(report, "oof", None)
            cand_post = postproc.fit(getattr(oof, "y_true", None), getattr(oof, "y_pred", None))
            if not cand_post.is_noop:
                fitted_post = cand_post
                postprocess_summary = cand_post.summary

    pipeline = FittedModelPipeline(
        feature_pipeline=fitted_pipe,
        estimator=estimator,
        target_transform=tt,
        feature_cols=list(x.columns),
        reserved=reserved,
        pre_transform=pre_transform,
        postprocessor=fitted_post,
    )
    return _FitResult(
        pipeline=pipeline,
        params=params,
        best_iter=best_iter,
        candidate_choices=outcome.candidate_choices,
        feature_cols=list(x.columns),
        postprocess_summary=postprocess_summary,
    )


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
    pre_transform: Transform | None = None,
) -> ModelBundle:
    """Şampiyonu tüm veride yeniden fit et → `ModelBundle`."""
    key = selection.champion.model_key
    candidate = next((c for c in candidates if c.key == key), None)
    if candidate is None:
        msg = f"Şampiyon aday bulunamadı: {key!r}"
        raise EngineError(msg)

    tuner = tuner or DefaultTuner()
    work = frame.reset_index(drop=True)

    if key == ENSEMBLE_KEY:
        return _refit_ensemble(
            candidate, selection, candidates, reports, work, plan, profile, task, config,
            tuner, pre_transform,
        )

    report = next((r for r in reports if r.candidate_key == key), None)
    fit = _fit_pipeline(
        candidate, report, work, plan, profile, task, config, tuner,
        with_postproc=True, pre_transform=pre_transform,
    )
    champ_row = next((r for r in selection.scoreboard.rows if r.model_key == key), None)
    metadata = BundleMetadata(
        feature_cols=fit.feature_cols,
        feature_set_hash=_feature_hash(fit.feature_cols),
        target_col=task.targets[0],
        model_key=key,
        scenario=selection.champion.scenario,
        best_iteration=fit.best_iter,
        adaptive_plan_summary={
            "structure": plan.structure,
            "committed_ops": len(plan.committed_ops),
            "candidate_choices": fit.candidate_choices,
        },
        params=fit.params,
        postprocess_summary=fit.postprocess_summary,
    )
    return ModelBundle(
        metadata=metadata,
        metrics_oof=dict(champ_row.all_metrics_mean) if champ_row else {},
        pipeline=fit.pipeline,
    )


def _refit_ensemble(
    ens_candidate: Candidate,
    selection: SelectionResult,
    candidates: list[Candidate],
    reports: list[ValidationReport],
    work: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    tuner: Tuner,
    pre_transform: Transform | None,
) -> ModelBundle:
    """`weighted_ensemble` şampiyonu — her üye postprocess'siz refit + `FittedEnsemblePipeline`."""
    members = ens_candidate.ensemble_members or {}
    fitted: list[FittedModelPipeline] = []
    weights: list[float] = []
    member_keys: list[str] = []
    for mkey, weight in members.items():
        mcand = next((c for c in candidates if c.key == mkey), None)
        if mcand is None:
            continue
        mreport = next((r for r in reports if r.candidate_key == mkey), None)
        fit = _fit_pipeline(
            mcand, mreport, work, plan, profile, task, config, tuner,
            with_postproc=False, pre_transform=None,
        )
        fitted.append(fit.pipeline)
        weights.append(float(weight))
        member_keys.append(mkey)
    if len(fitted) < 2:
        msg = "ensemble refit: 2'den az üye fit edilebildi"
        raise EngineError(msg)

    w_arr = np.asarray(weights, dtype=np.float64)
    weights = (w_arr / w_arr.sum()).tolist()

    ens_report = next((r for r in reports if r.candidate_key == ENSEMBLE_KEY), None)
    fitted_post: FittedPostprocessor | None = None
    postprocess_summary: dict[str, object] = {}
    postproc = build_postprocessor(config.postprocess, profile, task)
    if postproc.is_active and ens_report is not None:
        oof = getattr(ens_report, "oof", None)
        cand_post = postproc.fit(getattr(oof, "y_true", None), getattr(oof, "y_pred", None))
        if not cand_post.is_noop:
            fitted_post = cand_post
            postprocess_summary = cand_post.summary

    pipeline = FittedEnsemblePipeline(
        members=fitted, weights=weights, pre_transform=pre_transform, postprocessor=fitted_post
    )
    feature_cols = pipeline.feature_cols
    champ_row = next((r for r in selection.scoreboard.rows if r.model_key == ENSEMBLE_KEY), None)
    metadata = BundleMetadata(
        feature_cols=feature_cols,
        feature_set_hash=_feature_hash(feature_cols),
        target_col=task.targets[0],
        model_key=ENSEMBLE_KEY,
        scenario=selection.champion.scenario,
        best_iteration=None,
        adaptive_plan_summary={"structure": plan.structure, "committed_ops": len(plan.committed_ops)},
        params={},
        postprocess_summary=postprocess_summary,
        ensemble={"members": dict(zip(member_keys, weights, strict=True)), "method": "ges"},
    )
    return ModelBundle(
        metadata=metadata,
        metrics_oof=dict(champ_row.all_metrics_mean) if champ_row else {},
        pipeline=pipeline,
    )
