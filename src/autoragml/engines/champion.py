"""Şampiyon refit — seçilen adayı tüm train'de yeniden fit et (ADR 0013/0014/0021/0022).

**k-fold bagging (ADR 0022, varsayılan):** tek model / %100 train yerine k fold-modeli;
serving = ortalama (`FittedEnsemblePipeline`, eşit ağırlık). Bagged OOF postprocess'e girer.
`weighted_ensemble` şampiyonda her GES üyesi de bagged. ES modelleri her fold'da kendi
early stopping'ini yapar.
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
from autoragml.contracts.enums import Task
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.scoreboard import SelectionResult
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import ValidationReport
from autoragml.engines.ensemble_pipeline import FittedEnsemblePipeline
from autoragml.engines.model_pipeline import FittedModelPipeline, Predictor
from autoragml.engines.timeseries.classical import (
    CLASSICAL_ENSEMBLE_KEY,
    is_classical,
    refit_classical,
    refit_classical_ensemble,
)
from autoragml.ensembling import ENSEMBLE_KEY
from autoragml.exceptions import EngineError
from autoragml.logging import get_logger
from autoragml.models import build_estimator
from autoragml.postprocessors import FittedPostprocessor, build_postprocessor
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.validators import DefaultTuner, Tuner
from autoragml.validators.frame_ops import (
    column_roles,
    fit_estimator,
    reserved_columns,
    sdiff_ref,
    sdiff_ref_col,
    split_xy,
    target_transform_choice,
)
from autoragml.validators.splitters import Fold, SplitError, resolve_splitter

logger = get_logger(__name__)

Transform = Callable[[pd.DataFrame], pd.DataFrame]

# v1: bagging yalnız regresyon/forecasting — sınıflandırmada hard-label ortalaması sürekli
# değer üretir (olasılık ortalaması + argmax → v1.1, sınıflandırma GES ile birlikte).
_BAGGABLE_TASKS = {Task.REGRESSION, Task.FORECASTING, Task.QUANTILE_REGRESSION, Task.ORDINAL_REGRESSION}


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(cols)).encode()).hexdigest()[:16]


def _median_best_iteration(report: ValidationReport | None) -> int | None:
    if report is None:
        return None
    iters = [fr.best_iteration for fr in report.folds if fr.best_iteration is not None]
    return int(np.median(iters)) if iters else None


@dataclass
class _FitResult:
    pipeline: Predictor  # FittedModelPipeline | FittedEnsemblePipeline (bag)
    params: dict[str, object]
    best_iter: int | None
    candidate_choices: dict[str, str]
    feature_cols: list[str]
    n_bag: int = 1
    postprocess_summary: dict[str, object] = field(default_factory=dict)


def _ctx(task: TaskSpec, profile: DataProfile, config: RunConfig) -> PlanContext:
    return PlanContext(
        target=task.targets[0],
        task=task.task,
        column_roles=column_roles(profile),
        group_col=task.group_col,
        time_col=task.time_col,
        seed=config.seed,
    )


def _fit_one(
    candidate: Candidate,
    choices: dict[str, str],
    params: dict[str, object],
    train_df: pd.DataFrame,
    plan: AdaptivePlan,
    task: TaskSpec,
    config: RunConfig,
    ctx: PlanContext,
    *,
    fixed_iter: int | None,
    pre_transform: Transform | None = None,
    postprocessor: FittedPostprocessor | None = None,
) -> tuple[FittedModelPipeline, int | None]:
    """Bir aday + sabit config → `train_df` üzerinde tek `FittedModelPipeline`."""
    reserved = reserved_columns(task)
    target = task.targets[0]
    pipe = FeaturePipeline.from_plan(plan, choices)
    fitted_pipe, frame_t = pipe.fit_transform(train_df, ctx)
    x, y = split_xy(frame_t, reserved, target)

    choice = target_transform_choice(plan, choices)
    ref = sdiff_ref(frame_t, target, choice)
    if choice == "seasonal_difference" and ref is None:
        choice = "none"
    if ref is not None:  # seasonal_difference: ref NaN warmup at
        keep = ~np.isnan(ref)
        x, y, ref = x[keep], y[keep], ref[keep]
    tt = TargetTransform(choice).fit(y)
    y_fwd = tt.forward(y, ref=ref)

    run_params = dict(params)
    if fixed_iter and candidate.fidelity:
        run_params[candidate.fidelity] = fixed_iter
    estimator = build_estimator(candidate, task.task, run_params)
    if fixed_iter and candidate.fidelity:
        estimator.fit(x, y_fwd)
        best_iter: int | None = fixed_iter
    else:
        best_iter = fit_estimator(estimator, candidate, x, y_fwd, config, task)

    pipeline = FittedModelPipeline(
        feature_pipeline=fitted_pipe,
        estimator=estimator,
        target_transform=tt,
        feature_cols=list(x.columns),
        reserved=reserved,
        target_ref_col=sdiff_ref_col(target) if choice == "seasonal_difference" else None,
        pre_transform=pre_transform,
        postprocessor=postprocessor,
    )
    return pipeline, best_iter


def _bag_folds(
    work: pd.DataFrame, config: RunConfig, task: TaskSpec, profile: DataProfile, *, want_bag: bool
) -> list[Fold] | None:
    """Bagging fold'ları (≥2) ya da `None` (tek-model refit)."""
    bc = config.bagging
    if not (want_bag and bc.enabled and task.task in _BAGGABLE_TASKS):
        return None
    if bc.max_rows is not None and len(work) > bc.max_rows:
        logger.info("[champion] %d satır > bagging.max_rows — tek model refit", len(work))
        return None
    # Splitter türü task/profil'den; fold sayısı `bagging.folds`.
    bag_cfg = config.model_copy(
        update={
            "validation": config.validation.model_copy(
                update={"default_kfold_splits": bc.folds, "default_rolling_folds": bc.folds}
            )
        }
    )
    try:
        splitter = resolve_splitter(work, bag_cfg, task, profile)
        folds = splitter.split(work.reset_index(drop=True))
    except SplitError as exc:
        logger.info("[champion] bagging split kurulamadı (%s) — tek model refit", exc)
        return None
    return folds if len(folds) >= 2 else None


def _maybe_postproc(
    with_postproc: bool,
    config: RunConfig,
    profile: DataProfile,
    task: TaskSpec,
    y_true: object,
    y_pred: object,
) -> tuple[FittedPostprocessor | None, dict[str, object]]:
    if not with_postproc:
        return None, {}
    postproc = build_postprocessor(config.postprocess, profile, task)
    if not postproc.is_active:
        return None, {}
    fitted = postproc.fit(y_true, y_pred)  # type: ignore[arg-type]
    return (fitted, fitted.summary) if not fitted.is_noop else (None, {})


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
    want_bag: bool = True,
) -> _FitResult:
    """Adayı tüm train'de fit — k-fold bagged (varsayılan) veya tek model."""
    ctx = _ctx(task, profile, config)
    outcome = tuner.tune(candidate, work, plan, task, ctx, config)
    choices = outcome.candidate_choices
    params = dict(outcome.best_params)
    target = task.targets[0]

    folds = _bag_folds(work, config, task, profile, want_bag=want_bag)

    if folds is not None:
        members: list[FittedModelPipeline] = []
        best_iters: list[int] = []
        oof_true: list[np.ndarray] = []
        oof_pred: list[np.ndarray] = []
        for fold in folds:
            tr = work.iloc[fold.train_idx].reset_index(drop=True)
            va = work.iloc[fold.test_idx].reset_index(drop=True)
            member, bi = _fit_one(candidate, choices, params, tr, plan, task, config, ctx, fixed_iter=None)
            members.append(member)
            if bi is not None:
                best_iters.append(bi)
            oof_pred.append(np.asarray(member.predict(va), dtype=np.float64))
            oof_true.append(pd.to_numeric(va[target], errors="coerce").to_numpy(dtype=np.float64))

        fitted_post, summary = _maybe_postproc(
            with_postproc, config, profile, task, np.concatenate(oof_true), np.concatenate(oof_pred)
        )
        bag = FittedEnsemblePipeline(
            members=members,
            weights=[1.0 / len(members)] * len(members),
            pre_transform=pre_transform,
            postprocessor=fitted_post,
        )
        return _FitResult(
            pipeline=bag,
            params=params,
            best_iter=int(np.median(best_iters)) if best_iters else None,
            candidate_choices=choices,
            feature_cols=bag.feature_cols,
            n_bag=len(members),
            postprocess_summary=summary,
        )

    # tek model — full train (refit_full benzeri)
    oof = getattr(report, "oof", None)
    fitted_post, summary = _maybe_postproc(
        with_postproc, config, profile, task,
        getattr(oof, "y_true", None), getattr(oof, "y_pred", None),
    )
    single, best_iter = _fit_one(
        candidate, choices, params, work, plan, task, config, ctx,
        fixed_iter=_median_best_iteration(report),
        pre_transform=pre_transform,
        postprocessor=fitted_post,
    )
    return _FitResult(
        pipeline=single,
        params=params,
        best_iter=best_iter,
        candidate_choices=choices,
        feature_cols=single.feature_cols,
        postprocess_summary=summary,
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

    if is_classical(candidate) or key == CLASSICAL_ENSEMBLE_KEY:
        return _classical_bundle(
            candidate, candidates, selection, reports, frame, profile, task, config
        )

    report = next((r for r in reports if r.candidate_key == key), None)
    fit = _fit_pipeline(
        candidate, report, work, plan, profile, task, config, tuner,
        with_postproc=True, pre_transform=pre_transform,
    )
    champ_row = next((r for r in selection.scoreboard.rows if r.model_key == key), None)
    ensemble_meta: dict[str, object] = (
        {"bagged": True, "folds": fit.n_bag, "model": key} if fit.n_bag > 1 else {}
    )
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
        ensemble=ensemble_meta,
    )
    return ModelBundle(
        metadata=metadata,
        metrics_oof=dict(champ_row.all_metrics_mean) if champ_row else {},
        pipeline=fit.pipeline,
    )


def _classical_bundle(
    candidate: Candidate,
    candidates: list[Candidate],
    selection: SelectionResult,
    reports: list[ValidationReport],
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
) -> ModelBundle:
    """Klasik (StatsForecast) tek model veya EAT ansamblı şampiyonu → `ModelBundle` (ADR 0023/0024)."""
    if candidate.key == CLASSICAL_ENSEMBLE_KEY:
        forecaster = refit_classical_ensemble(candidate, candidates, frame, profile, task, config)
        ensemble_meta: dict[str, object] = {"members": dict(candidate.ensemble_members or {}), "method": "ges"}
    else:
        forecaster = refit_classical(candidate, frame, profile, task, config)
        ensemble_meta = {}
    champ_row = next((r for r in selection.scoreboard.rows if r.model_key == candidate.key), None)
    metadata = BundleMetadata(
        feature_cols=[],
        feature_set_hash=_feature_hash([candidate.key]),
        target_col=task.targets[0],
        model_key=candidate.key,
        scenario=selection.champion.scenario,
        best_iteration=None,
        adaptive_plan_summary={"structure": "per_series_classical"},
        params={"family": candidate.family, "engine": "statsforecast"},
        ensemble=ensemble_meta,
    )
    return ModelBundle(
        metadata=metadata,
        metrics_oof=dict(champ_row.all_metrics_mean) if champ_row else {},
        pipeline=forecaster,
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
    """`weighted_ensemble` şampiyonu — her üye (bagged) postprocess'siz refit + `FittedEnsemblePipeline`."""
    members = ens_candidate.ensemble_members or {}
    fitted: list[Predictor] = []
    weights: list[float] = []
    member_keys: list[str] = []
    total_bag = 0
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
        total_bag += fit.n_bag
    if len(fitted) < 2:
        msg = "ensemble refit: 2'den az üye fit edilebildi"
        raise EngineError(msg)

    w_arr = np.asarray(weights, dtype=np.float64)
    weights = (w_arr / w_arr.sum()).tolist()

    ens_report = next((r for r in reports if r.candidate_key == ENSEMBLE_KEY), None)
    oof = getattr(ens_report, "oof", None)
    fitted_post, postprocess_summary = _maybe_postproc(
        True, config, profile, task, getattr(oof, "y_true", None), getattr(oof, "y_pred", None)
    )

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
        ensemble={
            "members": dict(zip(member_keys, weights, strict=True)),
            "method": "ges",
            "bagged": total_bag > len(member_keys),
        },
    )
    return ModelBundle(
        metadata=metadata,
        metrics_oof=dict(champ_row.all_metrics_mean) if champ_row else {},
        pipeline=pipeline,
    )
