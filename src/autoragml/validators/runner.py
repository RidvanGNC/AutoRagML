"""Nested CV koşucusu — `ValidationReport` üretir (ADR 0010/6 + 0011 + 0013).

Dış fold'lar yalnız skorlar. HPO + `candidate_ops` seçimi **iç resample**'da (`Tuner`).
`FeaturePipeline` + `TargetTransform` fold içinde fit — split sınırını yalnız buradan görülür.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.tuning import TuningResult
from autoragml.contracts.validation import FoldReport, ValidationReport
from autoragml.logging import get_logger
from autoragml.models import build_estimator
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.scoring.metrics import compute_metrics
from autoragml.validators.frame_ops import (
    OOFArrays,
    column_roles,
    fit_estimator,
    prediction_health,
    reserved_columns,
    split_xy,
    target_transform_choice,
)
from autoragml.validators.leakage_checks import check_fold_leakage, merge_leakage
from autoragml.validators.splitters import Fold, Splitter, resolve_splitter

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]


@dataclass
class TunerOutcome:
    """İç resample sonucu."""

    best_params: dict[str, Any] = field(default_factory=dict)
    candidate_choices: dict[str, str] = field(default_factory=dict)
    best_iteration: int | None = None
    nested: bool = False
    tuning_result: TuningResult | None = None


class Tuner(Protocol):
    """İç resample'da HPO + candidate_ops seçimi (fine_tuners sağlar)."""

    def tune(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
    ) -> TunerOutcome: ...


class DefaultTuner:
    """HPO yok — plan varsayılanlarını döndürür (`hpo_level=none`)."""

    def tune(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
    ) -> TunerOutcome:
        choices = {g.group_name: g.default for g in plan.candidate_ops}
        return TunerOutcome(candidate_choices=choices, nested=False)


def run_validation(
    candidate: Candidate,
    frame: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    *,
    tuner: Tuner | None = None,
    splitter: Splitter | None = None,
    scenario: str = "scenario_1",
) -> ValidationReport:
    """Bir aday için nested CV → `ValidationReport`."""
    start = time.perf_counter()
    tuner = tuner or DefaultTuner()
    work = frame.reset_index(drop=True)
    splitter = splitter or resolve_splitter(work, config, task, profile)
    folds: list[Fold] = splitter.split(work)

    reserved = reserved_columns(task)
    roles = column_roles(profile)
    target = task.targets[0]

    fold_reports: list[FoldReport] = []
    oof_true: list[_Arr] = []
    oof_pred: list[_Arr] = []
    oof_group: list[np.ndarray] = []
    all_violations = []
    any_nested = False

    for fold in folds:
        train = work.iloc[fold.train_idx].reset_index(drop=True)
        test = work.iloc[fold.test_idx].reset_index(drop=True)

        ctx = PlanContext(
            target=target,
            task=task.task,
            column_roles=roles,
            group_col=task.group_col,
            time_col=task.time_col,
            fold_id=fold.fold_id,
            train_span=fold.train_span,
            seed=config.seed,
        )

        outcome = tuner.tune(candidate, train, plan, task, ctx, config)
        any_nested = any_nested or outcome.nested

        pipe = FeaturePipeline.from_plan(plan, outcome.candidate_choices)
        fitted_pipe, train_t = pipe.fit_transform(train, ctx)
        test_t = fitted_pipe.apply(test)

        all_violations.extend(check_fold_leakage(work, fold, task, fitted_pipe))

        x_train, y_train = split_xy(train_t, reserved, target)
        x_test, y_test = split_xy(test_t, reserved, target)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

        tt = TargetTransform(target_transform_choice(plan, outcome.candidate_choices)).fit(y_train)
        est = build_estimator(candidate, task.task, outcome.best_params)
        best_iter = fit_estimator(est, candidate, x_train, tt.forward(y_train), config, task)

        y_pred = tt.inverse(np.asarray(est.predict(x_test), dtype=np.float64))
        metrics = compute_metrics(y_test, y_pred, task.task)

        fold_reports.append(
            FoldReport(
                fold_id=fold.fold_id,
                train_span=fold.train_span,
                test_span=fold.test_span,
                n_train=len(train),
                n_test=len(test),
                metrics=metrics,
                best_iteration=best_iter,
            )
        )
        oof_true.append(np.asarray(y_test, dtype=np.float64))
        oof_pred.append(y_pred)
        if task.group_col and task.group_col in test.columns:
            oof_group.append(test[task.group_col].to_numpy().astype(object))

    oof_t = np.concatenate(oof_true)
    oof_p = np.concatenate(oof_pred)
    oof_metrics = compute_metrics(oof_t, oof_p, task.task)
    group_arr = np.concatenate(oof_group) if len(oof_group) == len(fold_reports) and oof_group else None

    oof_se: dict[str, float] = {}
    if len(fold_reports) >= 2:
        for key in oof_metrics:
            vals = [fr.metrics[key] for fr in fold_reports if key in fr.metrics]
            if len(vals) >= 2:
                oof_se[key] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    return ValidationReport(
        candidate_key=candidate.key,
        scenario=scenario,
        split_kind=splitter.kind,
        folds=fold_reports,
        oof_metrics=oof_metrics,
        oof_metric_se=oof_se,
        prediction_health=prediction_health(oof_t, oof_p),
        leakage=merge_leakage(all_violations),
        nested=any_nested,
        realized_seconds=round(time.perf_counter() - start, 3),
        oof=OOFArrays(y_true=oof_t, y_pred=oof_p, group=group_arr),
    )


def run_validation_suite(
    candidates: list[Candidate],
    frame: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    *,
    tuner: Tuner | None = None,
    scenario: str = "scenario_1",
) -> list[ValidationReport]:
    """Tüm adaylar için aynı split ile `ValidationReport` listesi."""
    work = frame.reset_index(drop=True)
    splitter = resolve_splitter(work, config, task, profile)
    reports: list[ValidationReport] = []
    for candidate in candidates:
        try:
            reports.append(
                run_validation(
                    candidate, work, plan, profile, task, config,
                    tuner=tuner, splitter=splitter, scenario=scenario,
                )
            )
        except Exception as exc:  # noqa: BLE001 - bir aday çökerse diğerleri devam
            logger.warning("[validators] aday `%s` doğrulaması başarısız: %s", candidate.key, exc)
    return reports
