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
from autoragml.contracts.enums import SemanticRole, Task
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, ValidationReport
from autoragml.logging import get_logger
from autoragml.models import build_estimator
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.scoring.metrics import compute_metrics
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


def _reserved_columns(task: TaskSpec) -> set[str]:
    return {c for c in (task.time_col, task.group_col, *task.targets) if c}


def _column_roles(profile: DataProfile) -> dict[str, SemanticRole]:
    return {c.name: c.semantic_role for c in profile.columns}


def _xy(frame: pd.DataFrame, reserved: set[str], target: str) -> tuple[pd.DataFrame, _Arr]:
    y = np.asarray(pd.to_numeric(frame[target], errors="coerce"), dtype=np.float64)
    drop = [c for c in reserved if c in frame.columns]
    x = frame.drop(columns=drop)
    numeric = x.select_dtypes(include=["number", "bool"])
    if numeric.shape[1] < x.shape[1]:
        dropped = sorted(set(x.columns) - set(numeric.columns))
        logger.warning("[validators] sayısal olmayan kolonlar X'ten düşürüldü: %s", dropped)
    return numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0), y


def _target_choice(plan: AdaptivePlan, outcome: TunerOutcome) -> str:
    for group in plan.candidate_ops:
        if group.group_name == "target":
            return outcome.candidate_choices.get("target", group.default)
    return "none"


def _inner_val_split(
    n: int, frac: float, is_time_ordered: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    n_val = max(1, int(round(n * frac)))
    if is_time_ordered:
        return np.arange(n - n_val), np.arange(n - n_val, n)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]


def _fit_estimator(
    est: Any,
    candidate: Candidate,
    x: pd.DataFrame,
    y: _Arr,
    config: RunConfig,
    task: TaskSpec,
) -> int | None:
    """Fit + (destekleniyorsa) fold-içi iç-val early stopping. Döner: best_iteration."""
    module = type(est).__module__
    if not candidate.supports_early_stopping or len(x) < 30:
        est.fit(x, y)
        return None

    rounds = candidate.early_stopping_rounds or 50
    frac = config.validation.early_stopping_fraction
    is_ts = task.task is Task.FORECASTING
    tr_idx, val_idx = _inner_val_split(len(x), frac, is_ts)
    x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    if module.startswith("lightgbm"):
        import lightgbm as lgb

        est.fit(
            x_tr,
            y_tr,
            eval_X=x_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(rounds, verbose=False)],
        )
        return int(getattr(est, "best_iteration_", 0)) or None
    if module.startswith("sklearn.ensemble") and hasattr(est, "set_params"):
        est.set_params(early_stopping=True, validation_fraction=frac, n_iter_no_change=rounds)
        est.fit(x, y)
        return int(getattr(est, "n_iter_", 0)) or None
    est.fit(x, y)
    return None


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

    reserved = _reserved_columns(task)
    roles = _column_roles(profile)
    target = task.targets[0]

    fold_reports: list[FoldReport] = []
    oof_true: list[_Arr] = []
    oof_pred: list[_Arr] = []
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

        x_train, y_train = _xy(train_t, reserved, target)
        x_test, y_test = _xy(test_t, reserved, target)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

        tt = TargetTransform(_target_choice(plan, outcome)).fit(y_train)
        est = build_estimator(candidate, task.task, outcome.best_params)
        best_iter = _fit_estimator(est, candidate, x_train, tt.forward(y_train), config, task)

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

    oof_t = np.concatenate(oof_true)
    oof_p = np.concatenate(oof_pred)
    oof_metrics = compute_metrics(oof_t, oof_p, task.task)

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
        leakage=merge_leakage(all_violations),
        nested=any_nested,
        realized_seconds=round(time.perf_counter() - start, 3),
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
