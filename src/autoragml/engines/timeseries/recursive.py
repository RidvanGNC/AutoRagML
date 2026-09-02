"""Recursive multi-step reduction — `shift(1)` özellik + adım-adım tahmin (ADR 0026 Bölüm B).

`RunConfig.forecast_reduction="recursive"`. Model **1-adım-ileri** eğitilir; serving/CV
**recursive-h**: her adımda özellik satırları güncel `y` (bilinen geçmiş + önceki tahminler)
ile yeniden kurulur. Leakage-safe: adım `k`'nin özellikleri yalnız `k−1` önceki tahmin/aktüeli görür.

**v1 sınırı (ADR 0026):** CV recursive-h birikimli hatayı ölçer; ama `weighted_ensemble`
recursive + direct adayları karıştırmaz — recursive modda ansambl devre dışı.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import SplitKind
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.engines.timeseries.reduction import build_reduction_features
from autoragml.logging import get_logger
from autoragml.models import build_estimator
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.preprocessors.target import FittedTargetTransform
from autoragml.scoring.metrics import compute_metrics
from autoragml.validators.frame_ops import (
    OOFArrays,
    column_roles,
    fit_estimator,
    prediction_health,
    reserved_columns,
    split_xy,
)
from autoragml.validators.splitters import SplitError, resolve_splitter

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
_SUPPORTED_TARGET = ("none", "log1p")


@dataclass(frozen=True)
class RecursiveRecipe:
    """`FittedRecursivePipeline`'ın özellik reçetesi (saf param — picklable)."""

    task: TaskSpec
    season: int
    add_calendar: bool
    horizon: int


class FittedRecursivePipeline:
    """1-adım model + recursive-h serving döngüsü. `Predictor` protokolü (ADR 0026)."""

    __slots__ = ("_estimator", "_feature_cols", "_feature_pipeline", "_recipe", "_reserved", "_target_transform")

    def __init__(
        self,
        *,
        feature_pipeline: Any,
        estimator: Any,
        target_transform: FittedTargetTransform,
        feature_cols: list[str],
        reserved: set[str],
        recipe: RecursiveRecipe,
    ) -> None:
        self._feature_pipeline = feature_pipeline
        self._estimator = estimator
        self._target_transform = target_transform
        self._feature_cols = feature_cols
        self._reserved = reserved
        self._recipe = recipe

    def _predict_rows(self, feats: pd.DataFrame, mask: np.ndarray) -> _Arr:
        transformed = self._feature_pipeline.apply(feats)
        x = transformed.drop(
            columns=[c for c in self._reserved if c in transformed.columns], errors="ignore"
        )
        x = x.reindex(columns=self._feature_cols, fill_value=0.0)
        x_np = x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)[mask]
        raw = np.asarray(self._estimator.predict(x_np), dtype=np.float64)
        return np.asarray(self._target_transform.inverse(raw), dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> _Arr:
        """Her seri için **son `horizon` satır** recursive tahmin; kalanlar NaN."""
        r = self._recipe
        tc = r.task.time_col or "ds"
        gc, target = r.task.group_col, r.task.targets[0]
        work = frame.copy()
        work[tc] = pd.to_datetime(work[tc], errors="coerce")
        sort_cols = [c for c in (gc, tc) if c and c in work.columns]
        work = work.sort_values(sort_cols).reset_index(drop=True)

        gser = work[gc].astype(str) if gc and gc in work.columns else pd.Series("s", index=work.index)
        tail = work.groupby(gser, sort=False).cumcount(ascending=False).to_numpy()
        y_work = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=np.float64).copy()
        y_work[tail < r.horizon] = np.nan  # forecast satırları başta bilinmiyor

        out = np.full(len(work), np.nan, dtype=np.float64)
        for k in range(r.horizon):
            step_mask = tail == (r.horizon - 1 - k)
            if not step_mask.any():
                continue
            tmp = work.copy()
            tmp[target] = y_work
            feats, _ = build_reduction_features(
                tmp, r.task, horizon=1, season=r.season, add_calendar=r.add_calendar, strategy="recursive"
            )
            # build_reduction_features aynı (gc, tc) sırasıyla döner → maske hizalı
            preds = self._predict_rows(feats, step_mask)
            y_work[step_mask] = preds
            out[step_mask] = preds
        return out

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    @property
    def estimator(self) -> Any:
        return self._estimator


def _target_choice(choices: dict[str, str]) -> str:
    """Recursive v1: yalnız `none`/`log1p` — seasonal_difference recursive'de ref üretmez."""
    c = choices.get("target", "none")
    return c if c in _SUPPORTED_TARGET else "none"


def _fit_one_step(
    candidate: Candidate,
    choices: dict[str, str],
    aug_train: pd.DataFrame,
    plan: AdaptivePlan,
    task: TaskSpec,
    config: RunConfig,
    ctx: PlanContext,
    recipe: RecursiveRecipe,
) -> FittedRecursivePipeline:
    """`aug_train` (recursive özellikleri hazır) üzerinde 1-adım model fit → `FittedRecursivePipeline`."""
    reserved = reserved_columns(task)
    pipe = FeaturePipeline.from_plan(plan, choices)
    fitted_pipe, train_t = pipe.fit_transform(aug_train, ctx)
    x, y = split_xy(train_t, reserved, task.targets[0])
    tt = TargetTransform(_target_choice(choices)).fit(y)
    est = build_estimator(candidate, task.task, {})
    fit_estimator(est, candidate, x, tt.forward(y), config, task)
    return FittedRecursivePipeline(
        feature_pipeline=fitted_pipe,
        estimator=est,
        target_transform=tt,
        feature_cols=list(x.columns),
        reserved=reserved,
        recipe=recipe,
    )


def fit_recursive_champion(
    candidate: Candidate,
    choices: dict[str, str],
    frame: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    *,
    season: int,
) -> FittedRecursivePipeline:
    """Şampiyon refit — tüm train'de tek 1-adım model (recursive; bagging yok, ADR 0026)."""
    horizon = int(task.horizon or (config.split_policy.horizon if config.split_policy else None) or 4)
    recipe = RecursiveRecipe(task=task, season=season, add_calendar=True, horizon=horizon)
    tc, gc = task.time_col or "ds", task.group_col
    work = frame.copy()
    work[tc] = pd.to_datetime(work[tc], errors="coerce")
    work = work.sort_values([c for c in (gc, tc) if c and c in work.columns]).reset_index(drop=True)
    aug, _ = build_reduction_features(
        work, task, horizon=1, season=season, add_calendar=True, strategy="recursive"
    )
    ctx = PlanContext(
        target=task.targets[0], task=task.task, column_roles=column_roles(profile),
        group_col=gc, time_col=tc, seed=config.seed,
    )
    return _fit_one_step(candidate, choices, aug, plan, task, config, ctx, recipe)


def run_recursive_reports(
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    candidates: list[Candidate],
    plan: AdaptivePlan,
    *,
    season: int,
) -> list[ValidationReport]:
    """Recursive reduction adayları → **recursive-h** rolling-origin CV → `ValidationReport`."""
    horizon = int(task.horizon or (config.split_policy.horizon if config.split_policy else None) or 4)
    recipe = RecursiveRecipe(task=task, season=season, add_calendar=True, horizon=horizon)
    tc, gc, target = task.time_col or "ds", task.group_col, task.targets[0]

    work = frame.copy()
    work[tc] = pd.to_datetime(work[tc], errors="coerce")
    work = work.sort_values([c for c in (gc, tc) if c and c in work.columns]).reset_index(drop=True)
    aug, _ = build_reduction_features(
        work, task, horizon=1, season=season, add_calendar=True, strategy="recursive"
    )

    try:
        folds = resolve_splitter(work, config, task, profile).split(work)
    except SplitError as exc:
        logger.warning("[recursive] split kurulamadı: %s", exc)
        return []

    ctx = PlanContext(
        target=target, task=task.task, column_roles=column_roles(profile),
        group_col=gc, time_col=tc, seed=config.seed,
    )
    choices = {g.group_name: g.default for g in plan.candidate_ops}

    reports: list[ValidationReport] = []
    for cand in candidates:
        start = time.perf_counter()
        fold_reports: list[FoldReport] = []
        oof_t: list[_Arr] = []
        oof_p: list[_Arr] = []
        oof_g: list[np.ndarray] = []
        try:
            for fi, fold in enumerate(folds, start=1):
                aug_train = aug.iloc[fold.train_idx].reset_index(drop=True)
                fitted = _fit_one_step(cand, choices, aug_train, plan, task, config, ctx, recipe)

                test = work.iloc[fold.test_idx]
                test_ids = set(test[gc].astype(str)) if gc else {"s"}
                hist = work.iloc[fold.train_idx]
                if gc:
                    hist = hist[hist[gc].astype(str).isin(test_ids)]
                eval_raw = pd.concat([hist, test], ignore_index=True)
                eval_raw = eval_raw.sort_values(
                    [c for c in (gc, tc) if c and c in eval_raw.columns]
                ).reset_index(drop=True)

                preds_all = fitted.predict(eval_raw)
                gs = eval_raw[gc].astype(str) if gc else pd.Series("s", index=eval_raw.index)
                is_fc = (
                    eval_raw.groupby(gs, sort=False).cumcount(ascending=False) < horizon
                ).to_numpy()
                y_pred = preds_all[is_fc]
                y_true = pd.to_numeric(eval_raw.loc[is_fc, target], errors="coerce").to_numpy(np.float64)
                fold_reports.append(
                    FoldReport(
                        fold_id=fi,
                        n_train=len(aug_train),
                        n_test=int(is_fc.sum()),
                        train_span=fold.train_span,
                        test_span=fold.test_span,
                        metrics=compute_metrics(y_true, y_pred, task.task),
                    )
                )
                oof_t.append(y_true)
                oof_p.append(y_pred)
                if gc:
                    oof_g.append(eval_raw.loc[is_fc, gc].to_numpy().astype(object))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[recursive] aday `%s` başarısız: %s", cand.key, exc)
            continue
        if not fold_reports:
            continue
        yt, yp = np.concatenate(oof_t), np.concatenate(oof_p)
        oof_metrics = compute_metrics(yt, yp, task.task)
        oof_se: dict[str, float] = {}
        for mk in oof_metrics:
            vals = [fr.metrics[mk] for fr in fold_reports if np.isfinite(fr.metrics.get(mk, np.nan))]
            if len(vals) >= 2:
                oof_se[mk] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        reports.append(
            ValidationReport(
                candidate_key=cand.key,
                split_kind=SplitKind.ROLLING_ORIGIN,
                folds=fold_reports,
                oof_metrics=oof_metrics,
                oof_metric_se=oof_se,
                prediction_health=prediction_health(yt, yp),
                leakage=LeakageReport(),
                nested=False,
                realized_seconds=round(time.perf_counter() - start, 2),
                oof=OOFArrays(y_true=yt, y_pred=yp, group=np.concatenate(oof_g) if oof_g else None),
            )
        )
    logger.info("[recursive] %d aday recursive-%d CV ile doğrulandı", len(reports), horizon)
    return reports
