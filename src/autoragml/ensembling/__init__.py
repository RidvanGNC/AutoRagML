"""ensembling — Caruana greedy selection ile post-hoc ağırlıklı ensemble (ADR 0021).

`build_weighted_ensemble(reports, candidates, config, task, profile)` → hizalı OOF
tahminlerinden GES/bagged-GES → sentetik `ValidationReport` + `Candidate` + `EnsembleSpec`.
Tek-model şampiyonuyla aynı 1-SE seçiminde yarışır. v1: regresyon + forecasting.
"""

from __future__ import annotations

import numpy as np

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.ensemble_spec import EnsembleSpec
from autoragml.contracts.enums import Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.ensembling.greedy import (
    bagged_greedy_selection,
    greedy_selection,
    greedy_selection_proba,
)
from autoragml.logging import get_logger
from autoragml.scoring.metrics import (
    compute_metrics,
    compute_proba_metrics,
    default_primary_metric,
    is_proba_metric,
    lower_is_better,
)
from autoragml.validators.frame_ops import OOFArrays, prediction_health

logger = get_logger(__name__)

__all__ = ["ENSEMBLE_KEY", "build_weighted_ensemble"]

ENSEMBLE_KEY = "weighted_ensemble"
_ELIGIBLE_TASKS = {Task.REGRESSION, Task.FORECASTING}
_CLASSIFICATION_TASKS = {Task.BINARY_CLASSIFICATION, Task.MULTICLASS_CLASSIFICATION}
_W_EPS = 1e-9

BuildResult = tuple[ValidationReport, Candidate, EnsembleSpec]


def _aligned_reports(reports: list[ValidationReport]) -> list[ValidationReport]:
    """OOF'u dolu ve ilk report ile aynı `y_true`/fold yapısına sahip olanlar."""
    usable = [
        r for r in reports
        if getattr(r, "oof", None) is not None
        and getattr(r.oof, "y_pred", None) is not None
        and getattr(r.oof, "y_true", None) is not None
    ]
    if not usable:
        return []
    ref = usable[0]
    ref_shape = ref.oof.y_true.shape
    ref_ntest = [f.n_test for f in ref.folds]
    out: list[ValidationReport] = []
    for r in usable:
        if r.oof.y_true.shape != ref_shape or r.oof.y_pred.shape != ref_shape:
            continue
        if [f.n_test for f in r.folds] != ref_ntest:
            continue
        if not np.allclose(r.oof.y_true, ref.oof.y_true, equal_nan=True):
            continue
        out.append(r)
    return out


def build_weighted_ensemble(
    reports: list[ValidationReport],
    candidates: list[Candidate],
    config: RunConfig,
    task: TaskSpec,
    profile: DataProfile,
) -> BuildResult | None:
    """Hizalı OOF'lardan ağırlıklı ensemble kur. Uygun değilse `None`."""
    ec = config.ensemble
    if not ec.enabled:
        return None
    if task.task in _CLASSIFICATION_TASKS:
        return _build_classification_ensemble(reports, candidates, config, task)
    if task.task not in _ELIGIBLE_TASKS:
        return None

    # ADR 0034: L2 stacker'lar 1-SE seçiminde doğrudan yarışır ama GES üyesi DEĞİL
    # (GES = L1-düzeyi blend; "GES over penultimate layer" → v1.1+).
    stack_keys = {c.key for c in candidates if c.family == "stack"}
    aligned = [r for r in _aligned_reports(reports) if r.candidate_key not in stack_keys]
    if len(aligned) < ec.min_base_models:
        return None

    keys = [r.candidate_key for r in aligned]
    y_true = aligned[0].oof.y_true.astype(np.float64)
    preds = np.column_stack([r.oof.y_pred.astype(np.float64) for r in aligned])

    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)

    def metric_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return compute_metrics(yt, yp, task.task).get(primary, float("inf"))

    if ec.bagging:
        weights = bagged_greedy_selection(
            preds, y_true, metric_fn=metric_fn, lower_is_better=lower,
            max_models=ec.max_models, sorted_init_k=ec.sorted_init_k,
            n_bags=ec.n_bags, bag_fraction=ec.bag_fraction, seed=config.seed,
        )
        method: str = "bagged_ges"
        n_bags = ec.n_bags
    else:
        weights = greedy_selection(
            preds, y_true, metric_fn=metric_fn, lower_is_better=lower,
            max_models=ec.max_models, sorted_init_k=ec.sorted_init_k,
        )
        method = "ges"
        n_bags = 0

    nz = np.flatnonzero(weights > _W_EPS)
    if nz.size < 2:  # tek üye → ensemble anlamsız
        logger.info("[ensembling] GES tek modele indi — ensemble atlandı")
        return None

    w = weights[nz] / weights[nz].sum()
    member_keys = [keys[i] for i in nz]
    member_weights = w.tolist()
    blend = preds[:, nz] @ w

    ref = aligned[0]
    offsets = np.cumsum([0, *[f.n_test for f in ref.folds]])
    fold_reports: list[FoldReport] = []
    for i, f in enumerate(ref.folds):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        fold_reports.append(
            FoldReport(
                fold_id=f.fold_id,
                train_span=f.train_span,
                test_span=f.test_span,
                n_train=f.n_train,
                n_test=f.n_test,
                metrics=compute_metrics(y_true[sl], blend[sl], task.task),
            )
        )

    oof_metrics = compute_metrics(y_true, blend, task.task)
    oof_se: dict[str, float] = {}
    if len(fold_reports) >= 2:
        for k in oof_metrics:
            vals = [fr.metrics[k] for fr in fold_reports if k in fr.metrics]
            if len(vals) >= 2:
                oof_se[k] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    ens_report = ValidationReport(
        candidate_key=ENSEMBLE_KEY,
        scenario=ref.scenario,
        split_kind=ref.split_kind,
        folds=fold_reports,
        oof_metrics=oof_metrics,
        oof_metric_se=oof_se,
        prediction_health=prediction_health(y_true, blend),
        leakage=LeakageReport(),
        nested=True,
        realized_seconds=0.0,
        oof=OOFArrays(y_true=y_true, y_pred=blend, group=ref.oof.group),
    )
    ens_candidate = Candidate(
        key=ENSEMBLE_KEY,
        name="Weighted Ensemble (GES)",
        family="ensemble",
        class_path="__ensemble__",
        modalities=[task.modality],
        tasks=[task.task],
        ensemble_members=dict(zip(member_keys, member_weights, strict=True)),
    )
    spec = EnsembleSpec(
        member_keys=member_keys,
        weights=member_weights,
        method=method,  # type: ignore[arg-type]
        n_bags=n_bags,
        oof_metric=float(oof_metrics.get(primary, float("inf"))),
        base_model_count=len(aligned),
    )
    logger.info(
        "[ensembling] %s: %d üye / %d taban (OOF %s=%.4g)",
        method, len(member_keys), len(aligned), primary, spec.oof_metric,
    )
    return ens_report, ens_candidate, spec


def _proba_aligned(reports: list[ValidationReport], stack_keys: set[str]) -> list[ValidationReport]:
    """`y_proba` dolu, aynı `y_true` + aynı sınıf sayısına sahip raporlar (ADR 0036)."""
    usable = [
        r for r in reports
        if r.candidate_key not in stack_keys
        and getattr(r, "oof", None) is not None
        and getattr(r.oof, "y_proba", None) is not None
        and getattr(r.oof, "y_true", None) is not None
    ]
    if not usable:
        return []
    ref = usable[0].oof
    ref_n, ref_c = ref.y_proba.shape
    ref_ntest = [f.n_test for f in usable[0].folds]
    out: list[ValidationReport] = []
    for r in usable:
        if r.oof.y_proba.shape != (ref_n, ref_c):
            continue
        if [f.n_test for f in r.folds] != ref_ntest:
            continue
        if not np.allclose(r.oof.y_true, ref.y_true, equal_nan=True):
            continue
        out.append(r)
    return out


def _build_classification_ensemble(
    reports: list[ValidationReport],
    candidates: list[Candidate],
    config: RunConfig,
    task: TaskSpec,
) -> BuildResult | None:
    """Sınıflandırma GES (ADR 0036) — olasılık OOF üstünde Caruana; blend proba → argmax."""
    ec = config.ensemble
    stack_keys = {c.key for c in candidates if c.family == "stack"}
    aligned = _proba_aligned(reports, stack_keys)
    if len(aligned) < ec.min_base_models:
        return None

    keys = [r.candidate_key for r in aligned]
    y_true = aligned[0].oof.y_true
    classes = aligned[0].oof.classes
    proba_stack = np.stack([r.oof.y_proba.astype(np.float64) for r in aligned])  # (m, n, C)

    primary = config.primary_metric or default_primary_metric(task.task)
    search_metric = primary if is_proba_metric(primary) else "log_loss"
    lower = lower_is_better(search_metric)

    def metric_fn(yt: np.ndarray, mean_proba: np.ndarray) -> float:
        return compute_proba_metrics(yt, mean_proba, classes=classes).get(search_metric, float("inf"))

    weights = greedy_selection_proba(
        proba_stack, y_true, metric_fn=metric_fn, lower_is_better=lower,
        max_models=ec.max_models, sorted_init_k=ec.sorted_init_k,
    )
    nz = np.flatnonzero(weights > _W_EPS)
    if nz.size < 2:
        logger.info("[ensembling] sınıflandırma GES tek modele indi — ensemble atlandı")
        return None
    w = weights[nz] / weights[nz].sum()
    member_keys = [keys[i] for i in nz]
    member_weights = w.tolist()
    blend = np.tensordot(w, proba_stack[nz], axes=(0, 0))  # (n, C)
    labels = np.asarray(classes) if classes is not None else np.unique(y_true)
    blend_pred = labels[np.argmax(blend, axis=1)].astype(np.float64)

    ref = aligned[0]
    offsets = np.cumsum([0, *[f.n_test for f in ref.folds]])
    fold_reports: list[FoldReport] = []
    for i, f in enumerate(ref.folds):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        fold_reports.append(
            FoldReport(
                fold_id=f.fold_id, n_train=f.n_train, n_test=f.n_test,
                metrics=compute_proba_metrics(y_true[sl], blend[sl], classes=classes),
            )
        )
    oof_metrics = compute_proba_metrics(y_true, blend, classes=classes)
    oof_se: dict[str, float] = {}
    if len(fold_reports) >= 2:
        for k in oof_metrics:
            vals = [fr.metrics[k] for fr in fold_reports if k in fr.metrics and np.isfinite(fr.metrics[k])]
            if len(vals) >= 2:
                oof_se[k] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    ens_report = ValidationReport(
        candidate_key=ENSEMBLE_KEY,
        scenario=ref.scenario,
        split_kind=ref.split_kind,
        folds=fold_reports,
        oof_metrics=oof_metrics,
        oof_metric_se=oof_se,
        prediction_health=prediction_health(y_true.astype(np.float64), blend_pred),
        leakage=LeakageReport(),
        nested=True,
        realized_seconds=0.0,
        oof=OOFArrays(y_true=y_true, y_pred=blend_pred, group=ref.oof.group, y_proba=blend, classes=classes),
    )
    ens_candidate = Candidate(
        key=ENSEMBLE_KEY,
        name="Weighted Ensemble (GES, proba)",
        family="ensemble",
        class_path="__ensemble__",
        modalities=[task.modality],
        tasks=[task.task],
        ensemble_members=dict(zip(member_keys, member_weights, strict=True)),
    )
    spec = EnsembleSpec(
        member_keys=member_keys,
        weights=member_weights,
        method="ges",
        n_bags=0,
        oof_metric=float(oof_metrics.get(primary, float("inf"))),
        base_model_count=len(aligned),
    )
    logger.info(
        "[ensembling] sınıflandırma GES: %d üye / %d taban (arama=%s, OOF %s=%.4g)",
        len(member_keys), len(aligned), search_metric, primary, spec.oof_metric,
    )
    return ens_report, ens_candidate, spec
