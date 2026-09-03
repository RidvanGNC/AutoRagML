"""L2 stacking — saf stacker katmanı (ADR 0034).

`build_stack_layer(reports, candidates, task, config)` → hizalı L1 OOF matrisi `Z` üstünde
her **çeşit** L1 model tipi bir L2 stacker olur (saf stacking — orijinal öznitelik yok).
L2 OOF, `Z` üstünde ayrı k-fold ile üretilir (L1 OOF zaten leakage-free → ADR 0011/0022).
Sentetik `ValidationReport` + `Candidate` döner; `engines/core` bunları GES/1-SE havuzuna ekler
("GES over penultimate layer" — AutoGluon deseni).

Hizalama kısıtı ADR 0021 (GES) ile aynı: yalnız nested-CV suite adayları. `neural`/`foundation`
aileleri stack'e girmez (6-sütunlu Z'de nöral anlamsız + picklability; K2 istisnası).
"""

from __future__ import annotations

import numpy as np

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import SplitKind, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.ensembling import _aligned_reports  # noqa: PLC2701 — kardeş modül, ADR 0021 ikizi
from autoragml.logging import get_logger
from autoragml.models.estimator import build_estimator, resolve_class_path
from autoragml.scoring.metrics import compute_metrics, default_primary_metric, lower_is_better
from autoragml.validators.frame_ops import OOFArrays, prediction_health

logger = get_logger(__name__)

__all__ = ["STACK_FAMILY", "STACK_PREFIX", "build_stack_layer", "is_stack"]

STACK_FAMILY = "stack"
STACK_PREFIX = "stack_"
_ELIGIBLE_TASKS = {Task.REGRESSION, Task.FORECASTING}
_EXCLUDED_FAMILIES = {"ensemble", "stack", "neural", "neural_ts", "foundation", "foundation_ts"}
_MIN_MEMBERS = 2
_L2_MAX_FOLDS = 5


def is_stack(candidate: Candidate) -> bool:
    return candidate.family == STACK_FAMILY


def _l1_fold_indices(n_test_per_fold: list[int], n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """L2 CV = L1 fold sınırlarını yeniden kullan (OOF, fold-sırasında bitişik dilimler).

    Böylece stacker OOF'u L1 fold yapısıyla **hizalı** kalır → GES havuzuna da girer
    ("GES over penultimate layer", AutoGluon deseni).
    """
    offsets = np.cumsum([0, *n_test_per_fold])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(n_test_per_fold)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi <= lo:
            continue
        te = np.arange(lo, hi)
        tr = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        folds.append((tr, te))
    return folds


def build_stack_layer(
    reports: list[ValidationReport],
    candidates: list[Candidate],
    task: TaskSpec,
    config: RunConfig,
) -> list[tuple[ValidationReport, Candidate]]:
    """Hizalı L1 OOF'lardan L2 stacker adayları kur. Uygun değilse `[]`."""
    if config.stacking_enabled == "off" or task.task not in _ELIGIBLE_TASKS:
        return []

    by_key = {c.key: c for c in candidates}
    aligned = [
        r for r in _aligned_reports(reports)
        if r.candidate_key in by_key and by_key[r.candidate_key].family not in _EXCLUDED_FAMILIES
    ]
    if len(aligned) < _MIN_MEMBERS:
        return []

    members = [by_key[r.candidate_key] for r in aligned]
    families = {c.family for c in members}
    n_rows = int(aligned[0].oof.y_true.shape[0])
    n_folds = len(aligned[0].folds)

    if config.stacking_enabled == "auto" and (
        n_rows < config.stacking_min_rows
        or len(families) < config.stacking_min_families
        or n_folds < _L2_MAX_FOLDS
    ):
        logger.info(
            "[stacking] auto kapı: n_rows=%d (≥%d?), %d aile (≥%d?), %d fold (≥%d?) — atlandı",
            n_rows, config.stacking_min_rows, len(families), config.stacking_min_families,
            n_folds, _L2_MAX_FOLDS,
        )
        return []

    member_keys = [c.key for c in members]
    y = aligned[0].oof.y_true.astype(np.float64)
    group = aligned[0].oof.group
    z = np.column_stack([r.oof.y_pred.astype(np.float64) for r in aligned])
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)
    l1_metrics = [r.oof_metrics.get(primary) for r in aligned]
    l1_finite = [m for m in l1_metrics if m is not None and np.isfinite(m)]
    if not l1_finite:
        return []
    best_l1 = min(l1_finite) if lower else max(l1_finite)

    folds = _l1_fold_indices([f.n_test for f in aligned[0].folds], n_rows)

    out: list[tuple[ValidationReport, Candidate]] = []
    seen: set[str] = set()
    for base in members:
        sig = f"{base.family}|{resolve_class_path(base.class_path, task.task)}"
        if sig in seen:
            continue
        seen.add(sig)
        oof_pred = _stack_oof(base, task, z, y, folds)
        if oof_pred is None:
            continue
        m = compute_metrics(y, oof_pred, task.task).get(primary)
        if m is None or not np.isfinite(m) or (lower and m >= best_l1) or (not lower and m <= best_l1):
            continue  # guard — en iyi L1'i geçemeyen stacker önerilmez
        stacker_key = f"{STACK_PREFIX}{base.key}"
        report = _stack_report(stacker_key, y, oof_pred, folds, group, task)
        cand = Candidate(
            key=stacker_key,
            name=f"Stack ({base.name})",
            family=STACK_FAMILY,
            class_path="__stack__",
            modalities=[task.modality],
            tasks=[task.task],
            default_params={"stack_base_key": base.key},
            ensemble_members=dict.fromkeys(member_keys, 1.0),
        )
        out.append((report, cand))
        logger.info("[stacking] %s: OOF %s=%.4g (en iyi L1=%.4g)", stacker_key, primary, m, best_l1)

    if not out:
        logger.info("[stacking] hiçbir L2 stacker en iyi L1'i geçmedi — stack katmanı yok")
    return out


def _stack_oof(
    base: Candidate,
    task: TaskSpec,
    z: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray | None:
    """`Z` üstünde k-fold → L2 OOF vektörü (leakage-free; L1 OOF zaten OOF)."""
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for tr, te in folds:
        try:
            est = build_estimator(base, task.task, {})
            est.fit(z[tr], y[tr])
            oof[te] = np.asarray(est.predict(z[te]), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — bir stacker çökerse diğerleri devam
            logger.warning("[stacking] `%s` stacker fit başarısız: %s", base.key, exc)
            return None
    return oof if not np.isnan(oof).any() else None


def _stack_report(
    key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    group: np.ndarray | None,
    task: TaskSpec,
) -> ValidationReport:
    """L2 OOF → sentetik `ValidationReport` (fold metrikleri + SE)."""
    fold_reports: list[FoldReport] = []
    for fid, (_, te) in enumerate(folds):
        fold_reports.append(
            FoldReport(
                fold_id=fid,
                n_train=0,
                n_test=int(len(te)),
                metrics=compute_metrics(y_true[te], y_pred[te], task.task),
            )
        )
    oof_metrics = compute_metrics(y_true, y_pred, task.task)
    oof_se: dict[str, float] = {}
    if len(fold_reports) >= 2:
        for k in oof_metrics:
            vals = [fr.metrics[k] for fr in fold_reports if k in fr.metrics and np.isfinite(fr.metrics[k])]
            if len(vals) >= 2:
                oof_se[k] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return ValidationReport(
        candidate_key=key,
        split_kind=SplitKind.KFOLD,
        folds=fold_reports,
        oof_metrics=oof_metrics,
        oof_metric_se=oof_se,
        prediction_health=prediction_health(y_true, y_pred),
        leakage=LeakageReport(),
        nested=True,
        oof=OOFArrays(y_true=y_true, y_pred=y_pred, group=group),
    )
