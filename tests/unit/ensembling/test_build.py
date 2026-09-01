"""ensembling.build_weighted_ensemble — sentetik OOF'lardan ensemble kurma (ADR 0021)."""

from __future__ import annotations

import numpy as np

from autoragml.config import resolve_run_config
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import (
    ColumnProfile,
    ColumnStats,
    DataProfile,
    TargetSummary,
)
from autoragml.contracts.enums import Modality, RawDtype, SemanticRole, SplitKind, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.ensembling import build_weighted_ensemble
from autoragml.validators.frame_ops import OOFArrays

_TASK = TaskSpec(task=Task.REGRESSION, modality=Modality.TABULAR, targets=["y"])


def _profile() -> DataProfile:
    tgt = ColumnProfile(
        name="y", raw_dtype=RawDtype.FLOAT, semantic_role=SemanticRole.TARGET,
        stats=ColumnStats(n_unique=100, missing_ratio=0.0, min=-5.0, max=5.0),
    )
    return DataProfile(columns=[tgt], n_rows=300, n_cols=1, target_profile=tgt, target_summary=TargetSummary())


def _report(key: str, y_true: np.ndarray, y_pred: np.ndarray, n_folds: int = 3) -> ValidationReport:
    n = len(y_true)
    sizes = [n // n_folds] * n_folds
    sizes[-1] += n - sum(sizes)
    folds, off = [], 0
    for i, s in enumerate(sizes):
        sl = slice(off, off + s)
        folds.append(FoldReport(
            fold_id=i + 1, n_train=n - s, n_test=s,
            metrics={"rmse": float(np.sqrt(np.mean((y_true[sl] - y_pred[sl]) ** 2)))},
        ))
        off += s
    return ValidationReport(
        candidate_key=key, split_kind=SplitKind.KFOLD, folds=folds,
        oof_metrics={"rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))},
        leakage=LeakageReport(),
        oof=OOFArrays(y_true=y_true, y_pred=y_pred),
    )


def _cands(keys: list[str]) -> list[Candidate]:
    return [
        Candidate(key=k, name=k, family="linear", class_path="x.Y",
                  modalities=[Modality.TABULAR], tasks=[Task.REGRESSION])
        for k in keys
    ]


def _cfg(**ov):
    return resolve_run_config(target="y", overrides=ov).config


def test_none_when_fewer_than_two_aligned() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    reports = [_report("a", y, y + rng.normal(0, 0.3, 200))]
    assert build_weighted_ensemble(reports, _cands(["a"]), _cfg(), _TASK, _profile()) is None


def test_none_when_disabled() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    reports = [_report(k, y, y + rng.normal(0, 0.3, 200)) for k in ("a", "b", "c")]
    cfg = _cfg(ensemble={"enabled": False})
    assert build_weighted_ensemble(reports, _cands(["a", "b", "c"]), cfg, _TASK, _profile()) is None


def test_builds_ensemble_that_beats_best_single() -> None:
    n = 300
    rng = np.random.default_rng(2)
    y = rng.normal(size=n)
    e = rng.normal(0, 0.7, n)
    preds = {"a": y + e, "b": y - e, "c": y + rng.normal(0, 2.0, n)}  # a,b tamamlayıcı
    reports = [_report(k, y, p) for k, p in preds.items()]
    out = build_weighted_ensemble(reports, _cands(list(preds)), _cfg(ensemble={"bagging": False}), _TASK, _profile())
    assert out is not None
    ens_report, ens_candidate, spec = out

    assert ens_candidate.key == "weighted_ensemble"
    assert ens_candidate.family == "ensemble"
    assert set(spec.member_keys) <= {"a", "b", "c"}
    assert abs(sum(spec.weights) - 1.0) < 1e-9
    best_single = min(r.oof_metrics["rmse"] for r in reports)
    assert spec.oof_metric < best_single
    assert len(ens_report.folds) == 3
    assert ens_report.oof_metric_se.get("rmse", 0.0) >= 0.0
    assert ens_candidate.ensemble_members is not None


def test_misaligned_reports_filtered_out() -> None:
    n = 200
    rng = np.random.default_rng(3)
    y = rng.normal(size=n)
    r_ok1 = _report("a", y, y + rng.normal(0, 0.3, n))
    r_ok2 = _report("b", y, y - rng.normal(0, 0.3, n))
    r_bad = _report("c", rng.normal(size=n), rng.normal(size=n))  # farklı y_true
    out = build_weighted_ensemble([r_ok1, r_ok2, r_bad], _cands(["a", "b", "c"]),
                                  _cfg(ensemble={"bagging": False}), _TASK, _profile())
    assert out is not None
    _, _, spec = out
    assert "c" not in spec.member_keys
    assert spec.base_model_count == 2
