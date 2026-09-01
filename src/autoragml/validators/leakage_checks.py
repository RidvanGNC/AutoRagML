"""Sert sızıntı denetimi — BLOCK (ADR 0011/5).

Kategoriler: `overlap` (satır/zaman/grup), `preprocessing` (split öncesi fit),
`multi_test` (runner tarafından garanti edilir — burada yalnız yapısal kontrol).
"""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.enums import LeakageCategory, Provenance, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import LeakageReport, LeakageViolation
from autoragml.validators.splitters import Fold


def check_fold_leakage(
    frame: pd.DataFrame,
    fold: Fold,
    task: TaskSpec,
    fitted_pipeline: object,
) -> list[LeakageViolation]:
    """Bir fold için ihlalleri döndür (boş → temiz)."""
    violations: list[LeakageViolation] = []

    if set(fold.train_idx.tolist()) & set(fold.test_idx.tolist()):
        violations.append(
            LeakageViolation(
                category=LeakageCategory.OVERLAP,
                detail="train/test satır örtüşmesi",
                fold_id=fold.fold_id,
            )
        )

    if task.time_col and task.time_col in frame.columns and task.task is Task.FORECASTING:
        ts = pd.to_datetime(frame[task.time_col], errors="coerce")
        train_max = ts.iloc[fold.train_idx].max()
        test_min = ts.iloc[fold.test_idx].min()
        if pd.notna(train_max) and pd.notna(test_min) and train_max >= test_min:
            violations.append(
                LeakageViolation(
                    category=LeakageCategory.OVERLAP,
                    detail=f"zaman örtüşmesi: train_max={train_max} >= test_min={test_min}",
                    fold_id=fold.fold_id,
                )
            )

    if task.group_col and task.group_col in frame.columns:
        g = frame[task.group_col]
        shared = set(g.iloc[fold.train_idx]) & set(g.iloc[fold.test_idx])
        if shared and task.task is not Task.FORECASTING:
            violations.append(
                LeakageViolation(
                    category=LeakageCategory.OVERLAP,
                    detail=f"grup örtüşmesi: {sorted(str(s) for s in shared)[:5]}",
                    fold_id=fold.fold_id,
                )
            )

    prov = getattr(fitted_pipeline, "provenance_fitted_on", None)
    if prov is not None and prov is not Provenance.TRAIN:
        violations.append(
            LeakageViolation(
                category=LeakageCategory.PREPROCESSING,
                detail=f"pipeline provenance={prov} (train bekleniyordu)",
                fold_id=fold.fold_id,
            )
        )

    return violations


def merge_leakage(all_violations: list[LeakageViolation]) -> LeakageReport:
    """Fold ihlallerini tek rapora topla."""
    return LeakageReport(
        status="FAIL" if all_violations else "PASS",
        violations=all_violations,
    )
