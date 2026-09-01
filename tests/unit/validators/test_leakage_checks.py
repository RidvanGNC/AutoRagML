"""validators.leakage_checks — overlap / preprocessing → BLOCK (ADR 0011/5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.contracts.enums import LeakageCategory, Modality, Provenance, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.validators.leakage_checks import check_fold_leakage, merge_leakage
from autoragml.validators.splitters import Fold


class _Pipe:
    def __init__(self, prov: Provenance) -> None:
        self.provenance_fitted_on = prov


def _reg_task() -> TaskSpec:
    return TaskSpec(task=Task.REGRESSION, modality=Modality.TABULAR, targets=["y"])


def test_row_overlap_detected() -> None:
    frame = pd.DataFrame({"y": range(20)})
    bad = Fold(1, np.array([0, 1, 2, 3, 4]), np.array([3, 4, 5, 6]))
    v = check_fold_leakage(frame, bad, _reg_task(), _Pipe(Provenance.TRAIN))
    assert any(x.category is LeakageCategory.OVERLAP for x in v)


def test_clean_fold_passes() -> None:
    frame = pd.DataFrame({"y": range(20)})
    good = Fold(1, np.arange(0, 15), np.arange(15, 20))
    assert check_fold_leakage(frame, good, _reg_task(), _Pipe(Provenance.TRAIN)) == []


def test_time_overlap_for_forecasting() -> None:
    ds = pd.date_range("2026-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"ds": ds, "y": range(20)})
    task = TaskSpec(task=Task.FORECASTING, modality=Modality.TIMESERIES, targets=["y"], time_col="ds")
    fold = Fold(1, np.arange(0, 12), np.arange(8, 16))  # zaman örtüşüyor
    v = check_fold_leakage(frame, fold, task, _Pipe(Provenance.TRAIN))
    assert any("zaman örtüşmesi" in x.detail for x in v)


def test_preprocessing_provenance_violation() -> None:
    frame = pd.DataFrame({"y": range(20)})
    good = Fold(1, np.arange(0, 15), np.arange(15, 20))
    v = check_fold_leakage(frame, good, _reg_task(), _Pipe(Provenance.TEST))
    assert any(x.category is LeakageCategory.PREPROCESSING for x in v)


def test_merge_leakage_status() -> None:
    assert merge_leakage([]).status == "PASS"
    frame = pd.DataFrame({"y": range(10)})
    bad = Fold(1, np.array([0, 1, 2]), np.array([2, 3]))
    v = check_fold_leakage(frame, bad, _reg_task(), _Pipe(Provenance.TRAIN))
    assert merge_leakage(v).status == "FAIL"
