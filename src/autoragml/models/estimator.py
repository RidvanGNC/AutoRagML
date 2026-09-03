"""Candidate → somut estimator (ADR 0012/0013).

`class_path` task ailesine göre çözülür (forecasting → reduction ile `regression` path'i).
`wrap=True` → linear/mesafe modelleri için imputer güvenlik ağı (preprocessors çoğunu
zaten yapar; bu artık NaN'lara karşı son savunma).
"""

from __future__ import annotations

import importlib
from typing import Any

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import Task
from autoragml.exceptions import AutoRagMLError

_REGRESSION_TASKS = {Task.REGRESSION, Task.FORECASTING, Task.QUANTILE_REGRESSION, Task.ORDINAL_REGRESSION}
_CLASSIFICATION_TASKS = {
    Task.BINARY_CLASSIFICATION,
    Task.MULTICLASS_CLASSIFICATION,
    Task.MULTILABEL_CLASSIFICATION,
}


class EstimatorBuildError(AutoRagMLError):
    """Estimator sınıfı çözülemedi / kurulamadı."""


def resolve_class_path(class_path: str | dict[str, str], task: Task) -> str:
    """Task ailesine göre estimator sınıf yolunu seç."""
    if isinstance(class_path, str):
        return class_path
    family = "regression" if task in _REGRESSION_TASKS else "classification"
    if family in class_path:
        return class_path[family]
    msg = f"{task} için class_path ailesi ({family}) yok: {sorted(class_path)}"
    raise EstimatorBuildError(msg)


def _load_class(class_path: str) -> type[Any]:
    module_path, _, class_name = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)  # type: ignore[no-any-return]
    except (ImportError, AttributeError) as exc:
        msg = f"Estimator sınıfı yüklenemedi: {class_path} ({exc})"
        raise EstimatorBuildError(msg) from exc


_NEURAL_ARCH = "__neural_arch__"
_FOUNDATION_TAB = "__foundation_tab__"


def build_estimator(
    candidate: Candidate,
    task: Task,
    params: dict[str, Any] | None = None,
) -> Any:
    """Bir aday + task + (opsiyonel) HPO paramları → fit edilmemiş estimator."""
    class_path = resolve_class_path(candidate.class_path, task)
    if class_path == _NEURAL_ARCH:  # ADR 0031: pytorch_tabular mimari arama sarımı
        from autoragml.models.neural_arch import TabularModelEstimator

        merged = {**candidate.default_params, **(params or {})}
        merged["task_kind"] = "regression" if task in _REGRESSION_TASKS else "classification"
        return TabularModelEstimator(**merged)
    if class_path == _FOUNDATION_TAB:  # ADR 0033: TabPFN in-context sarımı
        from autoragml.models.foundation_tab import FoundationTabEstimator

        merged = {**candidate.default_params, **(params or {})}
        merged["task_kind"] = "regression" if task in _REGRESSION_TASKS else "classification"
        return FoundationTabEstimator(**merged)
    cls = _load_class(class_path)
    merged = {**candidate.default_params, **(params or {})}
    try:
        estimator = cls(**merged)
    except TypeError as exc:
        msg = f"{candidate.key}: estimator parametreleri geçersiz ({merged}): {exc}"
        raise EstimatorBuildError(msg) from exc

    if candidate.wrap or candidate.scale:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if candidate.scale:
            from sklearn.preprocessing import StandardScaler

            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)
    return estimator
