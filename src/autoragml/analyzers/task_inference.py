"""Görev çıkarımı — `TaskSpec` (ADR 0010).

Kurallar hedef kolona bakar. `task_hint` her zaman kazanır; çelişkide uyarı.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from autoragml.contracts.analyzer_config import ThresholdConfig
from autoragml.contracts.data_profile import ColumnProfile
from autoragml.contracts.enums import Modality, RawDtype, SemanticRole, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec


def _infer_from_target(
    series: pd.Series,
    target_profile: ColumnProfile,
    thr: ThresholdConfig,
) -> tuple[Task, list[str]]:
    warnings: list[str] = []
    n_unique = target_profile.stats.n_unique
    raw = target_profile.raw_dtype

    if raw in {RawDtype.OBJECT, RawDtype.CATEGORY} and SemanticRole.TEXT in {target_profile.semantic_role}:
        warnings.append("Hedef metin gibi görünüyor — yanlış kolon olabilir.")
        return Task.MULTICLASS_CLASSIFICATION, warnings

    numeric = pd.to_numeric(series, errors="coerce")
    is_numeric = numeric.notna().mean() >= 0.99

    if not is_numeric:
        if n_unique == 2:
            return Task.BINARY_CLASSIFICATION, warnings
        if n_unique <= thr.max_classes_for_classification:
            return Task.MULTICLASS_CLASSIFICATION, warnings
        warnings.append(
            f"Hedef {n_unique} benzersiz kategori içeriyor (eşik {thr.max_classes_for_classification}) "
            "— ID/metin gibi görünüyor, yanlış kolon olabilir."
        )
        return Task.MULTICLASS_CLASSIFICATION, warnings

    if n_unique == 2:
        return Task.BINARY_CLASSIFICATION, warnings
    if pdt.is_integer_dtype(series) and n_unique <= thr.max_classes_for_classification:
        warnings.append(
            f"Hedef {n_unique} ayrık tamsayı değer içeriyor — multiclass sınıflandırma varsayıldı. "
            "Regresyon istiyorsanız task_hint=regression verin."
        )
        return Task.MULTICLASS_CLASSIFICATION, warnings
    return Task.REGRESSION, warnings


def infer_task(
    frame: pd.DataFrame,
    config: RunConfig,
    *,
    modality: Modality,
    target_profile: ColumnProfile,
) -> TaskSpec:
    """Görev tipini çıkar (veya `task_hint`'i doğrula)."""
    thr = config.analyzers.thresholds
    warnings: list[str] = []
    series = frame[config.target]

    inferred, inf_warnings = _infer_from_target(series, target_profile, thr)
    warnings.extend(inf_warnings)

    if config.quantiles is not None:
        inferred = Task.QUANTILE_REGRESSION

    if modality is Modality.TIMESERIES and config.task_hint is not Task.REGRESSION:
        task = Task.FORECASTING
        if config.task_hint not in (None, Task.FORECASTING):
            warnings.append(
                f"task_hint={config.task_hint} ama timeseries modalitesi → forecasting kullanılıyor."
            )
    elif config.task_hint is not None:
        task = config.task_hint
        if config.task_hint != inferred and inferred is not Task.FORECASTING:
            warnings.append(
                f"task_hint={config.task_hint} çıkarımla ({inferred}) çelişiyor — hint kullanılıyor."
            )
    else:
        task = inferred

    confidence = 1.0
    if warnings:
        confidence = 0.7

    horizon = None
    if task is Task.FORECASTING and config.split_policy is not None:
        horizon = config.split_policy.horizon

    return TaskSpec(
        task=task,
        modality=modality,
        targets=[config.target],
        horizon=horizon,
        group_col=config.group_col,
        time_col=config.time_col,
        quantiles=config.quantiles,
        inference_confidence=confidence,
        inference_warnings=warnings,
    )
