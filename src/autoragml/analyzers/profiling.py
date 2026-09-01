"""Kolon profilleme — `ColumnProfile` üretimi (ADR 0010).

AutoGluon FeatureMetadata deseni: `raw_dtype` + `special_types` + `semantic_role` +
`flags`. **Karar yok, fit yok** — yalnız ölçüm. `leakage_suspect` bayrağı sonradan
`leakage.scan` tarafından eklenir.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api import types as pdt

from autoragml.contracts.analyzer_config import ThresholdConfig
from autoragml.contracts.data_profile import ColumnProfile, ColumnStats, TargetSummary
from autoragml.contracts.enums import (
    ColumnFlag,
    InferenceSource,
    RawDtype,
    SemanticRole,
    SpecialType,
    Task,
)
from autoragml.io.schema import is_datetime_like, is_numeric_like

_TOP_VALUES_CAP = 20
_SAMPLE_VALUES = 5
_BOOL_TOKENS = {"0", "1", "true", "false", "yes", "no", "t", "f"}


def _skew(arr: npt.NDArray[np.float64]) -> float:
    std = float(np.std(arr))
    if std == 0.0:
        return 0.0
    z = (arr - float(np.mean(arr))) / std
    return float(np.mean(z**3))


def _kurtosis(arr: npt.NDArray[np.float64]) -> float:
    std = float(np.std(arr))
    if std == 0.0:
        return 0.0
    z = (arr - float(np.mean(arr))) / std
    return float(np.mean(z**4) - 3.0)


def _raw_dtype(series: pd.Series) -> RawDtype:
    if pdt.is_bool_dtype(series):
        return RawDtype.BOOL
    if pdt.is_datetime64_any_dtype(series):
        return RawDtype.DATETIME
    if isinstance(series.dtype, pd.CategoricalDtype):
        return RawDtype.CATEGORY
    if pdt.is_integer_dtype(series):
        return RawDtype.INT
    if pdt.is_float_dtype(series):
        return RawDtype.FLOAT
    return RawDtype.OBJECT


def _is_text_like(series: pd.Series, min_tokens: float) -> bool:
    if not (pdt.is_object_dtype(series) or pdt.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return False
    token_counts = sample.str.split().map(lambda parts: len(parts) if isinstance(parts, list) else 0)
    return float(token_counts.mean()) >= min_tokens


def _looks_boolean(series: pd.Series) -> bool:
    if pdt.is_bool_dtype(series):
        return True
    vals = {str(v).strip().lower() for v in series.dropna().unique().tolist()}
    return 0 < len(vals) <= 2 and vals <= _BOOL_TOKENS


def _special_types(series: pd.Series, raw: RawDtype, min_tokens: float) -> set[SpecialType]:
    out: set[SpecialType] = set()
    if raw is RawDtype.DATETIME or (raw is RawDtype.OBJECT and is_datetime_like(series)):
        out.add(SpecialType.DATETIME)
    if _looks_boolean(series):
        out.add(SpecialType.BOOLEAN)
    if raw is RawDtype.OBJECT and SpecialType.DATETIME not in out and is_numeric_like(series):
        out.add(SpecialType.EMBEDDED_NUMBER)
    if _is_text_like(series, min_tokens):
        out.add(SpecialType.TEXT)
    return out


def _column_stats(series: pd.Series, raw: RawDtype) -> ColumnStats:
    n = len(series)
    n_unique = int(series.nunique(dropna=True))
    missing_ratio = float(series.isna().mean()) if n else 0.0
    stats = ColumnStats(n_unique=n_unique, missing_ratio=missing_ratio)

    numeric = pd.to_numeric(series, errors="coerce") if raw is not RawDtype.DATETIME else None
    if numeric is not None and numeric.notna().any():
        valid = numeric.dropna()
        arr: npt.NDArray[np.float64] = np.asarray(valid, dtype=np.float64)
        stats.min = float(np.min(arr))
        stats.max = float(np.max(arr))
        stats.mean = float(np.mean(arr))
        stats.std = float(np.std(arr))
        if arr.size >= 3:
            stats.skew = _skew(arr)
            stats.kurtosis = _kurtosis(arr)
        stats.zero_ratio = float(np.mean(arr == 0.0))
    else:
        vc = series.dropna().astype(str).value_counts().head(_TOP_VALUES_CAP)
        stats.top_values = {str(k): int(v) for k, v in vc.items()}

    stats.sample_values = [str(v) for v in series.dropna().unique()[:_SAMPLE_VALUES].tolist()]
    return stats


def _semantic_role(
    series: pd.Series,
    *,
    name: str,
    target: str,
    raw: RawDtype,
    special: set[SpecialType],
    stats: ColumnStats,
    n_rows: int,
    thr: ThresholdConfig,
) -> SemanticRole:
    if name == target:
        return SemanticRole.TARGET
    if stats.n_unique <= 1:
        return SemanticRole.CONSTANT
    if SpecialType.BOOLEAN in special:
        return SemanticRole.BOOLEAN
    if SpecialType.DATETIME in special:
        return SemanticRole.DATETIME

    uniq_ratio = stats.n_unique / n_rows if n_rows else 0.0
    is_monotonic = bool(series.dropna().is_monotonic_increasing or series.dropna().is_monotonic_decreasing)

    # Tamsayı ID: yüksek benzersizlik + (monoton veya isim kalıbı) + yeterli satır.
    if raw is RawDtype.INT and n_rows >= 20 and uniq_ratio >= thr.id_uniqueness_ratio and is_monotonic:
        return SemanticRole.ID

    if raw in {RawDtype.INT, RawDtype.FLOAT} or SpecialType.EMBEDDED_NUMBER in special:
        if raw is RawDtype.INT and stats.n_unique <= thr.max_classes_for_classification:
            return SemanticRole.NUMERIC_DISCRETE
        return SemanticRole.NUMERIC_CONTINUOUS

    if SpecialType.TEXT in special:
        return SemanticRole.TEXT

    # Metinsel ID: object/kategorik, neredeyse benzersiz, yeterli satır.
    if (
        raw in {RawDtype.OBJECT, RawDtype.CATEGORY}
        and n_rows >= 20
        and uniq_ratio >= thr.id_uniqueness_ratio
        and stats.missing_ratio < 0.5
    ):
        return SemanticRole.ID

    if raw in {RawDtype.OBJECT, RawDtype.CATEGORY}:
        return SemanticRole.CATEGORICAL
    return SemanticRole.UNKNOWN


def _flags(
    series: pd.Series,
    *,
    raw: RawDtype,
    special: set[SpecialType],
    role: SemanticRole,
    stats: ColumnStats,
    n_rows: int,
    thr: ThresholdConfig,
) -> set[ColumnFlag]:
    out: set[ColumnFlag] = set()
    if stats.missing_ratio >= 1.0:
        out.add(ColumnFlag.ALL_MISSING)
    elif stats.missing_ratio >= thr.high_missing_ratio:
        out.add(ColumnFlag.HIGH_MISSING)

    non_null = series.dropna()
    if not non_null.empty and stats.n_unique > 1:
        dominant = float(non_null.astype(str).value_counts(normalize=True).iloc[0])
        if dominant >= thr.near_constant_freq:
            out.add(ColumnFlag.NEAR_CONSTANT)

    if role in {SemanticRole.CATEGORICAL, SemanticRole.TEXT, SemanticRole.ID}:
        uniq_ratio = stats.n_unique / n_rows if n_rows else 0.0
        if uniq_ratio >= thr.high_cardinality_ratio or stats.n_unique >= thr.high_cardinality_abs:
            out.add(ColumnFlag.HIGH_CARDINALITY)

    if stats.skew is not None and abs(stats.skew) >= thr.skew_abs:
        out.add(ColumnFlag.SKEWED)
    if stats.kurtosis is not None and stats.kurtosis >= thr.heavy_tail_kurtosis:
        out.add(ColumnFlag.HEAVY_TAILED)
    if stats.zero_ratio is not None and stats.zero_ratio >= thr.zero_inflated_ratio:
        out.add(ColumnFlag.ZERO_INFLATED)

    if raw is RawDtype.OBJECT and SpecialType.DATETIME in special:
        out.add(ColumnFlag.DATETIME_LIKE_STRING)
    if raw is RawDtype.OBJECT and SpecialType.EMBEDDED_NUMBER in special:
        out.add(ColumnFlag.NUMERIC_LIKE_STRING)

    monotonic_dtypes = {RawDtype.INT, RawDtype.FLOAT, RawDtype.DATETIME}
    if (
        raw in monotonic_dtypes
        and stats.n_unique > 1
        and not non_null.empty
        and (non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing)
    ):
        out.add(ColumnFlag.MONOTONIC)
    return out


def build_column_profile(
    series: pd.Series,
    *,
    name: str,
    target: str,
    n_rows: int,
    thr: ThresholdConfig,
    sampled: bool,
) -> ColumnProfile:
    """Tek bir kolonun profilini üret."""
    raw = _raw_dtype(series)
    special = _special_types(series, raw, thr.text_min_avg_tokens)
    stats = _column_stats(series, raw)
    role = _semantic_role(
        series, name=name, target=target, raw=raw, special=special, stats=stats, n_rows=n_rows, thr=thr
    )
    flags = _flags(series, raw=raw, special=special, role=role, stats=stats, n_rows=n_rows, thr=thr)
    confidence = 0.75 if sampled else 1.0
    if role is SemanticRole.UNKNOWN:
        confidence = min(confidence, 0.5)
    return ColumnProfile(
        name=name,
        raw_dtype=raw,
        special_types=special,
        semantic_role=role,
        flags=flags,
        stats=stats,
        confidence=confidence,
        inference_source=InferenceSource.RULE,
    )


def build_column_profiles(
    frame: pd.DataFrame,
    *,
    target: str,
    thr: ThresholdConfig,
    sampled: bool,
) -> list[ColumnProfile]:
    """Tüm kolonlar için profil + birebir kopya (`duplicate_of`) tespiti."""
    n_rows = len(frame)
    profiles: list[ColumnProfile] = []
    seen: list[tuple[str, pd.Series]] = []
    for col in frame.columns:
        name = str(col)
        series = frame[col]
        profile = build_column_profile(
            series, name=name, target=target, n_rows=n_rows, thr=thr, sampled=sampled
        )
        for prev_name, prev_series in seen:
            if series.equals(prev_series):
                profile.duplicate_of = prev_name
                break
        profiles.append(profile)
        seen.append((name, series))
    return profiles


def build_target_summary(series: pd.Series, task_hint: Task | None) -> TargetSummary:
    """Hedef kolonunun görev-ilgili özeti."""
    numeric = pd.to_numeric(series, errors="coerce")
    n_unique = int(series.nunique(dropna=True))
    zero_ratio = float((numeric == 0.0).mean()) if numeric.notna().any() else None

    classification_like = task_hint in {
        Task.BINARY_CLASSIFICATION,
        Task.MULTICLASS_CLASSIFICATION,
        Task.MULTILABEL_CLASSIFICATION,
    } or (not pdt.is_float_dtype(series) and n_unique <= 20)

    class_balance: dict[str, float] | None = None
    n_classes: int | None = None
    if classification_like:
        vc = series.dropna().astype(str).value_counts(normalize=True)
        class_balance = {str(k): float(v) for k, v in vc.head(50).items()}
        n_classes = n_unique

    note = "sürekli" if pdt.is_float_dtype(series) and n_unique > 20 else "ayrık/kategorik"
    return TargetSummary(
        n_classes=n_classes,
        class_balance=class_balance,
        zero_ratio=zero_ratio,
        distribution_note=note,
    )
