"""Sözleşme enum'ları — tek kaynak. Tümü StrEnum (JSON/YAML dostu).

Alan tabloları: docs/architecture/01_contracts.md · ADR 0008-0015.
"""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """Veri modalitesi. v1: tablo + zaman serisi (ADR 0002)."""

    TABULAR = "tabular"
    TIMESERIES = "timeseries"


class Task(StrEnum):
    """Öğrenme görevi. v1'de yedisi de desteklenir (ADR 0010)."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    QUANTILE_REGRESSION = "quantile_regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    FORECASTING = "forecasting"


class RawDtype(StrEnum):
    """Ham pandas dtype ailesi (AutoGluon FeatureMetadata deseni, ADR 0010)."""

    INT = "int"
    FLOAT = "float"
    CATEGORY = "category"
    OBJECT = "object"
    DATETIME = "datetime"
    BOOL = "bool"


class SpecialType(StrEnum):
    """Ham dtype'ın ötesinde anlam (0..n, ADR 0010)."""

    TEXT = "text"
    TEXT_NGRAM = "text_ngram"
    DATETIME = "datetime"
    EMBEDDED_NUMBER = "embedded_number"
    BOOLEAN = "boolean"


class SemanticRole(StrEnum):
    """Türetilmiş kolon rolü (ADR 0010)."""

    TARGET = "target"
    ID = "id"
    CATEGORICAL = "categorical"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class ColumnFlag(StrEnum):
    """Kolon özelliği bayrağı (çakışmaz değil, ADR 0010). `duplicate_of` ayrı alandır."""

    HIGH_CARDINALITY = "high_cardinality"
    NEAR_CONSTANT = "near_constant"
    HIGH_MISSING = "high_missing"
    ALL_MISSING = "all_missing"
    SKEWED = "skewed"
    HEAVY_TAILED = "heavy_tailed"
    ZERO_INFLATED = "zero_inflated"
    DATETIME_LIKE_STRING = "datetime_like_string"
    NUMERIC_LIKE_STRING = "numeric_like_string"
    MONOTONIC = "monotonic"
    LEAKAGE_SUSPECT = "leakage_suspect"


class InferenceSource(StrEnum):
    """Kolon/görev çıkarımının kaynağı (ADR 0010)."""

    RULE = "rule"
    USER_OVERRIDE = "user_override"
    ML_DETECTOR = "ml_detector"


class Materialization(StrEnum):
    """Veri belleğe alma kararı (ADR 0009, otomatik)."""

    EAGER = "eager"
    LAZY = "lazy"


class Layout(StrEnum):
    """Zaman serisi veri şekli (ADR 0009). Kanonik: long."""

    LONG = "long"
    WIDE_CONVERTED = "wide_converted"
    SINGLE_SERIES = "single_series"
    NA = "n/a"


class SourceKind(StrEnum):
    """Veri kaynağı türü (ADR 0009). DB opsiyonel."""

    DATAFRAME = "dataframe"
    CSV = "csv"
    PARQUET = "parquet"
    CSV_DIR = "csv_dir"
    PARQUET_DIR = "parquet_dir"
    DB = "db"


class SplitKind(StrEnum):
    """Doğrulama split stratejisi (ADR 0008/2)."""

    HOLDOUT = "holdout"
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    GROUP_KFOLD = "group_kfold"
    TIME_SERIES = "time_series"
    ROLLING_ORIGIN = "rolling_origin"
    FIXED_WINDOW = "fixed_window"


class TrackingBackend(StrEnum):
    """Deney takip backend'i (opsiyonel). Varsayılan jsonl."""

    NONE = "none"
    JSONL = "jsonl"
    MLFLOW = "mlflow"


class Provenance(StrEnum):
    """Veri frame'inin partition kimliği (ADR 0011). Sızıntı kontrolünün temeli."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    FULL = "full"


class PredictKind(StrEnum):
    """Modelin üretebildiği tahmin türü."""

    POINT = "point"
    PROBA = "proba"
    QUANTILE = "quantile"


class HpoLevel(StrEnum):
    """HPO yoğunluğu (ADR 0013). Varsayılan light."""

    NONE = "none"
    LIGHT = "light"
    THOROUGH = "thorough"


class HpoBackend(StrEnum):
    """HPO backend'i (ADR 0013)."""

    RANDOM_SEARCH = "random_search"
    OPTUNA = "optuna"
    FLAML = "flaml"


class SelectionRule(StrEnum):
    """Şampiyon seçim kuralı (ADR 0014). Varsayılan one_std_err."""

    BEST = "best"
    ONE_STD_ERR = "one_std_err"


class IntermittencyClass(StrEnum):
    """Talep süreksizlik sınıfı (SBC/KH, ADR 0010). Router değil, ipucu."""

    SMOOTH = "smooth"
    INTERMITTENT = "intermittent"
    ERRATIC = "erratic"
    LUMPY = "lumpy"
    INSUFFICIENT = "insufficient"


class ClassificationScheme(StrEnum):
    """Süreksizlik sınıflandırma şeması (ADR 0010)."""

    SBC = "sbc"
    KH = "kh"


class PerSeriesDetail(StrEnum):
    """Per-series TS profil ayrıntı düzeyi (ADR 0010)."""

    FULL = "full"
    SAMPLED = "sampled"
    SUMMARY_ONLY = "summary_only"


class LeakageCategory(StrEnum):
    """Sızıntı taksonomisi (ADR 0011/5). validators BLOCK eder."""

    OVERLAP = "overlap"
    PREPROCESSING = "preprocessing"
    MULTI_TEST = "multi_test"


class CandidateSource(StrEnum):
    """Model adayının kaynağı (ADR 0012)."""

    BUILTIN_CATALOG = "builtin_catalog"
    USER_CATALOG = "user_catalog"
    ENTRY_POINT = "entry_point"


class EngineStatus(StrEnum):
    """Engine koşum durumu (ADR 0015)."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class StageStatus(StrEnum):
    """RunManifest timeline aşama durumu (ADR 0015)."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
