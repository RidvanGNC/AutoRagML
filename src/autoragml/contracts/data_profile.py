"""DataProfile + ColumnProfile + TimeSeriesProfile — `analyzers` çıktısı. DONDU (ADR 0010).

`ColumnProfile`: AutoGluon FeatureMetadata deseni (`raw_dtype` + `special_types` +
`semantic_role` + `flags`). `TimeSeriesProfile`: infer_freq + ACF/STL doğrulama +
per-series ADI/CV² intermittency **ölçümü** (router değil — ADR 0010).
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import (
    ClassificationScheme,
    ColumnFlag,
    InferenceSource,
    IntermittencyClass,
    PerSeriesDetail,
    RawDtype,
    SemanticRole,
    SpecialType,
)


class ColumnStats(Contract):
    """Kolon istatistikleri (tür-koşullu alanlar `None` olabilir)."""

    n_unique: int = Field(ge=0)
    missing_ratio: float = Field(ge=0.0, le=1.0)
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    zero_ratio: float | None = None
    top_values: dict[str, int] | None = None
    sample_values: list[str] = Field(default_factory=list)


class ColumnProfile(Contract):
    """Tek bir kolonun betimlemesi. Karar yok — yalnız ölçüm."""

    name: str
    raw_dtype: RawDtype
    special_types: set[SpecialType] = Field(default_factory=set)
    semantic_role: SemanticRole
    flags: set[ColumnFlag] = Field(default_factory=set)
    duplicate_of: str | None = None
    stats: ColumnStats
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    inference_source: InferenceSource = InferenceSource.RULE


class TargetSummary(Contract):
    """Hedef kolonunun görev-ilgili özeti."""

    n_classes: int | None = None
    class_balance: dict[str, float] | None = None
    zero_ratio: float | None = None
    distribution_note: str | None = None


class SeriesProfile(Contract):
    """Bir grubun (per-series) zaman serisi ölçümleri."""

    group: str
    n_obs: int = Field(ge=0)
    n_nonzero: int = Field(ge=0)
    zero_ratio: float = Field(ge=0.0, le=1.0)
    adi: float | None = None
    cv2: float | None = None
    history_weeks: int = Field(ge=0)
    intermittency_class: IntermittencyClass
    intermittency_class_recent: IntermittencyClass | None = None
    class_changed_over_time: bool = False


class SeasonalityHint(Contract):
    """ACF/STL ile doğrulanmış mevsimsel periyot."""

    period: int = Field(ge=2)
    strength: float = Field(ge=0.0, le=1.0)


class TimeSeriesProfile(Contract):
    """Zaman serisi tanısı. per-series intermittency = ölçüm, router değil (ADR 0010)."""

    freq: str | None = None
    freq_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    span: tuple[str, str] | None = None
    regular: bool = False
    gaps: dict[str, int] = Field(default_factory=dict)  # grup -> eksik dönem sayısı
    seasonality: list[SeasonalityHint] = Field(default_factory=list)
    trend_strength: float | None = None
    stationarity_pvalue: float | None = None
    per_series: list[SeriesProfile] = Field(default_factory=list)
    classification_scheme: ClassificationScheme = ClassificationScheme.SBC
    per_series_detail: PerSeriesDetail = PerSeriesDetail.FULL


class LeakageSuspect(Contract):
    """Yumuşak sızıntı şüphesi (WARNING — ADR 0011/5). `validators` sert kontrolü ayrı."""

    column: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class QualityFlag(Contract):
    """Dataset düzeyi kalite uyarısı."""

    code: str
    detail: str | None = None


class DataProfile(Contract):
    """Bir veri kümesinin tam betimlemesi. Model eğitmez, fit etmez."""

    columns: list[ColumnProfile]
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    target_profile: ColumnProfile
    target_summary: TargetSummary
    timeseries: TimeSeriesProfile | None = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    leakage_suspects: list[LeakageSuspect] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
