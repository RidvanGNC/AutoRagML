"""AnalyzerConfig — `analyzers` katmanı eşikleri (ADR 0010).

`RunConfig.analyzers` altında taşınır. Tüm eşiklerin ADR 0010'daki varsayılanları.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import ClassificationScheme, PerSeriesDetail


class ThresholdConfig(Contract):
    """Kolon/görev çıkarımı eşikleri (ADR 0010)."""

    max_classes_for_classification: int = Field(default=20, ge=2)
    high_cardinality_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    high_cardinality_abs: int = Field(default=1000, ge=1)
    skew_abs: float = Field(default=1.0, ge=0.0)
    heavy_tail_kurtosis: float = Field(default=10.0, ge=0.0)
    near_constant_freq: float = Field(default=0.99, gt=0.0, le=1.0)
    high_missing_ratio: float = Field(default=0.4, gt=0.0, le=1.0)
    zero_inflated_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    id_uniqueness_ratio: float = Field(default=0.98, gt=0.0, le=1.0)
    text_min_avg_tokens: float = Field(default=3.0, gt=0.0)
    leakage_corr: float = Field(default=0.995, gt=0.0, le=1.0)
    tiny_data_rows: int = Field(default=50, ge=1)
    severe_imbalance_ratio: float = Field(default=0.02, gt=0.0, lt=1.0)


class TimeSeriesAnalyzerConfig(Contract):
    """Zaman serisi tanısı ayarları (ADR 0010)."""

    per_series_detail: PerSeriesDetail = PerSeriesDetail.FULL
    classification_scheme: ClassificationScheme = ClassificationScheme.SBC
    adi_threshold: float = Field(default=1.32, gt=0.0)
    cv2_threshold: float = Field(default=0.49, gt=0.0)
    recent_window_periods: int = Field(default=26, ge=4)
    min_history_periods: int = Field(default=12, ge=1)
    min_non_zero_points: int = Field(default=2, ge=1)
    max_seasonality_period: int = Field(default=366, ge=2)


class AnalyzerConfig(Contract):
    """`analyzers` katmanının tüm ayarları."""

    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    timeseries: TimeSeriesAnalyzerConfig = Field(default_factory=TimeSeriesAnalyzerConfig)
    profiling_sample_rows: int = Field(default=500_000, ge=1000)
