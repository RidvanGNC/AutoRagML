"""contracts — katmanlar arası tipli omurga.

HEPSİ DONDU (ADR 0008-0015). Alan tabloları: docs/architecture/01_contracts.md.
Bu paket saf veri sözleşmeleri içerir; iş mantığı yok.
"""

from __future__ import annotations

from autoragml.contracts import enums
from autoragml.contracts._base import Contract, FrozenContract
from autoragml.contracts.adaptive_plan import (
    AdaptivePlan,
    CandidateOpGroup,
    ColumnOp,
    RegimeDef,
)
from autoragml.contracts.analyzer_config import (
    AnalyzerConfig,
    ThresholdConfig,
    TimeSeriesAnalyzerConfig,
)
from autoragml.contracts.candidate import Candidate, SearchDim
from autoragml.contracts.config_resolution import ConfigResolution
from autoragml.contracts.data_profile import (
    ColumnProfile,
    ColumnStats,
    DataProfile,
    LeakageSuspect,
    QualityFlag,
    SeasonalityHint,
    SeriesProfile,
    TargetSummary,
    TimeSeriesProfile,
)
from autoragml.contracts.dataset import Dataset, DatasetShape, DataSource
from autoragml.contracts.dynamics_config import DynamicsConfig
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.ensemble_spec import EnsembleSpec
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.postprocess_config import (
    CalibrateConfig,
    ClipConfig,
    ConformalConfig,
    PostprocessConfig,
    RoundConfig,
)
from autoragml.contracts.promotion_config import PromotionConfig
from autoragml.contracts.run_config import (
    BudgetConfig,
    EnsembleConfig,
    GuardrailConfig,
    IOConfig,
    LLMConfig,
    RunConfig,
    SplitPolicy,
    TrackingConfig,
)
from autoragml.contracts.run_manifest import (
    DataSnapshot,
    EnvInfo,
    RunManifest,
    TimelineEntry,
)
from autoragml.contracts.run_result import RunResult
from autoragml.contracts.scoreboard import (
    ChampionInfo,
    ComparisonTests,
    PromotionResult,
    ScoreBoard,
    ScoreRow,
    SelectionResult,
)
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.tuning import Trial, TuningResult
from autoragml.contracts.validation import (
    FoldReport,
    LeakageReport,
    LeakageViolation,
    ValidationReport,
)
from autoragml.contracts.validation_config import ValidationConfig

__all__ = [
    "AdaptivePlan",
    "AnalyzerConfig",
    "BudgetConfig",
    "BundleMetadata",
    "CalibrateConfig",
    "Candidate",
    "CandidateOpGroup",
    "ChampionInfo",
    "ClipConfig",
    "ColumnOp",
    "ColumnProfile",
    "ColumnStats",
    "ComparisonTests",
    "ConfigResolution",
    "ConformalConfig",
    "Contract",
    "DataProfile",
    "DataSnapshot",
    "DataSource",
    "Dataset",
    "DatasetShape",
    "DynamicsConfig",
    "EngineResult",
    "EnsembleConfig",
    "EnsembleSpec",
    "EnvInfo",
    "FoldReport",
    "FrozenContract",
    "GuardrailConfig",
    "IOConfig",
    "LLMConfig",
    "LeakageReport",
    "LeakageSuspect",
    "LeakageViolation",
    "ModelBundle",
    "PlanContext",
    "PostprocessConfig",
    "PromotionConfig",
    "PromotionResult",
    "QualityFlag",
    "RegimeDef",
    "RoundConfig",
    "RunConfig",
    "RunManifest",
    "RunResult",
    "ScoreBoard",
    "ScoreRow",
    "SearchDim",
    "SeasonalityHint",
    "SelectionResult",
    "SeriesProfile",
    "SplitPolicy",
    "TargetSummary",
    "TaskSpec",
    "ThresholdConfig",
    "TimeSeriesAnalyzerConfig",
    "TimeSeriesProfile",
    "TimelineEntry",
    "TrackingConfig",
    "Trial",
    "TuningResult",
    "ValidationConfig",
    "ValidationReport",
    "enums",
]
