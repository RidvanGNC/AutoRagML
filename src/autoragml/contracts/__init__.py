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
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import (
    BudgetConfig,
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
    "Candidate",
    "CandidateOpGroup",
    "ChampionInfo",
    "ColumnOp",
    "ColumnProfile",
    "ColumnStats",
    "ComparisonTests",
    "ConfigResolution",
    "Contract",
    "DataProfile",
    "DataSnapshot",
    "DataSource",
    "Dataset",
    "DatasetShape",
    "DynamicsConfig",
    "EngineResult",
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
    "PromotionResult",
    "QualityFlag",
    "RegimeDef",
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
