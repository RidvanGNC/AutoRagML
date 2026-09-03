"""ScoreBoard + SelectionResult — `scoring` çıktısı. DONDU (ADR 0014).

Seçim yalnız OOF/validation'da. 1-SE kuralı varsayılan. Realized wall-clock + K,
`selection_bias_bound = σ·√(2 ln K)`. MCB / Diebold-Mariano opsiyonel (forecasting).
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import SelectionRule


class ScoreRow(Contract):
    """Bir (model, senaryo) kombinasyonunun OOF skoru + guardrail durumu."""

    model_key: str
    family: str = "ml"
    scenario: str = "scenario_1"
    oof_metric_mean: float
    oof_metric_se: float = Field(ge=0.0)
    all_metrics_mean: dict[str, float] = Field(default_factory=dict)
    guardrail_flags: list[str] = Field(default_factory=list)
    is_quarantined: bool = False
    selection_eligible: bool = True
    class_weighted_score: float | None = None
    realized_seconds: float = Field(default=0.0, ge=0.0)
    n_trials: int = Field(default=0, ge=0)
    n_folds: int = Field(default=0, ge=0)  # ADR 0035: aile-arası robustluk tie-break
    best_iteration: int | None = None


class ComparisonTests(Contract):
    """Çoklu karşılaştırma testleri (forecasting, opsiyonel — ADR 0014)."""

    mcb_ranks: dict[str, float] = Field(default_factory=dict)
    dm_pvalues: dict[str, float] = Field(default_factory=dict)


class ChampionInfo(Contract):
    """Seçilen şampiyon + gerekçe."""

    model_key: str
    scenario: str = "scenario_1"
    reason: str
    within_1se: list[str] = Field(default_factory=list)
    statistical_ties: list[str] = Field(default_factory=list)


class PromotionResult(Contract):
    """Mutlak eşik kapısı (DemandSensing `promotion_rules`)."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)


class ScoreBoard(Contract):
    """Tüm adayların sıralı skor tablosu."""

    rows: list[ScoreRow]
    primary_metric: str
    noise_floor: float = Field(ge=0.0)
    n_candidates: int = Field(ge=0)
    selection_bias_bound: float = Field(ge=0.0)  # σ·√(2 ln K)
    comparison_tests: ComparisonTests | None = None


class SelectionResult(Contract):
    """Şampiyon seçimi sonucu."""

    scoreboard: ScoreBoard
    selection_rule: SelectionRule = SelectionRule.ONE_STD_ERR
    champion: ChampionInfo
    promotion: PromotionResult
