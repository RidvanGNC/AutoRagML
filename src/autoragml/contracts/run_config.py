"""RunConfig — kullanıcı niyeti + akıllı varsayılanlar. Alanlar DONDU (ADR 0008).

**Sır taşımaz** — yalnız `*_env` adları. Sırları `config.settings.Settings` runtime'da
`.env`'den çözer (ADR 0008/4).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from autoragml.contracts._base import Contract
from autoragml.contracts.analyzer_config import AnalyzerConfig
from autoragml.contracts.enums import (
    HpoLevel,
    Modality,
    SelectionRule,
    SplitKind,
    Task,
    TrackingBackend,
)


class BudgetConfig(Contract):
    """Zaman/deneme bütçesi. Cömert varsayılan; sessiz kesme yok (ADR 0008/1)."""

    total_max_seconds: int | None = None
    per_model_max_seconds: int | None = None
    max_trials_per_model: int = Field(default=15, ge=1)
    min_trials_per_model: int = Field(default=3, ge=1)
    per_fold_timeout_seconds: int | None = None
    runtime_projection_warn_seconds: int = Field(default=7200, ge=0)

    @model_validator(mode="after")
    def _check_trial_bounds(self) -> BudgetConfig:
        if self.min_trials_per_model > self.max_trials_per_model:
            msg = "budget.min_trials_per_model, max_trials_per_model'i aşamaz"
            raise ValueError(msg)
        return self


class SplitPolicy(Contract):
    """Kısmi split politikası — verilen alan kazanır, gerisi `analyzers` seçer (ADR 0008/2)."""

    kind: SplitKind | None = None
    n_folds: int | None = Field(default=None, ge=2)
    horizon: int | None = Field(default=None, ge=1)
    step: int | None = Field(default=None, ge=1)
    min_train_periods: int | None = Field(default=None, ge=1)
    test_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    gap: int | None = Field(default=None, ge=0)


class IOConfig(Contract):
    """Yükleme davranışı (ADR 0009). Materialization otomatik; eşik override edilebilir."""

    eager_max_bytes: int | None = None
    recipe_paths: list[Path] = Field(default_factory=list)


class TrackingConfig(Contract):
    """Deney takibi (opsiyonel). Varsayılan jsonl (bağımlılıksız)."""

    backend: TrackingBackend = TrackingBackend.JSONL
    uri_env: str | None = None


class LLMConfig(Contract):
    """LLM sağlayıcı seçimi (v2). Sır yok — yalnız env adları (ADR 0005)."""

    provider: str
    model: str | None = None
    endpoint_env: str | None = None
    api_key_env: str | None = None


class GuardrailConfig(Contract):
    """Model seçim guardrail'leri (DemandSensing deseni, ADR 0014)."""

    enabled: bool = True
    smape_mean_max: float | None = None
    rmse_mean_max: float | None = None
    wmape_mean_max: float | None = None
    abs_bias_mean_max: float | None = None
    model_scenario_blocklist: dict[str, list[str]] = Field(default_factory=dict)


class RunConfig(Contract):
    """Bir AutoRagML koşumunun tüm yapılandırması. Serialize edilebilir; sır yok."""

    # --- zorunlu / kolon işaretleri ---
    target: str
    time_col: str | None = None
    group_col: str | None = None

    # --- ipuçları (analyzers doğrular) ---
    task_hint: Task | None = None
    modality_hint: Modality | None = None
    quantiles: list[float] | None = None

    # --- koşum kimliği ---
    project_name: str = "autoragml"
    output_dir: Path = Path("outputs")
    seed: int = 42
    autopilot: bool = False  # v1'de sabit false; v2 auto-mode girişi (ADR 0008/2)

    # --- alt yapılandırmalar ---
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    split_policy: SplitPolicy | None = None
    io: IOConfig = Field(default_factory=IOConfig)
    analyzers: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    llm: LLMConfig | None = None
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)

    # --- modelleme ---
    scenarios: list[str] = Field(default_factory=lambda: ["scenario_1"])
    hpo_level: HpoLevel = HpoLevel.LIGHT
    primary_metric: str | None = None
    metric_by_class: dict[str, str] | None = None
    selection_rule: SelectionRule = SelectionRule.ONE_STD_ERR
    engines: dict[str, object] | None = None
    model_catalog_override: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def _post_checks(self) -> RunConfig:
        if self.task_hint is Task.FORECASTING and self.time_col is None:
            msg = "task_hint=forecasting için time_col zorunlu (ADR 0008/3)"
            raise ValueError(msg)
        if self.autopilot:
            msg = "autopilot v1'de desteklenmiyor (v2 auto-mode)"
            raise ValueError(msg)
        if self.quantiles is not None and not all(0.0 < q < 1.0 for q in self.quantiles):
            msg = "quantiles değerleri (0, 1) aralığında olmalı"
            raise ValueError(msg)
        return self
