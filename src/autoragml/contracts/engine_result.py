"""EngineResult — bir engine'in orchestrator'a döndürdüğü sonuç. DONDU (ADR 0015)."""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.model_bundle import ModelBundle
from autoragml.contracts.scoreboard import ScoreBoard, SelectionResult
from autoragml.contracts.task_spec import TaskSpec


class EngineResult(Contract):
    """Tek bir engine koşumunun tüm çıktısı."""

    engine_key: str
    status: EngineStatus = EngineStatus.SUCCESS
    scoreboard: ScoreBoard
    selection: SelectionResult
    champion: ModelBundle
    data_profile: DataProfile
    task_spec: TaskSpec
    adaptive_plan: AdaptivePlan
    validation_reports_ref: str | None = None
    messages: list[str] = Field(default_factory=list)
