"""TuningResult — `fine_tuners` çıktısı. DONDU (ADR 0013).

HPO yalnız iç resample'da çalışır. `candidate_ops` seçimi arama uzayında.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import HpoBackend, HpoLevel


class Trial(Contract):
    """Tek bir HPO denemesi."""

    number: int = Field(ge=0)
    params: dict[str, object]
    value: float | None = None
    fidelity: float | None = None
    pruned: bool = False
    elapsed_seconds: float = Field(ge=0.0)


class TuningResult(Contract):
    """Bir aday için HPO çıktısı."""

    candidate_key: str
    best_params: dict[str, object]
    trials: list[Trial] = Field(default_factory=list)
    spent_budget: dict[str, float] = Field(default_factory=dict)  # {"trials": .., "seconds": ..}
    realized_seconds: float = Field(ge=0.0)
    early_stopped: bool = False
    best_iteration_per_fold: list[int] = Field(default_factory=list)
    fidelity_schedule: list[float] = Field(default_factory=list)
    backend: HpoBackend = HpoBackend.RANDOM_SEARCH
    hpo_level: HpoLevel = HpoLevel.LIGHT
