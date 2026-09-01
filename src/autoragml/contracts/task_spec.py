"""TaskSpec — `analyzers` çıktısı. Alanlar DONDU (ADR 0010).

`task` düz enum (v1'de yedisi de). `time_col` forecasting'de zorunlu.
Düşük güven → `inference_warnings` (akış durmaz).
"""

from __future__ import annotations

from pydantic import Field, model_validator

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import Modality, Task


class TaskSpec(Contract):
    """Çıkarılmış (veya kullanıcı-verili) görev tanımı."""

    task: Task
    modality: Modality
    targets: list[str] = Field(min_length=1)
    horizon: int | None = Field(default=None, ge=1)
    group_col: str | None = None
    time_col: str | None = None
    quantiles: list[float] | None = None
    inference_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    inference_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _post_checks(self) -> TaskSpec:
        if self.task is Task.FORECASTING:
            if self.time_col is None:
                msg = "forecasting görevi için time_col zorunlu (ADR 0010)"
                raise ValueError(msg)
            if self.modality is not Modality.TIMESERIES:
                msg = "forecasting görevi timeseries modalitesi gerektirir"
                raise ValueError(msg)
        if self.task is Task.QUANTILE_REGRESSION and not self.quantiles:
            msg = "quantile_regression için quantiles zorunlu"
            raise ValueError(msg)
        return self
