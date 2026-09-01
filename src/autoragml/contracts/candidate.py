"""Candidate — model kataloğundan (YAML) çözülmüş aday. DONDU (ADR 0012 + 0013).

`class_path` task ailesine göre estimator sınıfını verir. `search_space` HPO uzayı
(katalog YAML). `requires` kurulu değilse entry atlanır.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import CandidateSource, Modality, PredictKind, Task


class SearchDim(Contract):
    """Bir hiperparametrenin arama uzayı tanımı."""

    type: str  # "int" | "float" | "loguniform" | "categorical"
    low: float | None = None
    high: float | None = None
    choices: list[object] | None = None
    step: float | None = None


class Candidate(Contract):
    """Bir koşumda değerlendirilecek somut model adayı."""

    key: str
    name: str
    family: str  # "gbdt" | "linear" | "forest" | "baseline" | "statistical" | "intermittent" | ...
    class_path: str | dict[str, str]  # str VEYA {"regression": "...", "classification": "..."}
    modalities: list[Modality] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)
    predict_kind: list[PredictKind] = Field(default_factory=lambda: [PredictKind.POINT])
    default_params: dict[str, object] = Field(default_factory=dict)
    search_space: dict[str, SearchDim] = Field(default_factory=dict)
    fidelity: str | None = None  # multi-fidelity ekseni (ör. "n_estimators")
    supports_early_stopping: bool = False
    early_stopping_rounds: int | None = None
    requires: list[str] = Field(default_factory=list)
    wrap: bool = False  # imputer/scaler sarımı gerekli mi
    source: CandidateSource = CandidateSource.BUILTIN_CATALOG
