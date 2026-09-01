"""Engine protokolü (ADR 0015).

`Engine.run(Dataset, RunConfig, DataProfile, TaskSpec) -> EngineResult`.
`EngineRunner` (runners/) hangi süreçte koştuğunu soyutlar (v1: InProcess).
"""

from __future__ import annotations

from typing import Protocol

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec


class Engine(Protocol):
    """Bir (modalite, task) ailesi için uçtan uca orkestrasyon."""

    key: str

    def run(
        self,
        dataset: Dataset,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
    ) -> EngineResult: ...
