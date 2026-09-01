"""EngineRunner protokolü — engine hangi süreçte koşar (ADR 0006 + 0015).

v1: `InProcessRunner`. `SubprocessRunner` (venv izolasyonu) / `ContainerRunner` v1.1+.
"""

from __future__ import annotations

from typing import Protocol

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.base import Engine


class EngineRunner(Protocol):
    """Bir engine'i çalıştırma stratejisi."""

    def run(
        self,
        engine: Engine,
        dataset: Dataset,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
    ) -> EngineResult: ...
