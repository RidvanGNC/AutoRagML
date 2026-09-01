"""TabularCoreEngine — tablo görevleri için çekirdek engine (ADR 0004 + 0015).

Reduction FE yok (zaman ekseni yok); doğrudan ortak akış.
"""

from __future__ import annotations

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.core import run_core_pipeline
from autoragml.io import materialize_frame
from autoragml.validators import Tuner


class TabularCoreEngine:
    """sklearn/lightgbm/xgboost + baseline'lar; pooled; kendi CV/guardrail/champion döngüsü."""

    key = "tabular_core"

    def run(
        self,
        dataset: Dataset,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
        *,
        tuner: Tuner | None = None,
    ) -> EngineResult:
        frame = materialize_frame(dataset)
        return run_core_pipeline(self.key, frame, profile, task, config, tuner=tuner)
