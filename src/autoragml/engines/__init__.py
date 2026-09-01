"""engines — orkestrasyon (ADR 0015).

`select_engine(task, config)` → `TabularCoreEngine` | `TimeSeriesCoreEngine`.
Engine akışı: dynamics → models → validators (nested CV + tuner) → scoring →
şampiyon refit (tüm train) → `ModelBundle` → `EngineResult`.
`InProcessRunner` engine'i sarmalar (çökme → `status=FAILED`).
"""

from __future__ import annotations

from autoragml.engines.base import Engine
from autoragml.engines.champion import refit_champion
from autoragml.engines.core import run_core_pipeline
from autoragml.engines.model_pipeline import FittedModelPipeline
from autoragml.engines.registry import select_engine
from autoragml.engines.runners import EngineRunner, InProcessRunner
from autoragml.engines.tabular.core_engine import TabularCoreEngine
from autoragml.engines.timeseries.core_engine import TimeSeriesCoreEngine

__all__ = [
    "Engine",
    "EngineRunner",
    "FittedModelPipeline",
    "InProcessRunner",
    "TabularCoreEngine",
    "TimeSeriesCoreEngine",
    "refit_champion",
    "run_core_pipeline",
    "select_engine",
]
