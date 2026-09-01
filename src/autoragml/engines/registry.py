"""Engine seçimi — modalite → engine (ADR 0015).

`RunConfig.engines` override edebilir; yoksa `TaskSpec.modality` belirler.
"""

from __future__ import annotations

from autoragml.contracts.enums import Modality
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.base import Engine
from autoragml.engines.tabular.core_engine import TabularCoreEngine
from autoragml.engines.timeseries.core_engine import TimeSeriesCoreEngine
from autoragml.exceptions import EngineError

_BUILTIN: dict[str, type[Engine]] = {
    TabularCoreEngine.key: TabularCoreEngine,
    TimeSeriesCoreEngine.key: TimeSeriesCoreEngine,
}


def select_engine(task: TaskSpec, config: RunConfig) -> Engine:
    """Görev modalitesine (veya config override'ına) göre engine örneği."""
    override = (config.engines or {}).get("key") if config.engines else None
    if isinstance(override, str):
        if override not in _BUILTIN:
            msg = f"Bilinmeyen engine: {override!r}. Mevcut: {sorted(_BUILTIN)}"
            raise EngineError(msg)
        return _BUILTIN[override]()

    if task.modality is Modality.TIMESERIES:
        return TimeSeriesCoreEngine()
    return TabularCoreEngine()
