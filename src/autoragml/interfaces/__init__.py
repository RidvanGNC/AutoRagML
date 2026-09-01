"""interfaces — çekirdeğin tek dışa açık yüzü (ADR 0020).

`Orchestrator` (akış) · `AutoRagML` (Python facade) · `cli.main` (`autoragml run`).
Üçü de aynı `Orchestrator.run`'ı çağırır. `agent_tools` v2 (iskele).
"""

from __future__ import annotations

from autoragml.interfaces.api import AutoRagML, LoadedChampion
from autoragml.interfaces.holdout import HoldoutSplit, score_holdout, split_holdout
from autoragml.interfaces.orchestrator import Orchestrator

__all__ = [
    "AutoRagML",
    "HoldoutSplit",
    "LoadedChampion",
    "Orchestrator",
    "score_holdout",
    "split_holdout",
]
