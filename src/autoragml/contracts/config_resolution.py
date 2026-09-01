"""ConfigResolution — `config.resolve_run_config` çıktısı. DONDU (ADR 0016).

Nihai `RunConfig` + hangi alanın hangi katmandan geldiği (`provenance`) +
uygulanan katman sırası. `RunManifest.config_snapshot` bunu taşır.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.run_config import RunConfig


class ConfigResolution(Contract):
    """Katmanlı merge sonucu + izlenebilirlik."""

    config: RunConfig
    provenance: dict[str, str] = Field(default_factory=dict)
    layers: list[str] = Field(default_factory=list)
