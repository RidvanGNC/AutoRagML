"""Sözleşme taban sınıfları.

`Contract` — mutable ama katı (bilinmeyen alan yasak, atamada doğrulama).
`FrozenContract` — immutable (ör. `PlanContext`, `RunConfig` snapshot'ları).

Her ikisi de `extra="forbid"`: sözleşme dışı alan sessizce yutulmaz.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Katı, mutable sözleşme tabanı."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=False,
        populate_by_name=True,
    )


class FrozenContract(BaseModel):
    """Katı, immutable sözleşme tabanı."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        arbitrary_types_allowed=False,
        populate_by_name=True,
    )
