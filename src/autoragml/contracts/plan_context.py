"""PlanContext — `FittedTransform.fit` ikinci argümanı. FROZEN (ADR 0011 + 0015).

Salt-okunur. Test/full veriye, split nesnesine erişim **yok**. `provenance` her zaman
`"train"` — çağrı zamanında zorlanır.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoragml.contracts._base import FrozenContract
from autoragml.contracts.enums import SemanticRole, Task


class PlanContext(FrozenContract):
    """Bir dönüşümün fold içinde fit edilirken ihtiyaç duyduğu asgari bağlam."""

    target: str
    task: Task
    column_roles: dict[str, SemanticRole] = Field(default_factory=dict)
    group_col: str | None = None
    time_col: str | None = None
    fold_id: int | None = None
    train_span: tuple[str, str] | None = None  # ISO tarih sınırları (TS)
    seed: int = 42
    provenance: Literal["train"] = "train"
