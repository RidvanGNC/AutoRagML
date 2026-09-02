"""AdaptivePlan — `dynamics/planner` çıktısı. DONDU (ADR 0007 + 0010 + 0015).

Deklaratif, serialize edilebilir. Kod taşımaz; **referans** taşır.
`committed_ops` her zaman uygulanır; `candidate_ops` HPO arama uzayında seçilir.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from autoragml.contracts._base import Contract


class ColumnOp(Contract):
    """Bir kolona uygulanacak tek işlem. `op` katalog adı veya `recipe:<ad>`."""

    op: str  # "log1p" | "drop" | "date_expand" | "encode" | "recipe:acme_parser" ...
    column: str
    params: dict[str, object] = Field(default_factory=dict)


class CandidateOpGroup(Contract):
    """Bir muamele sınıfı için HPO'nun seçeceği op seçenekleri (ADR 0015)."""

    group_name: str  # ör. "heavy_tailed_numeric" | "target"
    columns: list[str]
    choices: list[str] = Field(min_length=1)  # ["none", "log1p", "yeo_johnson", "quantile"]
    default: str  # hpo_level=none / family_policy varsayılanı


class RegimeDef(Contract):
    """Senaryo/regime tanımı. Fit'i `validators` yönetir (fold-güvenli)."""

    name: str
    kind: str  # "trend" | "volatility" | "joint" | ...
    params: dict[str, object] = Field(default_factory=dict)


class AdaptivePlan(Contract):
    """Veriye-özel işleme planı. Karar üretir; fit etmez."""

    committed_ops: list[ColumnOp] = Field(default_factory=list)
    candidate_ops: list[CandidateOpGroup] = Field(default_factory=list)
    row_policies: list[str] = Field(default_factory=list)
    structure: Literal["pooled", "per_group_champion"] = "pooled"
    regimes: list[RegimeDef] = Field(default_factory=list)
    family_policy: dict[str, str] = Field(default_factory=dict)  # family -> op yoğunluğu
    recipes_used: list[str] = Field(default_factory=list)
    model_hints: dict[str, dict[str, Any]] = Field(default_factory=dict)  # aday/family -> param (ADR 0024)
