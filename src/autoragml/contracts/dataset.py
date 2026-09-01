"""Dataset — `io/` çıktısı. Alanlar DONDU (ADR 0009).

`fingerprint` STRICT: kanonik form üzerinden tüm hücreler, tek streaming geçiş.
Kanonik TS formatı `long`; `wide` → auto-melt. `relations` v1'de REZERVE (`None`).
`handle` canlı veri referansı (DataFrame / pyarrow dataset) — serialize edilmez.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import Layout, Materialization, SourceKind


class DataSource(Contract):
    """Verinin geldiği yer."""

    kind: SourceKind
    ref: str | Path | None = None  # yol / bağlantı adı; DataFrame'de None


class DatasetShape(Contract):
    """Boyut. `n_rows` her zaman tam sayım (lazy'de bile — ADR 0009/5)."""

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)


class Dataset(Contract):
    """Yüklenmiş veri kümesi + kimlik damgası."""

    source: DataSource
    schema_: dict[str, str] = Field(alias="schema")  # kolon -> dtype str
    shape: DatasetShape
    materialization: Materialization
    layout: Layout = Layout.NA
    fingerprint: str  # STRICT SHA256
    fingerprint_spec: str
    fingerprint_fast: str | None = None  # structural — hızlı drift sinyali, kimlik değil
    modparts: dict[str, str] = Field(default_factory=dict)  # v1: {"tabular": ...}
    relations: None = None  # REZERVE (ADR 0009/3)

    # Canlı veri referansı — serialize edilmez, dump'ta atlanır.
    handle: Any = Field(default=None, exclude=True, repr=False)
