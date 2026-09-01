"""Strict + fast fingerprint — tek streaming geçiş, sıra-bağımsız (ADR 0009).

**Strict:** kanonik şema (alfabetik kolon adları) + satır çoklu-kümesinin hash'i.
Satır çoklu-kümesi = her satır için `hash_pandas_object` → sıra-bağımsız birleşim
(`sum` mod 2^64, `xor`, `count`). Bir kopyanın satırları karıştırılsa bile aynı
fingerprint (modelleme için istenen semantik); tekrarlı satır sayılır; tek hücre
değişikliği yakalanır. Sıralama gerektirmez → gerçekten streaming.

**Fast (structural):** şema + boyut + kolon başına null sayısı + numerik toplam/min/max
+ non-numerik yaklaşık kardinalite. Yalnız hızlı drift sinyali — kimlik değil.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_U64 = 1 << 64
_NUNIQUE_CAP = 50_000


@dataclass
class FingerprintResult:
    """Bir veri kümesinin kimlik + structural damgası."""

    n_rows: int
    strict: str
    strict_spec: str
    fast: str


@dataclass
class _ColAccum:
    null_count: int = 0
    num_sum: float = 0.0
    num_min: float = float("inf")
    num_max: float = float("-inf")
    is_numeric: bool = True
    distinct: set[str] | None = field(default_factory=set)


class FingerprintAccumulator:
    """Chunk chunk beslenen sıra-bağımsız fingerprint biriktirici."""

    def __init__(self, columns: list[str]) -> None:
        self._columns = sorted(columns)
        self._n = 0
        self._sum = 0
        self._xor = 0
        self._cols: dict[str, _ColAccum] = {c: _ColAccum() for c in self._columns}

    def update(self, chunk: pd.DataFrame) -> None:
        if list(chunk.columns) != self._columns:
            chunk = chunk[self._columns]
        # Kategorik dtype'ın hash'i chunk'taki kategori kümesine bağlı → sıra-bağımsızlık
        # için değere göre hash'le.
        cat_cols = [c for c in chunk.columns if isinstance(chunk[c].dtype, pd.CategoricalDtype)]
        if cat_cols:
            chunk = chunk.astype({c: "object" for c in cat_cols})
        row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
        self._n += int(row_hashes.shape[0])
        if row_hashes.size:
            self._sum = (self._sum + int(row_hashes.sum(dtype=object))) % _U64
            self._xor ^= int(np.bitwise_xor.reduce(row_hashes))
        for col in self._columns:
            self._update_col(col, chunk[col])

    def _update_col(self, col: str, series: pd.Series) -> None:
        acc = self._cols[col]
        acc.null_count += int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        if acc.is_numeric and numeric.notna().sum() == series.notna().sum() and series.notna().any():
            valid = numeric.dropna()
            if not valid.empty:
                acc.num_sum += float(valid.sum())
                acc.num_min = min(acc.num_min, float(valid.min()))
                acc.num_max = max(acc.num_max, float(valid.max()))
        else:
            acc.is_numeric = False
        if acc.distinct is not None:
            for value in series.dropna().astype(str).unique().tolist():
                acc.distinct.add(value)
                if len(acc.distinct) > _NUNIQUE_CAP:
                    acc.distinct = None
                    break

    def _schema_repr(self, dtypes: dict[str, str]) -> str:
        return json.dumps({c: dtypes[c] for c in self._columns}, sort_keys=True, ensure_ascii=True)

    def finalize(self, dtypes: dict[str, str], *, pandas_version: str) -> FingerprintResult:
        schema_repr = self._schema_repr(dtypes)
        strict_payload = (
            f"{schema_repr}\x00n={self._n}\x00sum={self._sum}\x00xor={self._xor}"
        ).encode()
        strict = hashlib.sha256(strict_payload).hexdigest()
        strict_spec = (
            "strict/multiset-v1: sha256(sorted-schema || n || sum(hash_pandas_object) mod 2^64 "
            f"|| xor); row_order=ignored; pandas={pandas_version}"
        )

        fast_parts: list[str] = [schema_repr, f"n={self._n}"]
        for col in self._columns:
            acc = self._cols[col]
            if acc.is_numeric and acc.num_min != float("inf"):
                fast_parts.append(
                    f"{col}:num:nulls={acc.null_count}:sum={acc.num_sum:.6g}"
                    f":min={acc.num_min:.6g}:max={acc.num_max:.6g}"
                )
            else:
                nunique = "capped" if acc.distinct is None else str(len(acc.distinct))
                fast_parts.append(f"{col}:cat:nulls={acc.null_count}:nunique={nunique}")
        fast = hashlib.sha256("\x00".join(fast_parts).encode()).hexdigest()
        return FingerprintResult(n_rows=self._n, strict=strict, strict_spec=strict_spec, fast=fast)


def iter_frame_chunks(frame: pd.DataFrame, chunk_size: int = 200_000) -> Iterator[pd.DataFrame]:
    """Bir DataFrame'i sabit boyutlu parçalara böl (eager kaynak için tek geçiş)."""
    if len(frame) <= chunk_size:
        yield frame
        return
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start : start + chunk_size]


def compute_fingerprints(
    chunks: Iterable[pd.DataFrame],
    *,
    columns: list[str],
    dtypes: dict[str, str],
) -> FingerprintResult:
    """Chunk akışından tek geçişte strict + fast fingerprint üret."""
    acc = FingerprintAccumulator(columns)
    for chunk in chunks:
        acc.update(chunk)
    return acc.finalize(dtypes, pandas_version=pd.__version__)
