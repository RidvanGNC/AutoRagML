"""LazyFrame — büyük kaynak için chunk-akışlı `Dataset.handle` (ADR 0009).

Eşik üstündeki kaynaklar RAM'e alınmaz; `iter_chunks()` ile parça parça okunur.
`to_pandas()` tam materialize eder (uyarı loglar).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from autoragml.contracts.dataset import DataSource
from autoragml.logging import get_logger

logger = get_logger(__name__)

ChunkReader = Callable[[int], Iterator[pd.DataFrame]]


class LazyFrame:
    """Chunk-akışlı veri kaynağı sarımı."""

    __slots__ = ("_chunk_reader", "dtypes", "n_rows", "source")

    def __init__(
        self,
        *,
        source: DataSource,
        dtypes: dict[str, str],
        n_rows: int,
        chunk_reader: ChunkReader,
    ) -> None:
        self.source = source
        self.dtypes = dtypes
        self.n_rows = n_rows
        self._chunk_reader = chunk_reader

    def iter_chunks(self, chunk_size: int = 200_000) -> Iterator[pd.DataFrame]:
        """Kaynağı sabit boyutlu parçalar hâlinde oku."""
        yield from self._chunk_reader(chunk_size)

    def to_pandas(self) -> pd.DataFrame:
        """Tüm kaynağı belleğe al (uyarı loglar — lazy amacına aykırı)."""
        logger.warning(
            "LazyFrame.to_pandas(): %d satırlık kaynak tamamen belleğe alınıyor.",
            self.n_rows,
        )
        parts = list(self.iter_chunks())
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(self.dtypes))

    def __repr__(self) -> str:
        return f"LazyFrame(kind={self.source.kind}, n_rows={self.n_rows}, n_cols={len(self.dtypes)})"
