"""Analiz için çalışma DataFrame'i elde etme.

Eager kaynak → `handle` doğrudan. Lazy kaynak → `profiling_sample_rows`'a kadar
örneklem (ilk chunk'lar); istatistikler örneklemli olduğu için uyarı + düşük güven.
"""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.dataset import Dataset
from autoragml.contracts.enums import Materialization
from autoragml.contracts.run_config import RunConfig
from autoragml.io.lazyframe import LazyFrame
from autoragml.logging import get_logger

logger = get_logger(__name__)


def get_analysis_frame(dataset: Dataset, config: RunConfig) -> tuple[pd.DataFrame, bool]:
    """Döner: (frame, sampled). `sampled=True` ise istatistikler tam veri değil."""
    if dataset.materialization is Materialization.EAGER:
        handle = dataset.handle
        if not isinstance(handle, pd.DataFrame):  # pragma: no cover - sözleşme ihlali
            msg = "Eager Dataset.handle bir DataFrame olmalı"
            raise TypeError(msg)
        return handle, False

    if not isinstance(dataset.handle, LazyFrame):  # pragma: no cover
        msg = "Lazy Dataset.handle bir LazyFrame olmalı"
        raise TypeError(msg)

    limit = config.analyzers.profiling_sample_rows
    parts: list[pd.DataFrame] = []
    collected = 0
    for chunk in dataset.handle.iter_chunks(min(limit, 200_000)):
        parts.append(chunk)
        collected += len(chunk)
        if collected >= limit:
            break
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(dataset.dtypes))
    if collected > limit:
        frame = frame.iloc[:limit]
    sampled = collected < dataset.shape.n_rows
    if sampled:
        logger.warning(
            "Lazy kaynak: profil ilk %d satır örneklemi üzerinde (%d toplam). "
            "İstatistik güveni düşürüldü.",
            len(frame),
            dataset.shape.n_rows,
        )
    return frame, sampled
