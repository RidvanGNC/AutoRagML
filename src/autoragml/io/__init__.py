"""io — kaynak → `Dataset` (ADR 0009).

`load_dataset(src, config)`:
1. kaynağı çözümle (df/csv/parquet/dizin/DB) + boyut yokla
2. eager/lazy karar (`config.io.eager_max_bytes`, varsayılan 1 GiB)
3. eager: tam oku → wide→long melt → strict+fast fingerprint (tek geçiş)
   lazy: şema peek → wide-guard → chunk akışında fingerprint + tam satır sayımı
4. `Dataset` üret (v1: `modparts={"tabular": ...}`, `relations=None`)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoragml.contracts.dataset import Dataset, DatasetShape, DataSource
from autoragml.contracts.enums import Layout, Materialization
from autoragml.contracts.run_config import RunConfig
from autoragml.exceptions import DataLoadError
from autoragml.io.db import DbSource
from autoragml.io.fingerprint import compute_fingerprints, iter_frame_chunks
from autoragml.io.layout import determine_layout, guard_lazy_not_wide, normalize_layout
from autoragml.io.lazyframe import LazyFrame
from autoragml.io.schema import infer_schema
from autoragml.io.sources import ResolvedSource, resolve_source
from autoragml.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_EAGER_MAX_BYTES = 1_073_741_824  # 1 GiB

__all__ = ["DbSource", "LazyFrame", "load_dataset"]


def _eager_threshold(config: RunConfig) -> int:
    limit = config.io.eager_max_bytes
    return _DEFAULT_EAGER_MAX_BYTES if limit is None else int(limit)


def _load_eager(resolved: ResolvedSource, config: RunConfig) -> Dataset:
    frame = resolved.read_full()
    frame, layout, _time_col, _group_col = normalize_layout(
        frame,
        target=config.target,
        time_col=config.time_col,
        group_col=config.group_col,
    )
    if frame.empty:
        msg = "Yüklenen veri boş — modelleme yapılamaz."
        raise DataLoadError(msg)

    schema = infer_schema(frame)
    columns = sorted(schema)
    fp = compute_fingerprints(iter_frame_chunks(frame), columns=columns, dtypes=schema)

    return Dataset(
        source=DataSource(kind=resolved.kind, ref=resolved.ref),
        dtypes=schema,
        shape=DatasetShape(n_rows=len(frame), n_cols=frame.shape[1]),
        materialization=Materialization.EAGER,
        layout=layout,
        fingerprint=fp.strict,
        fingerprint_spec=fp.strict_spec,
        fingerprint_fast=fp.fast,
        modparts={"tabular": "inline"},
        handle=frame,
    )


def _load_lazy(resolved: ResolvedSource, config: RunConfig) -> Dataset:
    schema = resolved.peek_schema()
    guard_lazy_not_wide(
        schema,
        target=config.target,
        time_col=config.time_col,
        group_col=config.group_col,
    )
    columns = sorted(schema)
    fp = compute_fingerprints(
        (chunk for chunk in resolved.iter_chunks(200_000)),
        columns=columns,
        dtypes=schema,
    )
    if fp.n_rows == 0:
        msg = "Yüklenen veri boş — modelleme yapılamaz."
        raise DataLoadError(msg)

    layout = determine_layout(
        schema, target=config.target, time_col=config.time_col, group_col=config.group_col
    )
    source = DataSource(kind=resolved.kind, ref=resolved.ref)
    handle = LazyFrame(
        source=source,
        dtypes=schema,
        n_rows=fp.n_rows,
        chunk_reader=resolved.iter_chunks,
    )
    return Dataset(
        source=source,
        dtypes=schema,
        shape=DatasetShape(n_rows=fp.n_rows, n_cols=len(schema)),
        materialization=Materialization.LAZY,
        layout=layout if layout is not Layout.WIDE_CONVERTED else Layout.LONG,
        fingerprint=fp.strict,
        fingerprint_spec=fp.strict_spec,
        fingerprint_fast=fp.fast,
        modparts={"tabular": "lazy"},
        handle=handle,
    )


def load_dataset(
    src: pd.DataFrame | str | Path | DbSource,
    config: RunConfig,
) -> Dataset:
    """Bir kaynağı `Dataset`'e yükle — otomatik eager/lazy, strict fingerprint."""
    resolved = resolve_source(src)
    threshold = _eager_threshold(config)
    eager = resolved.size_bytes <= threshold
    logger.info(
        "Kaynak %s: ~%d byte, eşik %d -> %s",
        resolved.kind,
        resolved.size_bytes,
        threshold,
        "eager" if eager else "lazy",
    )
    return _load_eager(resolved, config) if eager else _load_lazy(resolved, config)
