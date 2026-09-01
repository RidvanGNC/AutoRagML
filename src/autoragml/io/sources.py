"""Kaynak çözümleme — girdi türü tespiti + okuma stratejisi (ADR 0009).

Desteklenen: DataFrame · `.csv`/`.tsv` · `.parquet` · csv/parquet dizini · `DbSource`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from autoragml.contracts.enums import SourceKind
from autoragml.exceptions import DataLoadError
from autoragml.io.db import DbSource, read_sql
from autoragml.io.schema import infer_schema, is_datetime_like

_CSV_SUFFIXES = {".csv", ".tsv"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}
_PEEK_ROWS = 2000

ChunkIter = Callable[[int], Iterator[pd.DataFrame]]


@dataclass
class ResolvedSource:
    """Çözümlenmiş kaynak + okuma callable'ları."""

    kind: SourceKind
    ref: str | None
    size_bytes: int
    known_n_rows: int | None
    read_full: Callable[[], pd.DataFrame]
    peek_schema: Callable[[], dict[str, str]]
    iter_chunks: ChunkIter


def _df_memory_bytes(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(deep=True).sum())


def _resolve_dataframe(frame: pd.DataFrame) -> ResolvedSource:
    def _chunks(chunk_size: int) -> Iterator[pd.DataFrame]:
        if len(frame) <= chunk_size:
            yield frame
            return
        for start in range(0, len(frame), chunk_size):
            yield frame.iloc[start : start + chunk_size]

    return ResolvedSource(
        kind=SourceKind.DATAFRAME,
        ref=None,
        size_bytes=_df_memory_bytes(frame),
        known_n_rows=len(frame),
        read_full=lambda: frame,
        peek_schema=lambda: infer_schema(frame),
        iter_chunks=_chunks,
    )


def _csv_datetime_cols(path: Path, sep: str) -> list[str]:
    head = pd.read_csv(path, sep=sep, nrows=_PEEK_ROWS)
    return [str(c) for c in head.columns if is_datetime_like(head[c])]


def _resolve_csv_files(paths: list[Path], kind: SourceKind, ref: str) -> ResolvedSource:
    sep = "\t" if paths[0].suffix == ".tsv" else ","

    def _read_full() -> pd.DataFrame:
        parse_dates = _csv_datetime_cols(paths[0], sep)
        frames = [
            pd.read_csv(p, sep=sep, parse_dates=parse_dates or None, low_memory=False)
            for p in paths
        ]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def _peek() -> dict[str, str]:
        return infer_schema(pd.read_csv(paths[0], sep=sep, nrows=_PEEK_ROWS))

    def _chunks(chunk_size: int) -> Iterator[pd.DataFrame]:
        parse_dates = _csv_datetime_cols(paths[0], sep)
        for p in paths:
            reader = pd.read_csv(
                p, sep=sep, parse_dates=parse_dates or None, chunksize=chunk_size
            )
            yield from reader

    return ResolvedSource(
        kind=kind,
        ref=ref,
        size_bytes=sum(p.stat().st_size for p in paths),
        known_n_rows=None,
        read_full=_read_full,
        peek_schema=_peek,
        iter_chunks=_chunks,
    )


def _resolve_parquet_files(paths: list[Path], kind: SourceKind, ref: str) -> ResolvedSource:
    import pyarrow.parquet as pq

    def _read_full() -> pd.DataFrame:
        frames = [pd.read_parquet(p) for p in paths]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def _peek() -> dict[str, str]:
        return infer_schema(pd.read_parquet(paths[0]).head(_PEEK_ROWS))

    def _chunks(chunk_size: int) -> Iterator[pd.DataFrame]:
        for p in paths:
            parquet_file = pq.ParquetFile(p)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()

    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    return ResolvedSource(
        kind=kind,
        ref=ref,
        size_bytes=sum(p.stat().st_size for p in paths),
        known_n_rows=total_rows,
        read_full=_read_full,
        peek_schema=_peek,
        iter_chunks=_chunks,
    )


def _resolve_db(source: DbSource) -> ResolvedSource:
    cache: dict[str, pd.DataFrame] = {}

    def _full() -> pd.DataFrame:
        if "df" not in cache:
            cache["df"] = read_sql(source)
        return cache["df"]

    def _chunks(chunk_size: int) -> Iterator[pd.DataFrame]:
        frame = _full()
        for start in range(0, len(frame), max(1, chunk_size)):
            yield frame.iloc[start : start + chunk_size]

    return ResolvedSource(
        kind=SourceKind.DB,
        ref=f"db:{source.url.split('://', 1)[0]}",
        size_bytes=0,  # sorgu çalıştırılmadan bilinemez → eager
        known_n_rows=None,
        read_full=_full,
        peek_schema=lambda: infer_schema(_full().head(_PEEK_ROWS)),
        iter_chunks=_chunks,
    )


def _resolve_path(path: Path) -> ResolvedSource:
    if path.is_file():
        if path.suffix in _CSV_SUFFIXES:
            return _resolve_csv_files([path], SourceKind.CSV, str(path))
        if path.suffix in _PARQUET_SUFFIXES:
            return _resolve_parquet_files([path], SourceKind.PARQUET, str(path))
        msg = f"Desteklenmeyen dosya uzantısı: {path.suffix} ({path})"
        raise DataLoadError(msg)
    if path.is_dir():
        parquet = sorted(p for p in path.iterdir() if p.suffix in _PARQUET_SUFFIXES)
        if parquet:
            return _resolve_parquet_files(parquet, SourceKind.PARQUET_DIR, str(path))
        csv = sorted(p for p in path.iterdir() if p.suffix in _CSV_SUFFIXES)
        if csv:
            return _resolve_csv_files(csv, SourceKind.CSV_DIR, str(path))
        msg = f"Dizinde .parquet veya .csv bulunamadı: {path}"
        raise DataLoadError(msg)
    msg = f"Kaynak yolu bulunamadı: {path}"
    raise DataLoadError(msg)


def resolve_source(src: pd.DataFrame | str | Path | DbSource) -> ResolvedSource:
    """Girdiyi çözümlenmiş kaynağa çevir."""
    if isinstance(src, pd.DataFrame):
        return _resolve_dataframe(src)
    if isinstance(src, DbSource):
        return _resolve_db(src)
    if isinstance(src, (str, Path)):
        return _resolve_path(Path(src))
    msg = f"Desteklenmeyen kaynak türü: {type(src).__name__}"
    raise DataLoadError(msg)
