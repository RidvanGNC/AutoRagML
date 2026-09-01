"""io.load_dataset — uçtan uca yükleme, eager/lazy, layout, fingerprint (ADR 0009)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import Layout, Materialization, SourceKind
from autoragml.exceptions import DataLoadError
from autoragml.io import LazyFrame, load_dataset


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides=over or None).config


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "g": rng.choice(["a", "b", "c"], n),
            "ds": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
            "y": rng.normal(size=n),
        }
    )


def test_dataframe_source_eager() -> None:
    ds = load_dataset(_frame(), _cfg(time_col="ds", group_col="g"))
    assert ds.materialization is Materialization.EAGER
    assert ds.source.kind is SourceKind.DATAFRAME
    assert ds.shape.n_rows == 40
    assert ds.layout is Layout.LONG
    assert isinstance(ds.handle, pd.DataFrame)
    assert len(ds.fingerprint) == 64
    assert ds.relations is None


def test_csv_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    _frame().to_csv(p, index=False)
    ds = load_dataset(p, _cfg(time_col="ds", group_col="g"))
    assert ds.source.kind is SourceKind.CSV
    assert ds.shape.n_rows == 40


def test_parquet_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    _frame().to_parquet(p)
    ds = load_dataset(p, _cfg(time_col="ds", group_col="g"))
    assert ds.source.kind is SourceKind.PARQUET


def test_parquet_dir(tmp_path: Path) -> None:
    _frame(20).to_parquet(tmp_path / "part1.parquet")
    _frame(15).to_parquet(tmp_path / "part2.parquet")
    ds = load_dataset(tmp_path, _cfg(time_col="ds", group_col="g"))
    assert ds.source.kind is SourceKind.PARQUET_DIR
    assert ds.shape.n_rows == 35


def test_eager_lazy_same_fingerprint(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    _frame(200).to_parquet(p)
    eager = load_dataset(p, _cfg(time_col="ds", group_col="g", io={"eager_max_bytes": 10**9}))
    lazy = load_dataset(p, _cfg(time_col="ds", group_col="g", io={"eager_max_bytes": 1}))
    assert lazy.materialization is Materialization.LAZY
    assert isinstance(lazy.handle, LazyFrame)
    assert eager.fingerprint == lazy.fingerprint
    assert eager.shape.n_rows == lazy.shape.n_rows == 200


def test_wide_dataframe_melted() -> None:
    wide = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-01", "2026-01-08"]),
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
            "C": [5.0, 6.0],
        }
    )
    ds = load_dataset(wide, _cfg())
    assert ds.layout is Layout.WIDE_CONVERTED
    assert ds.shape.n_rows == 6


def test_empty_raises() -> None:
    with pytest.raises(DataLoadError, match="boş"):
        load_dataset(pd.DataFrame({"y": []}), _cfg())


def test_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(DataLoadError, match="uzantı"):
        load_dataset(p, _cfg())
