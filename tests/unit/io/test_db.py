"""io.db — SQLAlchemy kaynağı (`[db]` extra, ADR 0009)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import SourceKind
from autoragml.io import DbSource, load_dataset
from autoragml.io.db import read_sql


def _sqlite_url(tmp_path: Path) -> str:
    db = tmp_path / "t.db"
    df = pd.DataFrame({"g": ["a", "b", "a"], "y": [1.0, 2.0, 3.0]})
    import sqlalchemy as sa

    engine = sa.create_engine(f"sqlite:///{db}")
    df.to_sql("sales", engine, index=False)
    engine.dispose()
    return f"sqlite:///{db}"


def test_read_sql_basic(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path)
    frame = read_sql(DbSource(url=url, query="SELECT * FROM sales"))
    assert list(frame.columns) == ["g", "y"]
    assert len(frame) == 3


def test_load_dataset_from_db(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path)
    cfg = resolve_run_config(target="y", overrides={"group_col": "g"}).config
    ds = load_dataset(DbSource(url=url, query="SELECT * FROM sales"), cfg)
    assert ds.source.kind is SourceKind.DB
    assert ds.source.ref == "db:sqlite"
    assert ds.shape.n_rows == 3
    assert len(ds.fingerprint) == 64
