"""Opsiyonel veritabanı kaynağı — SQLAlchemy (`[db]` extra, ADR 0009).

DB okuması v1'de her zaman eager (sorgu çalıştırılır → DataFrame).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autoragml.exceptions import DataLoadError


@dataclass(frozen=True)
class DbSource:
    """Bir SQL sorgusu kaynağı. `url` SQLAlchemy bağlantı dizgesi."""

    url: str
    query: str


def read_sql(source: DbSource) -> pd.DataFrame:
    """DB sorgusunu DataFrame'e oku. SQLAlchemy kurulu değilse net hata."""
    try:
        import sqlalchemy as sa
    except ModuleNotFoundError as exc:  # pragma: no cover - extra yoksa
        msg = "DB kaynağı için `sqlalchemy` gerekli: pip install 'autoragml[db]'"
        raise DataLoadError(msg) from exc

    engine = sa.create_engine(source.url)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sa.text(source.query), conn)
    except Exception as exc:  # noqa: BLE001 - kaynak hatasını sarmalıyoruz
        msg = f"DB okuma hatası: {exc}"
        raise DataLoadError(msg) from exc
    finally:
        engine.dispose()
