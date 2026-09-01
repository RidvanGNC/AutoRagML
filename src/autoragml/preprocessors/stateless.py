"""Parametre öğrenmeyen dönüşümler — drop / date_expand / log1p / hashing (ADR 0011).

`fit` train'e dokunmaz (deterministik); `apply` her partition'da aynı işlemi yapar.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from autoragml.contracts.plan_context import PlanContext
from autoragml.transform import BaseTransform, FittedTransform, StatelessFitted

_DATE_PARTS = (
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "quarter",
)
_DATE_FLAGS = ("is_month_start", "is_month_end", "is_quarter_start", "is_year_start")


class ColumnDropper(BaseTransform):
    """Verilen kolonları düşür."""

    name = "drop"

    def __init__(self, columns: list[str]) -> None:
        self._columns = list(columns)

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        cols = self._columns

        def _fn(df: pd.DataFrame) -> pd.DataFrame:
            present = [c for c in cols if c in df.columns]
            return df.drop(columns=present)

        return StatelessFitted(_fn, params={"transform": "drop", "columns": cols})


class DateExpander(BaseTransform):
    """Datetime kolonlarını takvim özelliklerine aç."""

    name = "date_expand"

    def __init__(self, columns: list[str], *, keep_original: bool = False) -> None:
        self._columns = list(columns)
        self._keep = keep_original

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        cols = self._columns
        keep = self._keep

        def _fn(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in cols:
                if col not in out.columns:
                    continue
                dt = pd.to_datetime(out[col], errors="coerce")
                for part in _DATE_PARTS:
                    out[f"{col}_{part}"] = getattr(dt.dt, part).astype("int64")
                for flag in _DATE_FLAGS:
                    out[f"{col}_{flag}"] = getattr(dt.dt, flag).astype("int8")
                out[f"{col}_weekofyear"] = dt.dt.isocalendar().week.astype("int64")
                if not keep:
                    out = out.drop(columns=[col])
            return out

        return StatelessFitted(
            _fn, params={"transform": "date_expand", "columns": cols, "keep_original": keep}
        )


class Log1pTransform(BaseTransform):
    """İşaretli log1p: `sign(x) * log1p(|x|)` — negatif değerlerde de güvenli."""

    name = "log1p"

    def __init__(self, columns: list[str]) -> None:
        self._columns = list(columns)

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        cols = self._columns

        def _fn(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in cols:
                if col not in out.columns:
                    continue
                x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
                out[col] = np.sign(x) * np.log1p(np.abs(x))
            return out

        return StatelessFitted(_fn, params={"transform": "log1p", "columns": cols})


class HashingEncoder(BaseTransform):
    """Kategorik → sabit hash kovası (CRC32, süreçler arası deterministik). Hedef kullanmaz."""

    name = "hashing"

    def __init__(self, columns: list[str], *, n_buckets: int = 256) -> None:
        self._columns = list(columns)
        self._n = n_buckets

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        cols = self._columns
        n = self._n

        def _bucket(value: object) -> int:
            return zlib.crc32(str(value).encode("utf-8")) % n

        def _fn(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in cols:
                if col not in out.columns:
                    continue
                out[col] = out[col].map(_bucket).astype("int64")
            return out

        return StatelessFitted(
            _fn, params={"transform": "hashing", "columns": cols, "n_buckets": n}
        )
