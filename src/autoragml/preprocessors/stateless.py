"""Parametre öğrenmeyen dönüşümler — drop / date_expand / log1p / hashing (ADR 0011).

`fit` train'e dokunmaz (deterministik); `apply` her partition'da aynı işlemi yapar.
Fitted op'lar **modül düzeyi callable sınıflar** (`__slots__`) — joblib/pickle ile
serialize edilebilir (yerel closure DEĞİL — ADR 0018 bundle serialize).
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


# --- picklable op'lar (yerel closure yerine) --------------------------------


class _DropOp:
    __slots__ = ("columns",)

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns if c in df.columns]
        return df.drop(columns=present)


class _DateExpandOp:
    __slots__ = ("columns", "keep")

    def __init__(self, columns: list[str], *, keep: bool) -> None:
        self.columns = columns
        self.keep = keep

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            dt = pd.to_datetime(out[col], errors="coerce")
            for part in _DATE_PARTS:
                out[f"{col}_{part}"] = getattr(dt.dt, part).astype("int64")
            for flag in _DATE_FLAGS:
                out[f"{col}_{flag}"] = getattr(dt.dt, flag).astype("int8")
            out[f"{col}_weekofyear"] = dt.dt.isocalendar().week.astype("int64")
            if not self.keep:
                out = out.drop(columns=[col])
        return out


class _Log1pOp:
    __slots__ = ("columns",)

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            out[col] = np.sign(x) * np.log1p(np.abs(x))
        return out


class _HashingOp:
    __slots__ = ("columns", "n_buckets")

    def __init__(self, columns: list[str], *, n_buckets: int) -> None:
        self.columns = columns
        self.n_buckets = n_buckets

    def _bucket(self, value: object) -> int:
        return zlib.crc32(str(value).encode("utf-8")) % self.n_buckets

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            out[col] = out[col].map(self._bucket).astype("int64")
        return out


# --- Transform tanımları -------------------------------------------------


class ColumnDropper(BaseTransform):
    """Verilen kolonları düşür."""

    name = "drop"

    def __init__(self, columns: list[str]) -> None:
        self._columns = list(columns)

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        return StatelessFitted(
            _DropOp(self._columns), params={"transform": "drop", "columns": self._columns}
        )


class DateExpander(BaseTransform):
    """Datetime kolonlarını takvim özelliklerine aç."""

    name = "date_expand"

    def __init__(self, columns: list[str], *, keep_original: bool = False) -> None:
        self._columns = list(columns)
        self._keep = keep_original

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        return StatelessFitted(
            _DateExpandOp(self._columns, keep=self._keep),
            params={"transform": "date_expand", "columns": self._columns, "keep_original": self._keep},
        )


class Log1pTransform(BaseTransform):
    """İşaretli log1p: `sign(x) * log1p(|x|)` — negatif değerlerde de güvenli."""

    name = "log1p"

    def __init__(self, columns: list[str]) -> None:
        self._columns = list(columns)

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        return StatelessFitted(
            _Log1pOp(self._columns), params={"transform": "log1p", "columns": self._columns}
        )


class HashingEncoder(BaseTransform):
    """Kategorik → sabit hash kovası (CRC32, süreçler arası deterministik). Hedef kullanmaz."""

    name = "hashing"

    def __init__(self, columns: list[str], *, n_buckets: int = 256) -> None:
        self._columns = list(columns)
        self._n = n_buckets

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        return StatelessFitted(
            _HashingOp(self._columns, n_buckets=self._n),
            params={"transform": "hashing", "columns": self._columns, "n_buckets": self._n},
        )
