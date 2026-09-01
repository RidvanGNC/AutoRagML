"""preprocessors.stateless — drop / date_expand / log1p / hashing (ADR 0011)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.preprocessors.stateless import (
    ColumnDropper,
    DateExpander,
    HashingEncoder,
    Log1pTransform,
)
from tests.unit.preprocessors._util import ctx


def test_column_dropper() -> None:
    df = pd.DataFrame({"y": [1, 2], "a": [3, 4], "b": [5, 6]})
    fitted = ColumnDropper(["a"]).fit(df, ctx())
    out = fitted.apply(df)
    assert list(out.columns) == ["y", "b"]
    # eksik kolon → sessiz geç
    assert "a" not in fitted.apply(df.drop(columns=["a"])).columns


def test_date_expander() -> None:
    df = pd.DataFrame(
        {"y": [1.0, 2.0], "ds": pd.to_datetime(["2026-03-15", "2026-07-01"])}
    )
    fitted = DateExpander(["ds"]).fit(df, ctx())
    out = fitted.apply(df)
    assert "ds" not in out.columns
    assert out["ds_year"].tolist() == [2026, 2026]
    assert out["ds_month"].tolist() == [3, 7]
    assert out["ds_quarter"].tolist() == [1, 3]
    assert "ds_weekofyear" in out.columns
    assert out["ds_is_quarter_start"].tolist() == [0, 1]


def test_log1p_signed() -> None:
    df = pd.DataFrame({"x": [0.0, -3.0, 7.0]})
    out = Log1pTransform(["x"]).fit(df, ctx()).apply(df)
    assert np.isclose(out["x"][0], 0.0)
    assert out["x"][1] < 0  # işaret korunur
    assert np.isclose(out["x"][2], np.log1p(7.0))


def test_hashing_deterministic_across_instances() -> None:
    df = pd.DataFrame({"c": ["apple", "banana", "apple", "cherry"]})
    a = HashingEncoder(["c"], n_buckets=16).fit(df, ctx()).apply(df)["c"].tolist()
    b = HashingEncoder(["c"], n_buckets=16).fit(df, ctx()).apply(df)["c"].tolist()
    assert a == b
    assert a[0] == a[2]  # aynı kategori → aynı kova
    assert all(0 <= v < 16 for v in a)
