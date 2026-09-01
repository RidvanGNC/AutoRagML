"""io.fingerprint — strict (sıra-bağımsız) + fast damgası (ADR 0009)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.io.fingerprint import compute_fingerprints, iter_frame_chunks
from autoragml.io.schema import infer_schema


def _fp(frame: pd.DataFrame) -> str:
    schema = infer_schema(frame)
    return compute_fingerprints(
        iter_frame_chunks(frame), columns=sorted(schema), dtypes=schema
    ).strict


def _base() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "g": ["a", "b", "a", "c", "b"],
            "t": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-08", "2026-01-01", "2026-01-08"]),
            "y": rng.integers(0, 100, 5).astype(float),
        }
    )


def test_row_order_does_not_change_strict() -> None:
    df = _base()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert _fp(df) == _fp(shuffled)


def test_column_order_does_not_change_strict() -> None:
    df = _base()
    assert _fp(df) == _fp(df[["y", "t", "g"]])


def test_single_cell_change_detected() -> None:
    df = _base()
    changed = df.copy()
    changed.loc[2, "y"] = changed.loc[2, "y"] + 1.0
    assert _fp(df) != _fp(changed)


def test_duplicate_row_changes_strict() -> None:
    df = _base()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert _fp(df) != _fp(dup)


def test_streaming_equals_single_pass() -> None:
    df = _base()
    schema = infer_schema(df)
    one = compute_fingerprints([df], columns=sorted(schema), dtypes=schema).strict
    streamed = compute_fingerprints(
        iter_frame_chunks(df, chunk_size=2), columns=sorted(schema), dtypes=schema
    ).strict
    assert one == streamed


def test_fast_fingerprint_reacts_to_schema() -> None:
    df = _base()
    schema = infer_schema(df)
    fast1 = compute_fingerprints([df], columns=sorted(schema), dtypes=schema).fast
    renamed = df.rename(columns={"y": "yy"})
    schema2 = infer_schema(renamed)
    fast2 = compute_fingerprints([renamed], columns=sorted(schema2), dtypes=schema2).fast
    assert fast1 != fast2


def test_n_rows_counted() -> None:
    df = _base()
    schema = infer_schema(df)
    res = compute_fingerprints(iter_frame_chunks(df, chunk_size=2), columns=sorted(schema), dtypes=schema)
    assert res.n_rows == 5
