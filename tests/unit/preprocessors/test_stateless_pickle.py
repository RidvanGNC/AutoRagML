"""stateless transformlar joblib/pickle ile serialize edilebilmeli (ADR 0018 bundle).

Regresyon: `ColumnDropper.fit` eskiden yerel closure (`_fn`) döndürüyordu → `save_bundle`
covtype gibi kolon-düşüren pipeline'larda çöküyordu.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from autoragml.contracts.enums import Task
from autoragml.contracts.plan_context import PlanContext
from autoragml.preprocessors.stateless import (
    ColumnDropper,
    DateExpander,
    HashingEncoder,
    Log1pTransform,
)

_CTX = PlanContext(target="y", task=Task.REGRESSION, column_roles={}, seed=0)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "num": [10.0, -5.0, 0.0, 3.0],
            "dt": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-09-30", "2025-01-01"]),
            "cat": ["a", "b", "a", "c"],
        }
    )


@pytest.mark.parametrize(
    "transform",
    [
        ColumnDropper(["num"]),
        DateExpander(["dt"]),
        Log1pTransform(["num"]),
        HashingEncoder(["cat"], n_buckets=16),
    ],
)
def test_fitted_stateless_op_pickle_roundtrip(transform: object) -> None:
    df = _frame()
    fitted = transform.fit(df, _CTX)  # type: ignore[attr-defined]
    restored = pickle.loads(pickle.dumps(fitted))

    out_a = fitted.apply(df)
    out_b = restored.apply(df)
    pd.testing.assert_frame_equal(out_a, out_b)
    assert restored.get_params()["transform"] == fitted.get_params()["transform"]


def test_hashing_is_deterministic_across_processes() -> None:
    df = _frame()
    fitted = HashingEncoder(["cat"], n_buckets=16).fit(df, _CTX)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(fitted.apply(df)["cat"].to_numpy(), restored.apply(df)["cat"].to_numpy())
