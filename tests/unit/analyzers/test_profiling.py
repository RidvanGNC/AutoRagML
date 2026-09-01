"""analyzers.profiling — kolon rolleri, flag'ler, duplicate, target summary (ADR 0010)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers.profiling import build_column_profiles, build_target_summary
from autoragml.contracts.analyzer_config import ThresholdConfig
from autoragml.contracts.enums import ColumnFlag, RawDtype, SemanticRole, SpecialType

THR = ThresholdConfig()


def _profiles(df: pd.DataFrame, target: str = "y"):
    return {p.name: p for p in build_column_profiles(df, target=target, thr=THR, sampled=False)}


def test_basic_roles() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=200),
            "cont": rng.normal(size=200),
            "disc": rng.integers(0, 4, 200),
            "cat": rng.choice(["a", "b", "c"], 200),
            "when": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 100, 200), unit="D"),
            "row_id": np.arange(200),
            "const": np.ones(200),
        }
    )
    p = _profiles(df)
    assert p["y"].semantic_role is SemanticRole.TARGET
    assert p["cont"].semantic_role is SemanticRole.NUMERIC_CONTINUOUS
    assert p["disc"].semantic_role is SemanticRole.NUMERIC_DISCRETE
    assert p["cat"].semantic_role is SemanticRole.CATEGORICAL
    assert p["when"].semantic_role is SemanticRole.DATETIME
    assert p["row_id"].semantic_role is SemanticRole.ID
    assert p["const"].semantic_role is SemanticRole.CONSTANT
    assert p["cont"].raw_dtype is RawDtype.FLOAT


def test_flags() -> None:
    n = 500
    df = pd.DataFrame(
        {
            "y": np.arange(n, dtype=float),
            "skewed": np.concatenate([np.zeros(n - 5), np.array([1e6] * 5)]),
            "near_const": ["a"] * (n - 2) + ["b", "c"],
            "high_card": [f"id_{i}" for i in range(n)],
            "mostly_null": [1.0] + [np.nan] * (n - 1),
            "monotonic_up": np.arange(n, dtype=float),
        }
    )
    p = _profiles(df)
    assert ColumnFlag.SKEWED in p["skewed"].flags
    assert ColumnFlag.ZERO_INFLATED in p["skewed"].flags
    assert ColumnFlag.NEAR_CONSTANT in p["near_const"].flags
    assert ColumnFlag.HIGH_CARDINALITY in p["high_card"].flags
    assert ColumnFlag.HIGH_MISSING in p["mostly_null"].flags
    assert ColumnFlag.MONOTONIC in p["monotonic_up"].flags


def test_duplicate_column_detected() -> None:
    df = pd.DataFrame({"y": [1, 2, 3], "a": [10, 20, 30], "a_copy": [10, 20, 30]})
    p = _profiles(df)
    assert p["a_copy"].duplicate_of == "a"
    assert p["a"].duplicate_of is None


def test_datetime_like_string_and_embedded_number() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "date_str": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
            "num_str": ["1", "2", "3", "4"],
            "text": ["the quick brown fox", "jumps over lazy dog", "hello there world", "a b c d e"],
        }
    )
    p = _profiles(df)
    assert SpecialType.DATETIME in p["date_str"].special_types
    assert ColumnFlag.DATETIME_LIKE_STRING in p["date_str"].flags
    assert SpecialType.EMBEDDED_NUMBER in p["num_str"].special_types
    assert SpecialType.TEXT in p["text"].special_types
    assert p["text"].semantic_role is SemanticRole.TEXT


def test_target_summary_regression_vs_classification() -> None:
    reg = build_target_summary(pd.Series(np.random.default_rng(1).normal(size=100)), None)
    assert reg.n_classes is None
    clf = build_target_summary(pd.Series(["a", "b", "a", "c"] * 25), None)
    assert clf.n_classes == 3
    assert clf.class_balance is not None
