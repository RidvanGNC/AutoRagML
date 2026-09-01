"""dynamics.planner — AdaptivePlan üretimi (ADR 0007 + 0010 + 0015)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.dynamics.recipes import RecipeError
from autoragml.io import load_dataset


def _plan(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(target="y", overrides=over or None).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    return build_plan(profile, task, cfg), profile, task


def _committed(plan, col: str) -> list[str]:
    return [op.op for op in plan.committed_ops if op.column == col]


def test_structural_drops() -> None:
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "const": np.ones(n),
            "a": rng.normal(size=n),
            "a_dup": None,
            "row_id": np.arange(n),
            "note": ["some free text here about things"] * n,
        }
    )
    df["a_dup"] = df["a"]
    plan, _, _ = _plan(df)
    assert "drop" in _committed(plan, "const")
    assert "drop" in _committed(plan, "a_dup")
    assert "drop" in _committed(plan, "row_id")
    assert "drop" in _committed(plan, "note")


def test_categorical_encoding_by_cardinality() -> None:
    n = 300
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "low_card": rng.choice(list("abc"), n),
            "high_card": [f"k{i % 250}" for i in range(n)],
        }
    )
    plan, _, _ = _plan(df)
    low_ops = [op for op in plan.committed_ops if op.column == "low_card"]
    high_ops = [op for op in plan.committed_ops if op.column == "high_card"]
    assert low_ops and low_ops[0].params["strategy"] == "onehot"
    assert high_ops and high_ops[0].params["strategy"] == "target_encode"


def test_missing_gets_impute() -> None:
    n = 200
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    x[:20] = np.nan
    df = pd.DataFrame({"y": rng.normal(size=n), "x": x})
    plan, _, _ = _plan(df)
    assert "impute" in _committed(plan, "x")


def test_skewed_numeric_candidate_op() -> None:
    n = 400
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "skew_pos": np.concatenate([np.zeros(n - 8), np.full(8, 1e5)]),
        }
    )
    plan, _, _ = _plan(df)
    groups = {g.group_name for g in plan.candidate_ops}
    assert "heavy_tailed_numeric" in groups


def test_target_transform_candidate_positive_only() -> None:
    n = 400
    rng = np.random.default_rng(4)
    skewed_pos = np.concatenate([rng.exponential(1.0, n - 5), np.full(5, 500.0)])
    df = pd.DataFrame({"y": skewed_pos, "x": rng.normal(size=n)})
    plan, _, _ = _plan(df)
    tgt = next((g for g in plan.candidate_ops if g.group_name == "target"), None)
    assert tgt is not None
    assert "log1p" in tgt.choices  # hedef >= 0


def test_forecasting_per_group_champion() -> None:
    weeks = pd.date_range("2024-01-01", periods=120, freq="W-MON")
    rng = np.random.default_rng(5)
    rows = [
        {"g": g, "ds": wk, "y": 100 + 10 * np.sin(i / 52 * 6.28) + rng.normal(0, 3)}
        for g in ["A", "B", "C", "D"]
        for i, wk in enumerate(weeks)
    ]
    plan, _, task = _plan(pd.DataFrame(rows), time_col="ds", group_col="g")
    assert plan.structure == "per_group_champion"
    assert plan.family_policy["gbdt"] == "minimal"


def test_forecasting_pooled_when_history_below_threshold() -> None:
    weeks = pd.date_range("2026-01-05", periods=8, freq="W-MON")
    rows = [{"g": g, "ds": wk, "y": float(i)} for g in ["A", "B"] for i, wk in enumerate(weeks)]
    # horizon 8 → per_group eşiği 2*8=16 > 8 gözlem → pooled
    plan, _, _ = _plan(pd.DataFrame(rows), time_col="ds", group_col="g", split_policy={"horizon": 8})
    assert plan.structure == "pooled"


def test_scenario_2_regimes() -> None:
    weeks = pd.date_range("2024-01-01", periods=100, freq="W-MON")
    rng = np.random.default_rng(6)
    rows = [{"g": "A", "ds": wk, "y": 50 + rng.normal(0, 5)} for wk in weeks]
    plan, _, _ = _plan(pd.DataFrame(rows), time_col="ds", group_col="g", scenarios=["scenario_1", "scenario_2"])
    assert {r.name for r in plan.regimes} == {"trend_regime", "volatility_regime", "joint_regime"}


def test_unknown_recipe_fails_at_plan_time() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
    with pytest.raises(RecipeError):
        _plan(df, dynamics={"recipes": ["not_registered"]})
