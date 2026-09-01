"""preprocessors.pipeline + target — uçtan uca (ADR 0011)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.io import load_dataset
from autoragml.preprocessors import FeaturePipeline, TargetTransform
from autoragml.preprocessors.catalog import PreprocessError, build_op
from tests.unit.preprocessors._util import ctx


def test_pipeline_from_plan_end_to_end() -> None:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "num": rng.normal(size=n),
            "cat": rng.choice(list("abc"), n),
            "const": np.ones(n),
            "when": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
        }
    )
    df.loc[:10, "num"] = np.nan
    cfg = resolve_run_config(target="y").config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    plan = build_plan(profile, task, cfg)

    pipe = FeaturePipeline.from_plan(plan)
    train, test = df.iloc[:320], df.iloc[320:]
    fitted, train_out = pipe.fit_transform(train, ctx())
    test_out = fitted.apply(test)

    assert "const" not in train_out.columns  # drop
    assert "cat" not in train_out.columns  # encode (onehot expand)
    assert "when" not in train_out.columns  # date_expand
    assert "when_month" in train_out.columns
    assert not train_out["num"].isna().any()  # impute
    assert list(train_out.columns) == list(test_out.columns)


def test_pipeline_candidate_choice_applied() -> None:
    n = 400
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"y": rng.normal(size=n), "skew": np.concatenate([np.zeros(n - 6), np.full(6, 1e5)])}
    )
    cfg = resolve_run_config(target="y").config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    plan = build_plan(profile, task, cfg)
    assert any(g.group_name == "heavy_tailed_numeric" for g in plan.candidate_ops)

    pipe = FeaturePipeline.from_plan(plan, candidate_choices={"heavy_tailed_numeric": "yeo_johnson"})
    _, out = pipe.fit_transform(df, ctx())
    # yeo-johnson + standardize → ortalama ~0, std ~1
    assert abs(float(out["skew"].mean())) < 0.2


def test_build_op_unknown_raises() -> None:
    with pytest.raises(PreprocessError, match="Bilinmeyen"):
        build_op("nonsense_op", ["a"], {})


@pytest.mark.parametrize("choice", ["none", "log1p", "yeo_johnson", "quantile"])
def test_target_transform_roundtrip(choice: str) -> None:
    rng = np.random.default_rng(2)
    y = np.abs(rng.gamma(2.0, 2.0, 500)).astype(np.float64)
    fitted = TargetTransform(choice).fit(y)
    recovered = fitted.inverse(fitted.forward(y))
    assert np.allclose(recovered, y, atol=1e-6)
