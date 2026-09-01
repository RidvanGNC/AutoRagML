"""engines — uçtan uca orkestrasyon (ADR 0015)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.enums import EngineStatus
from autoragml.engines import (
    InProcessRunner,
    TabularCoreEngine,
    TimeSeriesCoreEngine,
    select_engine,
)
from autoragml.exceptions import EngineError
from autoragml.io import load_dataset


def _tabular_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 4))
    y = x @ np.array([1.5, -2.0, 0.5, 0.3]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(4)}, "cat": rng.choice(list("ab"), n)})


def _panel_df() -> pd.DataFrame:
    weeks = pd.date_range("2022-01-03", periods=160, freq="W-MON")
    rng = np.random.default_rng(1)
    rows = [
        {"g": g, "ds": wk, "y": max(0.0, 100 + 25 * np.sin(i / 52 * 6.28) + rng.normal(0, 6))}
        for g in ["A", "B", "C"]
        for i, wk in enumerate(weeks)
    ]
    return pd.DataFrame(rows)


def _prep(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(
        target="y", overrides={"hpo_level": "none", **over}
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    return ds, cfg, profile, task


def test_select_engine_by_modality() -> None:
    _, cfg, _, task = _prep(_tabular_df())
    assert isinstance(select_engine(task, cfg), TabularCoreEngine)
    _, fcfg, _, ftask = _prep(_panel_df(), time_col="ds", group_col="g")
    assert isinstance(select_engine(ftask, fcfg), TimeSeriesCoreEngine)


def test_select_engine_override() -> None:
    _, _, _, task = _prep(_tabular_df())
    cfg = resolve_run_config(target="y", overrides={"engines": {"key": "timeseries_core"}}).config
    assert isinstance(select_engine(task, cfg), TimeSeriesCoreEngine)
    bad = resolve_run_config(target="y", overrides={"engines": {"key": "nope"}}).config
    import pytest

    with pytest.raises(EngineError):
        select_engine(task, bad)


def test_tabular_engine_end_to_end() -> None:
    ds, cfg, profile, task = _prep(_tabular_df())
    result = InProcessRunner().run(TabularCoreEngine(), ds, cfg, profile, task)
    assert result.status is EngineStatus.SUCCESS
    assert result.engine_key == "tabular_core"
    assert result.scoreboard.n_candidates >= 3
    assert result.selection.champion.model_key
    assert result.champion.metadata.feature_cols
    pred = result.champion.pipeline.predict(_tabular_df(20))
    assert pred.shape == (20,)
    assert not np.isnan(pred).any()


def test_timeseries_engine_end_to_end_with_reduction() -> None:
    df = _panel_df()
    ds, cfg, profile, task = _prep(df, time_col="ds", group_col="g")
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert result.engine_key == "timeseries_core"
    assert result.adaptive_plan is not None
    assert any("reduction" in m for m in result.messages)
    # lag özellikleri şampiyon feature'larına girdi
    assert any("_lag_" in c for c in result.champion.metadata.feature_cols)
    # predict: tam geçmiş verilir, reduction FE yeniden uygulanır
    pred = result.champion.pipeline.predict(df)
    assert len(pred) == len(df)
    assert not np.isnan(pred).any()


def test_postprocess_embedded_in_champion_bundle() -> None:
    ds, cfg, profile, task = _prep(
        _tabular_df(), postprocess={"clip": {"lower": 5.0, "auto_nonneg": False}}
    )
    result = InProcessRunner().run(TabularCoreEngine(), ds, cfg, profile, task)
    assert result.champion.metadata.postprocess_summary["clip"]["lower"] == 5.0
    pred = result.champion.pipeline.predict(_tabular_df(50))
    assert (pred >= 5.0).all()


def test_runner_wraps_engine_failure() -> None:
    ds, cfg, profile, task = _prep(_tabular_df(50))

    class _BoomEngine:
        key = "boom"

        def run(self, *_a: object, **_k: object) -> object:
            raise EngineError("patladı")

    result = InProcessRunner().run(_BoomEngine(), ds, cfg, profile, task)  # type: ignore[arg-type]
    assert result.status is EngineStatus.FAILED
    assert result.engine_key == "boom"
    assert any("patladı" in m for m in result.messages)
