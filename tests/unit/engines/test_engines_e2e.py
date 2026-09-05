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
    # nöral/foundation adayları kuruluysa auto-modda havuza girer ve şampiyon olabilir;
    # bu e2e testleri klasik/reduction/recursive/segmented mantığını izole eder (nöral için ayrı dosyalar).
    cfg = resolve_run_config(
        target="y",
        overrides={"hpo_level": "none", "neural_enabled": "off", "foundation_enabled": "off", **over},
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


def test_timeseries_engine_reduction_only() -> None:
    df = _panel_df()
    ds, cfg, profile, task = _prep(df, time_col="ds", group_col="g", classical_forecasting=False)
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert result.engine_key == "timeseries_core"
    assert any("reduction" in m for m in result.messages)
    assert any("_lag_" in c for c in result.champion.metadata.feature_cols)  # reduction şampiyonu
    pred = result.champion.pipeline.predict(df)
    assert len(pred) == len(df)
    assert not np.isnan(pred).any()


def test_timeseries_reduction_predict_interval_per_group() -> None:
    """ADR 0044: reduction-forecasting şampiyonu (FittedModelPipeline/Ensemble) + per_group conformal."""
    df = _panel_df()
    ds, cfg, profile, task = _prep(
        df, time_col="ds", group_col="g", classical_forecasting=False,
        postprocess={"conformal": {"enabled": True, "coverage": 0.9, "per_group": True}},
    )
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    pipeline = result.champion.pipeline
    assert hasattr(pipeline, "predict_interval")
    point = pipeline.predict(df)
    lower, upper = pipeline.predict_interval(df)
    assert lower.shape == upper.shape == point.shape
    assert np.all(upper >= lower)
    assert np.all((point >= lower - 1e-6) & (point <= upper + 1e-6))


def _hierarchy_panel(n_periods: int = 30) -> pd.DataFrame:
    """2 eyalet × 2 bölge — globally-unique bottom kimliği `region` (ADR 0045)."""
    months = pd.date_range("2020-01-01", periods=n_periods, freq="MS")
    rng = np.random.default_rng(3)
    rows = []
    for state in ["A", "B"]:
        for zone in ["x", "y"]:
            region = f"{state}-{zone}"
            base = 60 + rng.normal(0, 5)
            for i, m in enumerate(months):
                y = max(0.0, base + 10 * np.sin(i / 12 * 6.28) + rng.normal(0, 2))
                rows.append({"state": state, "zone": zone, "region": region, "ds": m, "y": y})
    return pd.DataFrame(rows)


def test_timeseries_engine_hierarchical_reconciliation() -> None:
    """ADR 0045: hierarchy_cols → genişletilmiş panel + MinTrace(wls_struct) reconciliation."""
    df = _hierarchy_panel()
    ds, cfg, profile, task = _prep(
        df, time_col="ds", group_col="region", classical_forecasting=False,
        hierarchy_cols=["state", "zone"],
    )
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert any("hierarchical" in m for m in result.messages)
    assert type(result.champion.pipeline).__name__ == "FittedHierarchicalForecaster"

    future = pd.DataFrame({
        "region": ["A-x", "A-y", "B-x", "B-y"],
        "ds": [df["ds"].max() + pd.DateOffset(months=1)] * 4,
    })
    preds = result.champion.pipeline.predict(future)
    assert preds.shape == (4,)
    assert not np.isnan(preds).any()


def _monthly_panel() -> pd.DataFrame:
    """Küçük aylık panel — klasik CV hızlı (season 12, ~60 nokta)."""
    months = pd.date_range("2019-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {"g": g, "ds": m, "y": max(0.0, 80 + 20 * np.sin(i / 12 * 6.28) + rng.normal(0, 4))}
        for g in ("A", "B", "C", "D")
        for i, m in enumerate(months)
    )


def test_timeseries_engine_classical_competes() -> None:
    df = _monthly_panel()
    ds, cfg, profile, task = _prep(
        df, time_col="ds", group_col="g", split_policy={"horizon": 6}
    )  # classical açık (varsayılan)
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    keys = [r.model_key for r in result.scoreboard.rows]
    assert any(k in {"auto_ets", "auto_arima", "auto_theta", "mstl"} for k in keys)  # klasik yarıştı
    pred = result.champion.pipeline.predict(df)
    assert len(pred) == len(df)
    assert not np.isnan(pred).any()


def test_timeseries_engine_recursive_reduction() -> None:
    """ADR 0026 B: forecast_reduction=recursive → 1-adım model + recursive-h serving."""
    df = _monthly_panel()  # küçük panel — recursive-h CV maliyeti yüksek
    ds, cfg, profile, task = _prep(
        df,
        time_col="ds",
        group_col="g",
        classical_forecasting=False,
        forecast_reduction="recursive",
        split_policy={"horizon": 3, "n_folds": 2},
    )
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert any("recursive" in m for m in result.messages)
    assert result.champion.metadata.params.get("strategy") == "recursive"
    # ansambl recursive modda devre dışı
    assert all(r.model_key != "weighted_ensemble" for r in result.scoreboard.rows)
    pred = result.champion.pipeline.predict(df)
    assert len(pred) == len(df)
    tail = df.groupby("g", sort=False).cumcount(ascending=False) < 3
    assert not np.isnan(pred[tail.to_numpy()]).any()  # tahmin ufku dolu


def _mixed_intermittency_panel() -> pd.DataFrame:
    """Kesikli-baskın aylık panel (6/9 kesikli) — SBC segmentasyonu tetikler (ADR 0028)."""
    months = pd.date_range("2019-01-01", periods=72, freq="MS")
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for i in range(3):
        for j, m in enumerate(months):
            rows.append({"g": f"sm{i}", "ds": m, "y": max(0.0, 80 + 15 * np.sin(j / 12 * 6.28) + rng.normal(0, 4))})
    for i in range(6):
        for m in months:
            rows.append({"g": f"it{i}", "ds": m, "y": float(rng.integers(2, 9)) if rng.random() < 0.15 else 0.0})
    return pd.DataFrame(rows)


def test_timeseries_engine_segmented_champion() -> None:
    """ADR 0028: karışık panel → segment başına şampiyon + yönlendirmeli serving."""
    df = _mixed_intermittency_panel()
    ds, cfg, profile, task = _prep(
        df, time_col="ds", group_col="g", classical_forecasting=False,
        split_policy={"horizon": 6},
        dynamics={"structure": "per_group_champion", "segment_min_series": 3},
    )
    assert len(cfg.dynamics.recipes) == 0
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert result.champion.metadata.model_key == "segmented"
    segs = result.champion.metadata.adaptive_plan_summary["segments"]
    assert len(segs) >= 2
    assert any("segment" in m for m in result.messages)
    # birleşik scoreboard: sentetik "segmented" satırı + segment-prefiksli satırlar (ADR 0028)
    keys = {r.model_key for r in result.scoreboard.rows}
    assert "segmented" in keys
    assert any("::" in k for k in keys)
    pred = result.champion.pipeline.predict(df)
    assert len(pred) == len(df)
    assert not np.isnan(pred).any()


def test_timeseries_engine_per_series_champion() -> None:
    """ADR 0046: `structure="per_series_champion"` → her seri KENDİ tam nested-CV şampiyonu."""
    df = _panel_df()  # 3 seri (A/B/C)
    ds, cfg, profile, task = _prep(
        df, time_col="ds", group_col="g", classical_forecasting=False,
        dynamics={"structure": "per_series_champion"},
    )
    result = InProcessRunner().run(TimeSeriesCoreEngine(), ds, cfg, profile, task)
    assert result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert result.champion.metadata.model_key == "segmented"  # aynı serving mekanizması (ADR 0028)
    segs = result.champion.metadata.adaptive_plan_summary["segments"]
    assert len(segs) == 3  # her seri kendi segmenti
    assert result.champion.metadata.params["source"] == "per_series"
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
