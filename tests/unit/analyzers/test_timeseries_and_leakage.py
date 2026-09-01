"""analyzers.timeseries + leakage (ADR 0010 / 0011)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers.leakage import scan_leakage
from autoragml.analyzers.profiling import build_column_profiles
from autoragml.analyzers.timeseries import diagnose_timeseries
from autoragml.contracts.analyzer_config import ThresholdConfig, TimeSeriesAnalyzerConfig
from autoragml.contracts.enums import ColumnFlag, IntermittencyClass

TS = TimeSeriesAnalyzerConfig()
THR = ThresholdConfig()


def _seasonal_panel() -> pd.DataFrame:
    weeks = pd.date_range("2024-01-01", periods=120, freq="W-MON")
    rng = np.random.default_rng(0)
    rows = []
    for g in ["A", "B"]:
        for i, wk in enumerate(weeks):
            val = 200.0 + 40.0 * np.sin(i / 52 * 2 * np.pi) + rng.normal(0, 5)
            rows.append({"g": g, "ds": wk, "y": max(0.0, val)})
    return pd.DataFrame(rows)


def _mixed_panel() -> pd.DataFrame:
    weeks = pd.date_range("2025-01-06", periods=80, freq="W-MON")
    rng = np.random.default_rng(1)
    rows = []
    for i, wk in enumerate(weeks):
        rows.append({"g": "smooth", "ds": wk, "y": max(0.0, 100 + 10 * np.sin(i / 52 * 6.28) + rng.normal(0, 3))})
    for wk in weeks:
        rows.append({"g": "lumpy", "ds": wk, "y": float(rng.integers(0, 60)) if rng.random() < 0.15 else 0.0})
    return pd.DataFrame(rows)


def test_freq_and_seasonality_detected() -> None:
    profile, _ = diagnose_timeseries(
        _seasonal_panel(), target="y", time_col="ds", group_col="g", config=TS
    )
    assert profile.freq in {"W-MON", "W"}
    assert profile.freq_confidence > 0.9
    assert any(s.period == 52 for s in profile.seasonality)


def test_per_series_intermittency_classes() -> None:
    profile, _ = diagnose_timeseries(
        _mixed_panel(), target="y", time_col="ds", group_col="g", config=TS
    )
    by_group = {s.group: s.intermittency_class for s in profile.per_series}
    assert by_group["smooth"] is IntermittencyClass.SMOOTH
    assert by_group["lumpy"] in {
        IntermittencyClass.LUMPY,
        IntermittencyClass.INTERMITTENT,
        IntermittencyClass.ERRATIC,
    }
    assert profile.intermittency_summary


def test_insufficient_history() -> None:
    short = pd.DataFrame(
        {"g": ["a"] * 5, "ds": pd.date_range("2026-01-05", periods=5, freq="W-MON"), "y": [1.0, 0, 0, 2.0, 0]}
    )
    profile, _ = diagnose_timeseries(short, target="y", time_col="ds", group_col="g", config=TS)
    assert profile.per_series[0].intermittency_class is IntermittencyClass.INSUFFICIENT


def test_gaps_detected() -> None:
    ds = list(pd.date_range("2026-01-05", periods=10, freq="W-MON"))
    del ds[4:6]  # 2 hafta boşluk
    df = pd.DataFrame({"g": ["a"] * len(ds), "ds": ds, "y": [10.0] * len(ds)})
    profile, _ = diagnose_timeseries(df, target="y", time_col="ds", group_col="g", config=TS)
    assert profile.gaps.get("a", 0) == 2


def test_leakage_near_perfect_and_name() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    df = pd.DataFrame(
        {
            "y": y,
            "leak_copy": y * 2.0 + 1e-9,
            "final_result": y + rng.normal(0, 1e-4, 300),
            "clean": rng.normal(size=300),
        }
    )
    profiles = build_column_profiles(df, target="y", thr=THR, sampled=False)
    suspects = scan_leakage(df, columns=profiles, target="y", time_col=None, thr=THR)
    names = {s.column for s in suspects}
    assert "leak_copy" in names
    assert "final_result" in names
    assert "clean" not in names
    by_name = {p.name: p for p in profiles}
    assert ColumnFlag.LEAKAGE_SUSPECT in by_name["leak_copy"].flags


def test_leakage_future_dated() -> None:
    n = 50
    df = pd.DataFrame(
        {
            "y": np.arange(n, dtype=float),
            "ds": pd.date_range("2026-01-01", periods=n, freq="D"),
            "delivered_at": pd.date_range("2026-01-04", periods=n, freq="D"),
        }
    )
    profiles = build_column_profiles(df, target="y", thr=THR, sampled=False)
    suspects = scan_leakage(df, columns=profiles, target="y", time_col="ds", thr=THR)
    assert any("future_dated" in s.reason for s in suspects)
