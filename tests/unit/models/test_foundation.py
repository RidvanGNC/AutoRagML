"""models.foundation_tab + foundation_gate + engines.timeseries.foundation_ts (ADR 0033).

tabpfn / chronos kurulu OLMASA da geçer: katalog, kapı bandı, is_foundation_ts, token kapısı.
Kuruluysa (GPU) e2e testler skipif ile açılır.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import Modality, Task
from autoragml.engines.timeseries.foundation_ts import foundation_ts_available, is_foundation_ts
from autoragml.io import load_dataset
from autoragml.models import build_candidates, load_catalog
from autoragml.models.foundation_gate import prepare_foundation_candidates

_HAS_TABPFN = importlib.util.find_spec("tabpfn") is not None
_HAS_CHRONOS = importlib.util.find_spec("chronos") is not None
_HAS_TIMESFM = importlib.util.find_spec("timesfm") is not None


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides={"time_col": "ds", **over}).config


def _tab_cand(key: str = "tabpfn") -> Candidate:
    return Candidate(
        key=key, name=key, family="foundation", class_path="__foundation_tab__",
        modalities=[Modality.TABULAR], tasks=[Task.REGRESSION],
        default_params={"random_state": 42, "n_estimators": 4},
    )


def _ts_cand(key: str, size: str) -> Candidate:
    return Candidate(
        key=key, name=key, family="foundation_ts", class_path="__foundation_ts__",
        modalities=[Modality.TIMESERIES], tasks=[Task.FORECASTING],
        default_params={"checkpoint": f"amazon/{key}", "size": size},
    )


def _tab_profile(n_rows: int = 200, n_cols: int = 6):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"x{i}" for i in range(n_cols)])
    df["y"] = df["x0"] * 2 + rng.normal(0, 0.1, n_rows)
    cfg = resolve_run_config(target="y").config
    return analyze(load_dataset(df, cfg), cfg)


def _panel_profile(n_series: int = 30, n_periods: int = 60):
    rng = np.random.default_rng(1)
    months = pd.date_range("2018-01-01", periods=n_periods, freq="MS")
    df = pd.DataFrame([
        {"unique_id": f"s{g}", "ds": m, "y": 40 + g + 10 * np.sin(i / 12 * 6.28) + rng.normal(0, 2)}
        for g in range(n_series) for i, m in enumerate(months)
    ])
    cfg = _cfg(group_col="unique_id")
    return analyze(load_dataset(df, cfg), cfg)


# --- katalog ---

def test_foundation_catalog_present() -> None:
    cat = load_catalog()
    assert "tabpfn" in cat
    assert cat["tabpfn"].get("enabled") is False  # KULLANICI KARARI: lisans-kapılı → varsayılan kapalı
    assert "tabicl" in cat and cat["tabicl"]["default_params"]["backend"] == "tabicl"  # ADR 0040 — auth'suz
    assert {"chronos_bolt", "chronos_bolt_small", "timesfm_2p5"} <= set(cat)  # ADR 0041: TimesFM
    assert cat["chronos_2"].get("enabled") is False  # opsiyonel
    assert cat["chronos_bolt"]["default_params"]["backend"] == "chronos"
    assert cat["timesfm_2p5"]["default_params"]["backend"] == "timesfm"


def test_is_foundation_ts() -> None:
    assert is_foundation_ts(_ts_cand("chronos_bolt", "base"))
    assert not is_foundation_ts(_tab_cand())


@pytest.mark.skipif(_HAS_TABPFN, reason="tabpfn kurulu — 'yok' senaryosu")
def test_tabpfn_skipped_without_lib() -> None:
    assert "tabpfn" not in {c.key for c in build_candidates(_cfg(foundation_enabled="on"))}


def test_tabicl_gate_no_token_needed(monkeypatch) -> None:
    """ADR 0040: TabICL auth'suz — TabPFN token yokluğu onu düşürmez."""
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    monkeypatch.setattr("autoragml.models.foundation_tab.ensure_tabpfn_token", lambda _e: False)
    monkeypatch.setattr("autoragml.models.foundation_tab.tabpfn_weights_cached", lambda: False)
    prof, task = _tab_profile(n_rows=300)
    tabicl = Candidate(
        key="tabicl", name="tabicl", family="foundation", class_path="__foundation_tab__",
        modalities=[Modality.TABULAR], tasks=[Task.REGRESSION],
        default_params={"backend": "tabicl", "random_state": 42},
    )
    out = prepare_foundation_candidates(
        [tabicl, _tab_cand()], prof, task, _cfg(foundation_enabled="on", foundation_tab_max_rows=1000)
    )
    keys = [c.key for c in out]
    assert "tabicl" in keys      # token gerekmez
    assert "tabpfn" not in keys  # token yok → düşer


@pytest.mark.skipif(_HAS_CHRONOS, reason="chronos kurulu — 'yok' senaryosu")
def test_chronos_skipped_without_lib() -> None:
    assert foundation_ts_available() is False
    keys = {c.key for c in build_candidates(_cfg(foundation_enabled="on"))}
    assert "chronos_bolt" not in keys


# --- kapı ---

def test_gate_off_drops_all(monkeypatch) -> None:
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    prof, task = _tab_profile()
    cands = [_tab_cand(), _ts_cand("chronos_bolt", "base")]
    out = prepare_foundation_candidates(cands, prof, task, _cfg(foundation_enabled="off"))
    assert out == []


def test_gate_auto_needs_gpu(monkeypatch) -> None:
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: False)
    prof, task = _tab_profile()
    out = prepare_foundation_candidates([_tab_cand()], prof, task, _cfg(foundation_enabled="auto"))
    assert out == []


def test_gate_tabpfn_row_band(monkeypatch) -> None:
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    monkeypatch.setattr(
        "autoragml.models.foundation_tab.ensure_tabpfn_token", lambda _env: True
    )
    prof, task = _tab_profile(n_rows=300)
    # bant içi → kalır
    keep = prepare_foundation_candidates(
        [_tab_cand()], prof, task, _cfg(foundation_enabled="auto", foundation_tab_max_rows=1000)
    )
    assert [c.key for c in keep] == ["tabpfn"]
    assert keep[0].default_params["token_env"] == "TABPFN_TOKEN"
    # bant dışı → düşer
    drop = prepare_foundation_candidates(
        [_tab_cand()], prof, task, _cfg(foundation_enabled="auto", foundation_tab_max_rows=100)
    )
    assert drop == []


def test_gate_tabpfn_needs_token(monkeypatch) -> None:
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    monkeypatch.setattr("autoragml.models.foundation_tab.ensure_tabpfn_token", lambda _env: False)
    monkeypatch.setattr("autoragml.models.foundation_tab.tabpfn_weights_cached", lambda: False)
    prof, task = _tab_profile()
    out = prepare_foundation_candidates([_tab_cand()], prof, task, _cfg(foundation_enabled="on"))
    assert out == []


def test_gate_chronos_size_autoselect(monkeypatch) -> None:
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    prof, task = _panel_profile(n_series=30)  # < 50 → küçük panel → _small
    cands = [_ts_cand("chronos_bolt", "base"), _ts_cand("chronos_bolt_small", "small")]
    out = prepare_foundation_candidates(cands, prof, task, _cfg(foundation_enabled="on", group_col="unique_id"))
    assert [c.key for c in out] == ["chronos_bolt_small"]


def test_gate_chronos_min_series(monkeypatch) -> None:
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    prof, task = _panel_profile(n_series=30)
    out = prepare_foundation_candidates(
        [_ts_cand("chronos_bolt_small", "small")], prof, task,
        _cfg(foundation_enabled="on", group_col="unique_id", foundation_ts_min_series=100),
    )
    assert out == []


def test_gate_timesfm_not_size_gated(monkeypatch) -> None:
    """ADR 0041: `size` alanı olmayan backend (timesfm) küçük panelde boyut-kapısına takılmaz."""
    monkeypatch.setattr("autoragml.models.foundation_gate.has_cuda", lambda: True)
    prof, task = _panel_profile(n_series=25)  # küçük panel → chronos boyut-seçimi tetiklenir
    tf = Candidate(
        key="timesfm_2p5", name="timesfm_2p5", family="foundation_ts", class_path="__foundation_ts__",
        modalities=[Modality.TIMESERIES], tasks=[Task.FORECASTING],
        default_params={"backend": "timesfm", "checkpoint": "google/timesfm-2.5-200m-pytorch"},
    )
    out = prepare_foundation_candidates(
        [tf, _ts_cand("chronos_bolt", "base")], prof, task,
        _cfg(foundation_enabled="on", group_col="unique_id"),
    )
    keys = [c.key for c in out]
    assert "timesfm_2p5" in keys       # boyut-kapısına takılmadı
    assert "chronos_bolt" not in keys  # base → küçük panelde elendi


@pytest.mark.skipif(not _HAS_TIMESFM, reason="timesfm kurulu değil")
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_timesfm_reports_and_serving() -> None:
    """GPU'da (yoksa CPU) TimesFM CV + refit + serving — gerçek forecast."""
    from autoragml.analyzers import analyze
    from autoragml.engines.timeseries.foundation_ts import (
        refit_foundation_ts,
        run_foundation_ts_reports,
    )
    from autoragml.io import load_dataset, materialize_frame

    rng = np.random.default_rng(0)
    months = pd.date_range("2018-01-01", periods=64, freq="MS")
    df = pd.DataFrame([
        {"unique_id": f"s{g}", "ds": m, "y": 40 + g + 10 * np.sin(i / 12 * 6.28) + rng.normal(0, 2)}
        for g in range(15) for i, m in enumerate(months)
    ])
    cfg = resolve_run_config(
        target="y",
        overrides={"time_col": "ds", "group_col": "unique_id", "split_policy": {"horizon": 6},
                   "foundation_enabled": "on"},
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    tf = next(c for c in build_candidates(cfg) if c.key == "timesfm_2p5")

    reports, _ = run_foundation_ts_reports(frame, profile, task, cfg, [tf])
    assert reports and reports[0].candidate_key == "timesfm_2p5"

    fc = refit_foundation_ts(tf, df, profile, task, cfg)
    last = df["ds"].max()
    fut = pd.date_range(last, periods=7, freq="MS")[1:]
    future = pd.DataFrame([{"unique_id": u, "ds": m, "y": np.nan}
                           for u in df["unique_id"].unique() for m in fut])
    preds = fc.predict(future)
    assert preds.shape == (len(future),) and np.isfinite(preds).all()
    assert float(np.std(preds)) > 0  # gerçek forecast, fallback değil
