"""engines.timeseries.neural_ts — native neuralforecast yolu (ADR 0032).

neuralforecast kurulu OLMASA da geçer: katalog-atlama, kapı, is_neural_ts.
Kuruluysa (GPU) küçük panel e2e + bundle round-trip.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import Modality, Task
from autoragml.engines.timeseries.neural_ts import is_neural_ts, neural_ts_available
from autoragml.models import build_candidates, load_catalog

_HAS_NF = importlib.util.find_spec("neuralforecast") is not None


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides={"time_col": "ds", **over}).config


def test_is_neural_ts() -> None:
    c = Candidate(key="nhits", name="NHITS", family="neural_ts", class_path={"regression": "x.R"},
                  modalities=[Modality.TIMESERIES], tasks=[Task.FORECASTING])
    assert is_neural_ts(c)
    c2 = Candidate(key="auto_ets", name="ETS", family="statistical", class_path={"regression": "x.R"},
                   modalities=[Modality.TIMESERIES], tasks=[Task.FORECASTING])
    assert not is_neural_ts(c2)


def test_neural_ts_catalog_yaml_present() -> None:
    cat = load_catalog()
    assert {"nhits", "patchtst", "tft", "nbeatsx", "itransformer", "tsmixer", "tide"} <= set(cat)
    assert {"auto_nhits", "auto_patchtst", "auto_tft"} <= set(cat)
    assert cat["tide"]["family"] == "neural_ts"  # ADR 0043


@pytest.mark.skipif(_HAS_NF, reason="neuralforecast kurulu — 'yok' senaryosu")
def test_neural_ts_skipped_without_lib() -> None:
    assert neural_ts_available() is False
    keys = {c.key for c in build_candidates(_cfg())}
    assert "nhits" not in keys and "patchtst" not in keys


@pytest.mark.skipif(not _HAS_NF, reason="neuralforecast kurulu değil")
def test_neural_ts_resolved_when_installed() -> None:
    assert neural_ts_available() is True
    # sabit modeller neural_search=False iken çözülür; Auto* çözülmez
    keys = {c.key for c in build_candidates(_cfg(neural_search=False))}
    assert {"nhits", "patchtst"} <= keys


def _panel(n_series: int = 25, n_periods: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    months = pd.date_range("2018-01-01", periods=n_periods, freq="MS")
    rows = [
        {"unique_id": f"s{g}", "ds": m,
         "y": 50 + g + 15 * np.sin(i / 12 * 6.28) + rng.normal(0, 3)}
        for g in range(n_series)
        for i, m in enumerate(months)
    ]
    return pd.DataFrame(rows)


@pytest.mark.skipif(not _HAS_NF, reason="neuralforecast kurulu değil")
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_neural_ts_reports_and_bundle_roundtrip(tmp_path) -> None:
    """GPU'da (yoksa CPU) 1 model CV + refit + serving + bundle sidecar round-trip."""
    from autoragml.analyzers import analyze
    from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
    from autoragml.engines.timeseries.neural_ts import refit_neural_ts, run_neural_ts_reports
    from autoragml.io import load_dataset
    from autoragml.persistence.bundle import _NEURAL_TS_DIR, load_bundle, save_bundle

    df = _panel()
    cfg = resolve_run_config(
        target="y",
        overrides={"time_col": "ds", "group_col": "unique_id", "hpo_level": "none",
                   "split_policy": {"horizon": 6}, "neural_enabled": "on",
                   "neural_ts_min_series": 10},
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)

    nhits = next(c for c in build_candidates(cfg) if c.key == "nhits")
    nhits = nhits.model_copy(update={"default_params": {"max_steps": 8}})  # hızlı

    reports, _ = run_neural_ts_reports(df, profile, task, cfg, [nhits])
    assert reports and reports[0].candidate_key == "nhits"
    assert reports[0].oof is not None

    fc = refit_neural_ts(nhits, df, profile, task, cfg)
    # GELECEK frame: df sonrası 6 ay (nöral tahmin, fallback değil)
    last_ds = df["ds"].max()
    future_months = pd.date_range(last_ds, periods=7, freq="MS")[1:]
    future = pd.DataFrame([
        {"unique_id": u, "ds": m, "y": np.nan}
        for u in df["unique_id"].unique() for m in future_months
    ])
    preds = fc.predict(future)
    assert preds.shape == (len(future),)
    assert not np.isnan(preds).any()
    assert float(np.std(preds)) > 0  # gerçek forecast — sabit fallback değil

    bundle = ModelBundle(
        metadata=BundleMetadata(feature_cols=[], feature_set_hash="h", target_col="y",
                                model_key="nhits", params={"family": "neural_ts"}),
        pipeline=fc,
    )
    dest = tmp_path / "champion.joblib"
    save_bundle(bundle, dest)
    assert (tmp_path / _NEURAL_TS_DIR).is_dir()
    reloaded = load_bundle(dest)
    got = reloaded.pipeline.predict(future)
    assert np.allclose(got, preds, atol=1e-2)  # yeniden yüklenen model aynı forecast
