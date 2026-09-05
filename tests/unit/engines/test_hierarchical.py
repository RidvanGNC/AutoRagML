"""engines.timeseries.hierarchical — MinTrace(wls_struct) reconciliation (ADR 0045)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoragml.engines.timeseries.hierarchical import (
    FittedHierarchicalForecaster,
    build_hierarchy,
    hierarchical_available,
    reconcile,
)


def _panel(n_periods: int = 12) -> pd.DataFrame:
    months = pd.date_range("2021-01-01", periods=n_periods, freq="MS")
    rng = np.random.default_rng(0)
    rows = []
    for state in ["A", "B"]:
        for zone in ["x", "y"]:
            region = f"{state}-{zone}"  # globally-unique bottom kimliği
            base = rng.normal(60, 5)
            for m in months:
                rows.append({"state": state, "zone": zone, "region": region, "ds": m, "y": base})
    return pd.DataFrame(rows)


def test_hierarchical_available() -> None:
    assert hierarchical_available() is True  # dev venv'de kurulu


def test_build_hierarchy_shapes() -> None:
    hspec = build_hierarchy(
        _panel(), hierarchy_cols=["state", "zone"], group_col="region", time_col="ds", target_col="y",
    )
    # state (2) + state/zone (4) + state/zone/region (4) = 10 düğüm, 4 bottom
    assert len(hspec.node_order) == 10
    assert len(hspec.bottom_ids) == 4
    assert hspec.s_matrix.shape == (10, 4)
    assert set(hspec.bottom_to_raw.values()) == {"A-x", "A-y", "B-x", "B-y"}
    # her bottom düğüm S'de kendi sütununda 1, aggregate satırlar çocuklarının toplamı
    assert hspec.s_matrix.sum(axis=1).tolist() == [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]


def test_build_hierarchy_rejects_nonunique_group_col() -> None:
    """group_col hiyerarşi genelinde tekrarlanıyorsa (globally-unique değilse) net hata."""
    df = _panel().rename(columns={"region": "_drop"})
    df["zone_dup"] = df["zone"]  # yalnız 2 değer, state'ler arasında tekrarlı
    with pytest.raises(ValueError, match="GLOBAL OLARAK BENZERSİZ"):
        build_hierarchy(
            df, hierarchy_cols=["state"], group_col="zone_dup", time_col="ds", target_col="y",
        )


def test_reconcile_coherence() -> None:
    """MinTrace(wls_struct) sonrası çocuklar toplamı = ebeveyn (matematiksel garanti)."""
    hspec = build_hierarchy(
        _panel(), hierarchy_cols=["state", "zone"], group_col="region", time_col="ds", target_col="y",
    )
    rng = np.random.default_rng(1)
    h = 3
    y_hat = rng.normal(50, 10, size=(len(hspec.node_order), h))  # tutarsız ham tahminler
    reconciled = reconcile(hspec.s_matrix, y_hat)
    assert reconciled.shape == y_hat.shape
    bottom_idx = [hspec.node_order.index(b) for b in hspec.bottom_ids]
    state_a_idx = hspec.node_order.index("A")
    az_x_idx = hspec.node_order.index("A/x")
    az_y_idx = hspec.node_order.index("A/y")
    np.testing.assert_allclose(
        reconciled[state_a_idx], reconciled[az_x_idx] + reconciled[az_y_idx], atol=1e-6
    )
    total_idx = [i for i, n in enumerate(hspec.node_order) if n in {"A", "B"}]
    np.testing.assert_allclose(
        reconciled[total_idx].sum(axis=0), reconciled[bottom_idx].sum(axis=0), atol=1e-6
    )


class _StubInner:
    """`FittedHierarchicalForecaster` sarmalayıcısını izole test etmek için sahte şampiyon.

    Her düğüme sabit bir taban değer + tarih-indeksi ekler (öngörülebilir, kontrollü tahmin)."""

    def __init__(self, base: dict[str, float]) -> None:
        self._base = base

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self._base.get(g, 0.0) for g in frame["region"]], dtype=np.float64
        )


def test_fitted_hierarchical_forecaster_predict_bottom_only() -> None:
    hspec = build_hierarchy(
        _panel(), hierarchy_cols=["state", "zone"], group_col="region", time_col="ds", target_col="y",
    )
    base = dict.fromkeys(hspec.node_order, 10.0)
    for b in hspec.bottom_ids:
        base[b] = 25.0  # bottom'lar tutarsız yüksek tahmin — reconciliation'ın etkisini görebilelim
    inner = _StubInner(base)
    fc = FittedHierarchicalForecaster(inner=inner, hspec=hspec)

    future = pd.DataFrame({
        "region": ["A-x", "A-y", "B-x", "B-y"],
        "ds": [pd.Timestamp("2022-06-01")] * 4,
    })
    preds = fc.predict(future)
    assert preds.shape == (4,)
    assert not np.isnan(preds).any()
    assert fc.feature_cols == []
    assert fc.inner is inner


def test_fitted_hierarchical_forecaster_unknown_group_yields_nan() -> None:
    hspec = build_hierarchy(
        _panel(), hierarchy_cols=["state", "zone"], group_col="region", time_col="ds", target_col="y",
    )
    inner = _StubInner(dict.fromkeys(hspec.node_order, 10.0))
    fc = FittedHierarchicalForecaster(inner=inner, hspec=hspec)
    future = pd.DataFrame({"region": ["ghost"], "ds": [pd.Timestamp("2022-06-01")]})
    preds = fc.predict(future)
    assert np.isnan(preds).all()
