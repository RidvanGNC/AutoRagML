"""io.layout — wide tespiti + melt + layout kararı (ADR 0009)."""

from __future__ import annotations

import pandas as pd
import pytest

from autoragml.contracts.enums import Layout
from autoragml.exceptions import DataLoadError
from autoragml.io.layout import (
    determine_layout,
    guard_lazy_not_wide,
    looks_wide,
    normalize_layout,
)


def _wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tarih": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
            "A": [10.0, 12.0, 11.0],
            "B": [3.0, 4.0, 5.0],
            "C": [0.0, 1.0, 0.0],
        }
    )


def _long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "urun": ["A", "A", "B"],
            "ds": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-01"]),
            "adet": [10.0, 12.0, 3.0],
        }
    )


def test_looks_wide_true() -> None:
    assert looks_wide(_wide(), target="adet", time_col=None, group_col=None)


def test_looks_wide_false_when_group_col_given() -> None:
    assert not looks_wide(_wide(), target="adet", time_col=None, group_col="urun")


def test_looks_wide_false_for_long() -> None:
    assert not looks_wide(_long(), target="adet", time_col="ds", group_col="urun")


def test_normalize_melts_wide() -> None:
    frame, layout, time_col, group_col = normalize_layout(
        _wide(), target="adet", time_col=None, group_col=None
    )
    assert layout is Layout.WIDE_CONVERTED
    assert time_col == "tarih"
    assert group_col == "series_id"
    assert set(frame.columns) == {"tarih", "series_id", "adet"}
    assert len(frame) == 9  # 3 tarih * 3 seri


def test_normalize_long_stays_long() -> None:
    frame, layout, _t, _g = normalize_layout(_long(), target="adet", time_col="ds", group_col="urun")
    assert layout is Layout.LONG
    assert len(frame) == 3


def test_determine_layout_single_series() -> None:
    schema = {"ds": "datetime64[ns]", "value": "float64"}
    assert determine_layout(schema, target="value", time_col="ds", group_col=None) is Layout.SINGLE_SERIES


def test_determine_layout_na_without_time() -> None:
    schema = {"x": "float64", "y": "float64"}
    assert determine_layout(schema, target="y", time_col=None, group_col=None) is Layout.NA


def test_guard_lazy_not_wide_raises() -> None:
    schema = {"tarih": "datetime64[ns]", "A": "float64", "B": "float64", "C": "int64"}
    with pytest.raises(DataLoadError, match="wide-format"):
        guard_lazy_not_wide(schema, target="adet", time_col=None, group_col=None)
