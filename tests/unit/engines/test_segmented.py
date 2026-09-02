"""engines.segmented — FittedSegmentedPipeline yönlendirme + run_segmented (ADR 0028)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.engines.segmented import FittedSegmentedPipeline


class _ConstMember:
    """Sabit değer döndüren sahte alt-pipeline."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self._value, dtype=np.float64)

    @property
    def feature_cols(self) -> list[str]:
        return [f"f_{self._value:g}"]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["A", "A", "B", "C", "D", "B"],
            "ds": pd.date_range("2024-01-01", periods=6, freq="D"),
            "y": [1.0, 2, 3, 4, 5, 6],
        }
    )


def test_segmented_routes_by_group() -> None:
    pipe = FittedSegmentedPipeline(
        members={"smooth": _ConstMember(10.0), "lumpy": _ConstMember(99.0)},
        group_to_seg={"A": "smooth", "B": "lumpy", "C": "smooth"},
        fallback="smooth",
        group_col="g",
    )
    out = pipe.predict(_frame())
    # A→smooth(10), B→lumpy(99), C→smooth(10), D→bilinmiyor→fallback smooth(10)
    assert list(out) == [10.0, 10.0, 99.0, 10.0, 10.0, 99.0]


def test_segmented_feature_cols_union() -> None:
    pipe = FittedSegmentedPipeline(
        members={"s": _ConstMember(1.0), "l": _ConstMember(2.0)},
        group_to_seg={"A": "s", "B": "l"},
        fallback="s",
        group_col="g",
    )
    assert set(pipe.feature_cols) == {"f_1", "f_2"}


def test_segmented_unknown_group_uses_fallback() -> None:
    pipe = FittedSegmentedPipeline(
        members={"big": _ConstMember(7.0)},
        group_to_seg={},
        fallback="big",
        group_col="g",
    )
    out = pipe.predict(_frame())
    assert (out == 7.0).all()
