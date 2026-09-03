"""FittedStackPipeline — saf L2 stacking runtime nesnesi (ADR 0034).

L1 üyeleri **postprocess'siz + pre_transform'suz** `Predictor`'lardır; stack pipeline
`pre_transform`'u (TS reduction) bir kez uygular, üye tahminlerini `Z` matrisine dizer,
L2 stacker'ı çalıştırır, ardından **tek** stack-düzeyi postprocessor uygular.

joblib-picklable (sklearn L1 + sklearn L2) — sidecar gerekmez.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.engines.model_pipeline import Predictor
from autoragml.postprocessors import FittedPostprocessor

_Arr = npt.NDArray[np.float64]


class FittedStackPipeline:
    """L1 üye tahminleri → `Z` → L2 stacker → (opsiyonel) stack-düzeyi postprocess."""

    __slots__ = ("_l2", "_members", "_postprocessor", "_pre_transform")

    def __init__(
        self,
        *,
        members: Sequence[Predictor],
        l2_estimator: Any,
        pre_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        postprocessor: FittedPostprocessor | None = None,
    ) -> None:
        if len(members) < 2:
            msg = "FittedStackPipeline: en az 2 L1 üye gerekir"
            raise ValueError(msg)
        self._members = list(members)
        self._l2 = l2_estimator
        self._pre_transform = pre_transform
        self._postprocessor = postprocessor

    def predict(self, frame: pd.DataFrame) -> _Arr:
        if self._pre_transform is not None:
            frame = self._pre_transform(frame)
        z = np.column_stack(
            [np.asarray(m.predict(frame), dtype=np.float64) for m in self._members]
        )
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.asarray(self._l2.predict(z), dtype=np.float64)
        if self._postprocessor is not None:
            out = self._postprocessor.apply(out)
        return out

    @property
    def feature_cols(self) -> list[str]:
        cols: set[str] = set()
        for m in self._members:
            cols.update(m.feature_cols)
        return sorted(cols)

    @property
    def members(self) -> list[Predictor]:
        return list(self._members)
