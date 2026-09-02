"""FittedEnsemblePipeline — ağırlıklı ensemble runtime nesnesi (ADR 0021).

Üyeler **postprocess'siz + pre_transform'suz** `FittedModelPipeline`'lardır; ensemble
`pre_transform`'u (TS reduction) bir kez uygular, üye tahminlerinin ağırlıklı ortalamasını
alır, ardından **tek** ensemble-düzeyi postprocessor çalıştırır.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.engines.model_pipeline import Predictor
from autoragml.postprocessors import FittedPostprocessor

_Arr = npt.NDArray[np.float64]


class FittedEnsemblePipeline:
    """Ağırlıklı üye-tahmin ortalaması + ensemble-düzeyi postprocess — tek `predict`.

    Üyeler `FittedModelPipeline` (GES) veya iç içe `FittedEnsemblePipeline` (bagged üye — ADR 0022).
    """

    __slots__ = ("_members", "_postprocessor", "_pre_transform", "_weights")

    def __init__(
        self,
        *,
        members: Sequence[Predictor],
        weights: list[float],
        pre_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        postprocessor: FittedPostprocessor | None = None,
    ) -> None:
        if len(members) != len(weights) or not members:
            msg = "FittedEnsemblePipeline: üye/ağırlık sayısı tutarsız"
            raise ValueError(msg)
        self._members = list(members)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._pre_transform = pre_transform
        self._postprocessor = postprocessor

    def predict(self, frame: pd.DataFrame) -> _Arr:
        if self._pre_transform is not None:
            frame = self._pre_transform(frame)
        stacked = np.column_stack([np.asarray(m.predict(frame), dtype=np.float64) for m in self._members])
        blended = stacked @ self._weights
        if self._postprocessor is not None:
            blended = self._postprocessor.apply(blended)
        return blended

    @property
    def feature_cols(self) -> list[str]:
        cols: set[str] = set()
        for m in self._members:
            cols.update(m.feature_cols)
        return sorted(cols)

    @property
    def members(self) -> list[Predictor]:
        return list(self._members)

    @property
    def weights(self) -> list[float]:
        return self._weights.tolist()
