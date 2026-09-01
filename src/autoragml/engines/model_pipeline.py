"""FittedModelPipeline — `ModelBundle.pipeline` runtime nesnesi (ADR 0011 + 0015).

Ham DataFrame → feature pipeline → X → estimator → hedef inverse → tahmin.
`RunResult.predict()` bunu çağırır. Serialize: `persistence` (joblib) — burada saf çalışma.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.preprocessors.target import FittedTargetTransform
from autoragml.transform import FittedTransform

_Arr = npt.NDArray[np.float64]


class FittedModelPipeline:
    """Fitted feature pipeline + estimator + hedef dönüşümü — tek `predict`."""

    __slots__ = (
        "_estimator",
        "_feature_cols",
        "_feature_pipeline",
        "_pre_transform",
        "_reserved",
        "_target_transform",
    )

    def __init__(
        self,
        *,
        feature_pipeline: FittedTransform,
        estimator: Any,
        target_transform: FittedTargetTransform,
        feature_cols: list[str],
        reserved: set[str],
        pre_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        self._feature_pipeline = feature_pipeline
        self._estimator = estimator
        self._target_transform = target_transform
        self._feature_cols = feature_cols
        self._reserved = reserved
        self._pre_transform = pre_transform

    def predict(self, frame: pd.DataFrame) -> _Arr:
        """Ham feature frame'inden nokta tahmini.

        Zaman serisi için: `frame` lag'leri hesaplayacak kadar geçmiş içermeli
        (reduction FE `pre_transform` ile yeniden uygulanır).
        """
        if self._pre_transform is not None:
            frame = self._pre_transform(frame)
        transformed = self._feature_pipeline.apply(frame)
        x = transformed.drop(columns=[c for c in self._reserved if c in transformed.columns], errors="ignore")
        x = x.reindex(columns=self._feature_cols, fill_value=0.0)
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw = np.asarray(self._estimator.predict(x), dtype=np.float64)
        return self._target_transform.inverse(raw)

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)
