"""FittedModelPipeline — `ModelBundle.pipeline` runtime nesnesi (ADR 0011 + 0015).

Ham DataFrame → feature pipeline → X → estimator → hedef inverse → tahmin.
`RunResult.predict()` bunu çağırır. Serialize: `persistence` (joblib) — burada saf çalışma.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.postprocessors import FittedPostprocessor
from autoragml.preprocessors.target import FittedTargetTransform
from autoragml.transform import FittedTransform

_Arr = npt.NDArray[np.float64]


@runtime_checkable
class Predictor(Protocol):
    """`predict` + `feature_cols` sunan fitted nesne — bag/ensemble üyeleri bu protokolü karşılar."""

    def predict(self, frame: pd.DataFrame) -> _Arr: ...

    @property
    def feature_cols(self) -> list[str]: ...


class FittedModelPipeline:
    """Fitted feature pipeline + estimator + hedef dönüşümü + postprocess — tek `predict`."""

    __slots__ = (
        "_estimator",
        "_feature_cols",
        "_feature_pipeline",
        "_postprocessor",
        "_pre_transform",
        "_reserved",
        "_target_ref_col",
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
        postprocessor: FittedPostprocessor | None = None,
        target_ref_col: str | None = None,
    ) -> None:
        self._feature_pipeline = feature_pipeline
        self._estimator = estimator
        self._target_transform = target_transform
        self._feature_cols = feature_cols
        self._reserved = reserved
        self._pre_transform = pre_transform
        self._postprocessor = postprocessor
        self._target_ref_col = target_ref_col  # seasonal_difference ref kolonu (ADR 0026)

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
        ref = None
        if self._target_ref_col is not None and self._target_ref_col in transformed.columns:
            ref = pd.to_numeric(transformed[self._target_ref_col], errors="coerce").to_numpy(dtype=np.float64)
        out = self._target_transform.inverse(raw, ref=ref)
        if self._postprocessor is not None:
            out = self._postprocessor.apply(out)
        return out

    def _design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._pre_transform is not None:
            frame = self._pre_transform(frame)
        transformed = self._feature_pipeline.apply(frame)
        x = transformed.drop(
            columns=[c for c in self._reserved if c in transformed.columns], errors="ignore"
        )
        x = x.reindex(columns=self._feature_cols, fill_value=0.0)
        out: pd.DataFrame = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return out

    def predict_proba(self, frame: pd.DataFrame) -> _Arr:
        """Sınıflandırma olasılık matrisi (ADR 0036) — estimator `predict_proba` sunmalı."""
        return np.asarray(self._estimator.predict_proba(self._design_matrix(frame)), dtype=np.float64)

    @property
    def supports_proba(self) -> bool:
        """Estimator `predict_proba` sunuyor mu (RidgeClassifier/LinearSVC vb. sunmaz — ADR 0036).

        sklearn `Pipeline` + `_available_if` descriptor'ı da doğru propagate eder → `getattr`
        eksik metotta `AttributeError` atar, `None` döner.
        """
        return callable(getattr(self._estimator, "predict_proba", None))

    @property
    def classes(self) -> Any:
        return getattr(self._estimator, "classes_", getattr(self._estimator, "_classes", None))

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    @property
    def estimator(self) -> Any:
        """Fitted tahminci — introspection (feature importance, explain) için salt-okunur."""
        return self._estimator
