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

    __slots__ = ("_classes", "_group_col", "_members", "_postprocessor", "_pre_transform", "_weights")

    def __init__(
        self,
        *,
        members: Sequence[Predictor],
        weights: list[float],
        pre_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        postprocessor: FittedPostprocessor | None = None,
        classes: object | None = None,
        group_col: str | None = None,
    ) -> None:
        if len(members) != len(weights) or not members:
            msg = "FittedEnsemblePipeline: üye/ağırlık sayısı tutarsız"
            raise ValueError(msg)
        self._members = list(members)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._pre_transform = pre_transform
        self._postprocessor = postprocessor
        # ADR 0036: sınıflandırma modu — üyeler `predict_proba` sunuyor + sınıf sırası verildi
        self._classes = np.asarray(classes) if classes is not None else None
        self._group_col = group_col  # ADR 0044: predict_interval per_group için

    def _blend_proba(self, frame: pd.DataFrame) -> _Arr:
        acc: _Arr | None = None
        for m, w in zip(self._members, self._weights, strict=True):
            p = np.asarray(m.predict_proba(frame), dtype=np.float64)  # type: ignore[attr-defined]
            acc = w * p if acc is None else acc + w * p
        assert acc is not None
        return acc

    def _blend_point(self, frame: pd.DataFrame) -> tuple[_Arr, npt.NDArray[np.object_] | None]:
        """Postprocessor'dan ÖNCE ağırlıklı üye ortalaması + (varsa) grup dizisi."""
        if self._pre_transform is not None:
            frame = self._pre_transform(frame)
        stacked = np.column_stack([np.asarray(m.predict(frame), dtype=np.float64) for m in self._members])
        blended: _Arr = stacked @ self._weights
        group = None
        if self._group_col is not None and self._group_col in frame.columns:
            group = frame[self._group_col].to_numpy(dtype=object)
        return blended, group

    def predict(self, frame: pd.DataFrame) -> _Arr:
        if self._classes is not None:  # sınıflandırma → olasılık blend + argmax
            blend = self._blend_proba(
                self._pre_transform(frame) if self._pre_transform is not None else frame
            )
            return np.asarray(self._classes[np.argmax(blend, axis=1)], dtype=np.float64)
        blended, _ = self._blend_point(frame)
        if self._postprocessor is not None:
            blended = self._postprocessor.apply(blended)
        return blended

    def predict_interval(
        self, frame: pd.DataFrame, *, coverage: float | None = None
    ) -> tuple[_Arr, _Arr]:
        """Nokta tahmin ± split-conformal aralık (ADR 0044). Sınıflandırma modunda desteklenmez."""
        if self._classes is not None:
            msg = "FittedEnsemblePipeline.predict_interval: sınıflandırmada desteklenmiyor (ADR 0044)"
            raise NotImplementedError(msg)
        blended, group = self._blend_point(frame)
        if self._postprocessor is None:
            return blended, blended
        return self._postprocessor.interval(blended, group=group, coverage=coverage)

    def predict_proba(self, frame: pd.DataFrame) -> _Arr:
        if self._pre_transform is not None:
            frame = self._pre_transform(frame)
        return self._blend_proba(frame)

    @property
    def classes(self) -> object | None:
        return None if self._classes is None else self._classes

    @property
    def supports_proba(self) -> bool:
        return all(getattr(m, "supports_proba", False) for m in self._members)

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
