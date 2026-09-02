"""Hedef dönüşümü — `y` üzerinde forward + inverse (ADR 0010/0013/0026).

`candidate_ops` `target` grubunun seçimi (`none`/`log1p`/`yeo_johnson`/`quantile`/
`seasonal_difference`). `engine`/`fine_tuners` estimator'ın etrafında kullanır: fit'te
`forward(y)`, tahminde `inverse(y_pred)`. Fit yalnız train `y`'sinde.

`seasonal_difference` (ADR 0026): `y_t − y_{t−s}` (`s ≥ h` iken tersine çevrilebilir);
`ref` = `y_{t−s}` dizisi (reduction'ın `{target}_sdiff_ref` kolonu). Forward/inverse
`ref` argümanı alır (diğer seçimler yok sayar).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

_Array = npt.NDArray[np.float64]
_SEASONAL_DIFF = "seasonal_difference"


class FittedTargetTransform:
    """Fitted hedef dönüşümü — forward/inverse."""

    __slots__ = ("_choice", "_est")

    def __init__(self, choice: str, est: object | None) -> None:
        self._choice = choice
        self._est = est

    @property
    def choice(self) -> str:
        return self._choice

    def forward(self, y: _Array, ref: _Array | None = None) -> _Array:
        if self._choice == "none":
            return y
        if self._choice == _SEASONAL_DIFF:
            assert ref is not None, "seasonal_difference forward: ref gerekli"
            return np.asarray(y - ref, dtype=np.float64)
        if self._choice == "log1p":
            return np.asarray(np.sign(y) * np.log1p(np.abs(y)), dtype=np.float64)
        assert self._est is not None
        return np.asarray(self._est.transform(y.reshape(-1, 1)).ravel(), dtype=np.float64)  # type: ignore[attr-defined]

    def inverse(self, y: _Array, ref: _Array | None = None) -> _Array:
        if self._choice == "none":
            return y
        if self._choice == _SEASONAL_DIFF:
            assert ref is not None, "seasonal_difference inverse: ref gerekli"
            return np.asarray(y + ref, dtype=np.float64)
        if self._choice == "log1p":
            return np.asarray(np.sign(y) * np.expm1(np.abs(y)), dtype=np.float64)
        assert self._est is not None
        return np.asarray(
            self._est.inverse_transform(y.reshape(-1, 1)).ravel(), dtype=np.float64  # type: ignore[attr-defined]
        )


class TargetTransform:
    """Fit edilmemiş hedef dönüşümü."""

    __slots__ = ("_choice",)

    def __init__(self, choice: str) -> None:
        self._choice = choice

    def fit(self, y: _Array) -> FittedTargetTransform:
        if self._choice in {"none", "log1p", _SEASONAL_DIFF}:
            return FittedTargetTransform(self._choice, None)
        if self._choice == "yeo_johnson":
            est: object = PowerTransformer(method="yeo-johnson", standardize=True)
        elif self._choice == "quantile":
            est = QuantileTransformer(
                n_quantiles=max(10, min(1000, y.shape[0])), output_distribution="normal"
            )
        else:
            msg = f"Bilinmeyen hedef dönüşümü: {self._choice!r}"
            raise ValueError(msg)
        est.fit(y.reshape(-1, 1))  # type: ignore[attr-defined]
        return FittedTargetTransform(self._choice, est)
