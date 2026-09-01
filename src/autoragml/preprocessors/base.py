"""sklearn transformer'larını `Transform`/`FittedTransform` protokolüne saran taban (ADR 0011).

sklearn dizi tabanlı; burada DataFrame girer/çıkar, yalnız hedef kolon alt kümesine
uygulanır. `cross_fitted=True` (TargetEncoder) → `fit_transform` train'e cross-fitting,
`apply` test'e full-train kodlaması.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone

from autoragml.contracts.enums import Provenance
from autoragml.contracts.plan_context import PlanContext
from autoragml.transform import BaseTransform, FittedTransform


def _to_dense(arr: Any) -> Any:
    return arr.toarray() if hasattr(arr, "toarray") else arr


def _write_block(
    frame: pd.DataFrame, input_cols: list[str], output_cols: list[str], values: Any, *, expands: bool
) -> pd.DataFrame:
    block = pd.DataFrame(_to_dense(values), columns=output_cols, index=frame.index)
    if expands:
        return pd.concat([frame.drop(columns=input_cols), block], axis=1)
    out = frame.copy()
    out[output_cols] = block.to_numpy()
    return out


class _SklearnFitted:
    """Fitted sklearn transformer + kolon eşlemesi."""

    __slots__ = ("_est", "_expands", "_input_cols", "_name", "_output_cols", "provenance_fitted_on")

    def __init__(
        self,
        *,
        name: str,
        est: BaseEstimator,
        input_cols: list[str],
        output_cols: list[str],
        expands: bool,
    ) -> None:
        self._name = name
        self._est = est
        self._input_cols = input_cols
        self._output_cols = output_cols
        self._expands = expands
        self.provenance_fitted_on = Provenance.TRAIN

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self._est.transform(frame[self._input_cols])
        return _write_block(
            frame, self._input_cols, self._output_cols, transformed, expands=self._expands
        )

    def get_params(self) -> dict[str, object]:
        return {
            "transform": self._name,
            "input_cols": list(self._input_cols),
            "output_cols": list(self._output_cols),
        }


class SklearnColumnTransform(BaseTransform):
    """Bir sklearn transformer'ı kolon alt kümesine uygulayan `Transform`."""

    def __init__(
        self,
        *,
        name: str,
        estimator_factory: Callable[[pd.DataFrame], BaseEstimator],
        columns: list[str],
        needs_target: bool = False,
        cross_fitted: bool = False,
        expands: bool = False,
    ) -> None:
        self.name = name
        self._factory = estimator_factory
        self._columns = list(columns)
        self._needs_target = needs_target
        self._cross_fitted = cross_fitted
        self._expands = expands

    def _output_cols(self, est: BaseEstimator) -> list[str]:
        if self._expands and hasattr(est, "get_feature_names_out"):
            return [str(c) for c in est.get_feature_names_out(self._columns)]
        return list(self._columns)

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        est = clone(self._factory(frame))
        y = frame[ctx.target] if self._needs_target and ctx.target in frame.columns else None
        est.fit(frame[self._columns], y)
        return _SklearnFitted(
            name=self.name,
            est=est,
            input_cols=self._columns,
            output_cols=self._output_cols(est),
            expands=self._expands,
        )

    def fit_transform(
        self, frame: pd.DataFrame, ctx: PlanContext
    ) -> tuple[FittedTransform, pd.DataFrame]:
        if not self._cross_fitted:
            return super().fit_transform(frame, ctx)
        est = clone(self._factory(frame))
        transformed = est.fit_transform(frame[self._columns], frame[ctx.target])
        output_cols = self._output_cols(est)
        fitted = _SklearnFitted(
            name=self.name,
            est=est,
            input_cols=self._columns,
            output_cols=output_cols,
            expands=self._expands,
        )
        out = _write_block(frame, self._columns, output_cols, transformed, expands=self._expands)
        return fitted, out
