"""Op adı → `Transform` fabrikası (ADR 0011 + 0012).

`AdaptivePlan.committed_ops` / seçilmiş `candidate_ops` bu katalogla somut transform'a çevrilir.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)

from autoragml.dynamics.recipes import get_recipe
from autoragml.exceptions import AutoRagMLError
from autoragml.preprocessors.base import SklearnColumnTransform
from autoragml.preprocessors.stateless import (
    ColumnDropper,
    DateExpander,
    HashingEncoder,
    Log1pTransform,
)
from autoragml.transform import Transform

_Factory = Callable[[pd.DataFrame], BaseEstimator]


class PreprocessError(AutoRagMLError):
    """Bilinmeyen op / kurulamayan transform."""


def build_encode(columns: list[str], strategy: str) -> Transform:
    """`encode` op'unu stratejisine göre somutlaştır."""
    if strategy == "onehot":
        factory: _Factory = lambda _f: OneHotEncoder(  # noqa: E731
            handle_unknown="ignore", sparse_output=False
        )
        return SklearnColumnTransform(name="onehot", estimator_factory=factory, columns=columns, expands=True)
    if strategy == "ordinal":
        factory = lambda _f: OrdinalEncoder(  # noqa: E731
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
        )
        return SklearnColumnTransform(name="ordinal", estimator_factory=factory, columns=columns)
    if strategy == "target_encode":
        return SklearnColumnTransform(
            name="target_encode",
            estimator_factory=lambda _f: TargetEncoder(smooth="auto"),
            columns=columns,
            needs_target=True,
            cross_fitted=True,
        )
    if strategy == "hashing":
        return HashingEncoder(columns)
    msg = f"Bilinmeyen encode stratejisi: {strategy!r}"
    raise PreprocessError(msg)


def build_numeric_transform(columns: list[str], choice: str) -> Transform | None:
    """`candidate_ops` seçimi → numerik transform (`none` → None)."""
    if choice in {"none", ""}:
        return None
    if choice == "log1p":
        return Log1pTransform(columns)
    if choice == "yeo_johnson":
        return SklearnColumnTransform(
            name="yeo_johnson",
            estimator_factory=lambda _f: PowerTransformer(method="yeo-johnson", standardize=True),
            columns=columns,
        )
    if choice == "quantile":
        return SklearnColumnTransform(
            name="quantile",
            estimator_factory=lambda f: QuantileTransformer(
                n_quantiles=max(10, min(1000, len(f))), output_distribution="normal"
            ),
            columns=columns,
        )
    msg = f"Bilinmeyen numerik transform seçimi: {choice!r}"
    raise PreprocessError(msg)


def build_op(op: str, columns: list[str], params: dict[str, object]) -> Transform | None:
    """Tek bir committed op → transform."""
    if op == "drop":
        return ColumnDropper(columns)
    if op == "date_expand":
        return DateExpander(columns, keep_original=bool(params.get("keep_original", False)))
    if op == "impute":
        strategy = str(params.get("strategy", "median"))
        factory: _Factory = lambda _f: SimpleImputer(  # noqa: E731
            strategy=strategy, keep_empty_features=True
        )
        return SklearnColumnTransform(name=f"impute_{strategy}", estimator_factory=factory, columns=columns)
    if op == "encode":
        return build_encode(columns, str(params.get("strategy", "onehot")))
    if op == "scale":
        return SklearnColumnTransform(
            name="scale", estimator_factory=lambda _f: StandardScaler(), columns=columns
        )
    if op.startswith("recipe:"):
        return get_recipe(op.split(":", 1)[1])()
    msg = f"Bilinmeyen committed op: {op!r}"
    raise PreprocessError(msg)
