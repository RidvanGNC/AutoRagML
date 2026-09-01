"""Dönüşüm protokolleri — leakage-safe by construction (ADR 0011).

`Transform` (fit edilmemiş) → `FittedTransform` (immutable). `preprocessors` katalog
transformları ve `dynamics/recipes` custom transformlar bu protokole uyar.

`fit`/`fit_transform`'u **yalnız `validators`** çağırır (fold içinde,
`PlanContext.provenance == "train"`). Kullanıcı/recipe kodu split sınırını görmez.

`fit_transform` ayrı bir metottur çünkü bazı dönüşümler (ör. target encoding) train
çıktısını **cross-fitting** ile üretir ama test'e full-train kodlaması uygular
(`fit().apply() != fit_transform()`, sklearn deseni).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import pandas as pd

from autoragml.contracts.enums import Provenance
from autoragml.contracts.plan_context import PlanContext


@runtime_checkable
class FittedTransform(Protocol):
    """Öğrenilmiş, immutable dönüşüm. Yalnız `apply` — parametre değiştirmez."""

    provenance_fitted_on: Provenance

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Öğrenilmiş parametreyi uygula (saf)."""
        ...

    def get_params(self) -> dict[str, object]:
        """Serialize edilebilir parametreler (`ModelBundle` metadata'sı)."""
        ...


@runtime_checkable
class Transform(Protocol):
    """Fit edilmemiş dönüşüm tanımı."""

    name: str

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        """Yalnız train frame'inden öğren; immutable `FittedTransform` döndür."""
        ...

    def fit_transform(
        self, frame: pd.DataFrame, ctx: PlanContext
    ) -> tuple[FittedTransform, pd.DataFrame]:
        """Fit + train çıktısını üret (cross-fitting olabilir)."""
        ...


class BaseTransform(ABC):
    """Transform protokolü için taban — `fit_transform` varsayılanı fit + apply."""

    name: str = "transform"

    @abstractmethod
    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedTransform:
        """Alt sınıf uygular."""

    def fit_transform(
        self, frame: pd.DataFrame, ctx: PlanContext
    ) -> tuple[FittedTransform, pd.DataFrame]:
        fitted = self.fit(frame, ctx)
        return fitted, fitted.apply(frame)


class StatelessFitted:
    """Parametre öğrenmeyen dönüşümler için `FittedTransform` sarımı."""

    __slots__ = ("_fn", "_params", "provenance_fitted_on")

    def __init__(
        self,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        params: dict[str, object] | None = None,
        provenance: Provenance = Provenance.TRAIN,
    ) -> None:
        self._fn = fn
        self._params = params or {}
        self.provenance_fitted_on = provenance

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._fn(frame)

    def get_params(self) -> dict[str, object]:
        return dict(self._params)
