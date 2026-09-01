"""Benchmark dataset kaydı — ilk dalga: OpenML + sklearn builtin tablo verileri.

Her loader `(DataFrame, target_col)` döndürür; DataFrame hedef kolonu **içerir**.
İndirme: `sklearn.datasets.fetch_openml` → `~/scikit_learn_data/` altına cache'lenir.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Loader = Callable[[], "tuple[pd.DataFrame, str]"]

_SUBSAMPLE_CAP = 60_000  # büyük setleri seed'li örnekle (süre değil, akışı doğrulamak istiyoruz)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Bir benchmark verisi + beklenen görev + naive baseline."""

    name: str
    loader: Loader
    task_hint: str  # regression | binary_classification | multiclass_classification
    naive: str  # mean | median | majority
    notes: str
    tags: list[str] = field(default_factory=list)


# --- yardımcılar ---------------------------------------------------------


def _subsample(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    if len(df) <= _SUBSAMPLE_CAP:
        return df
    return df.sample(n=_SUBSAMPLE_CAP, random_state=seed).reset_index(drop=True)


def _fetch_openml(name: str | None = None, *, data_id: int | None = None) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    kwargs: dict[str, object] = {"as_frame": True, "parser": "auto"}
    if data_id is not None:
        kwargs["data_id"] = data_id
    else:
        kwargs["name"] = name
        kwargs["version"] = "active"
    bunch = fetch_openml(**kwargs)  # type: ignore[arg-type]
    df = bunch.frame.copy()
    df.columns = [str(c) for c in df.columns]
    return df


# --- loader'lar --------------------------------------------------------


def _california() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.copy()
    return df, "MedHouseVal"


def _bike_sharing() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=42712)  # Bike_Sharing_Demand
    target = "count" if "count" in df.columns else df.columns[-1]
    return _subsample(df), target


def _adult() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=1590)  # adult, version 2
    target = "class"
    df[target] = df[target].astype(str).str.strip().str.rstrip(".")
    return _subsample(df), target


def _credit_g() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=31)  # credit-g
    return df, "class"


def _bank_marketing() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=1461)  # bank-marketing
    target = "Class" if "Class" in df.columns else df.columns[-1]
    return _subsample(df), target


def _covtype() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype(as_frame=True)
    df = bunch.frame.copy()
    return _subsample(df, seed=42).reset_index(drop=True), "Cover_Type"


# --- kayıt ------------------------------------------------------------

DATASETS: list[BenchmarkDataset] = [
    BenchmarkDataset(
        name="california_housing", loader=_california, task_hint="regression", naive="mean",
        notes="Temiz sayısal regresyon (20640×8).", tags=["regression", "small", "clean"],
    ),
    BenchmarkDataset(
        name="bike_sharing", loader=_bike_sharing, task_hint="regression", naive="mean",
        notes="Sayısal + kategorik + tarih regresyonu (17k).", tags=["regression", "mixed"],
    ),
    BenchmarkDataset(
        name="adult", loader=_adult, task_hint="binary_classification", naive="majority",
        notes="Kategorik + eksik + dengesiz ikili (49k).", tags=["binary", "categorical", "missing"],
    ),
    BenchmarkDataset(
        name="credit_g", loader=_credit_g, task_hint="binary_classification", naive="majority",
        notes="Küçük veri ikili (1000×20).", tags=["binary", "small"],
    ),
    BenchmarkDataset(
        name="bank_marketing", loader=_bank_marketing, task_hint="binary_classification", naive="majority",
        notes="Ağır dengesiz ikili (45k).", tags=["binary", "imbalanced"],
    ),
    BenchmarkDataset(
        name="covtype", loader=_covtype, task_hint="multiclass_classification", naive="majority",
        notes="7-sınıf orman örtüsü, tümü sayısal (60k örnek).", tags=["multiclass", "large"],
    ),
]

BY_NAME: dict[str, BenchmarkDataset] = {d.name: d for d in DATASETS}


def naive_prediction(train_y: pd.Series, n_test: int, kind: str) -> np.ndarray:
    """Test seti için naive baseline tahmini. Sınıflandırmada train_y dtype korunur."""
    if kind == "mean":
        return np.full(n_test, float(pd.to_numeric(train_y, errors="coerce").mean()))
    if kind == "median":
        return np.full(n_test, float(pd.to_numeric(train_y, errors="coerce").median()))
    mode = train_y.mode()
    value = mode.iloc[0] if not mode.empty else train_y.iloc[0]
    return np.full(n_test, value)  # majority — skaler dtype'ı (int/kategori) korunur
