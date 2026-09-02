"""Benchmark dataset kaydı.

- **1. dalga (tablo):** OpenML + sklearn builtin → `~/scikit_learn_data/`.
- **2. dalga (zaman serisi panel):** Nixtla `datasetsforecast` (M3 / TourismLarge / M5 alt-küme) →
  `scripts/benchmarks/_data/`. Long-format `unique_id, ds, y` — AutoRagML ile birebir uyumlu.

Her loader `(DataFrame, target_col)` döndürür. TS meta'sı (`time_col`/`group_col`/`horizon`/
`season_length`) `BenchmarkDataset` alanlarındadır (statik, dataset'e özel).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

Loader = Callable[[], "tuple[pd.DataFrame, str]"]

_SUBSAMPLE_CAP = 60_000
_TS_DATA_DIR = Path(__file__).parent / "_data"
_M3_MAX_SERIES = 400   # M3 1428 seri → temsili alt-küme (klasik CV maliyeti)
_M5_MAX_SERIES = 1500  # M5 30k seri → seed'li alt-küme (akışı doğrulamak yeterli)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Bir benchmark verisi + beklenen görev + naive baseline (+ TS meta)."""

    name: str
    loader: Loader
    task_hint: str  # regression | binary_classification | multiclass_classification | forecasting
    naive: str  # mean | median | majority | seasonal_naive
    notes: str
    tags: list[str] = field(default_factory=list)
    time_col: str | None = None
    group_col: str | None = None
    horizon: int = 1
    season_length: int = 1
    primary_metric: str | None = None  # düşük-hacimli/intermittent panelde wmape (sMAPE patlar)

    @property
    def is_timeseries(self) -> bool:
        return self.time_col is not None


# --- yardımcılar -------------------------------------------------------


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


def _nixtla_long(y_df: pd.DataFrame) -> pd.DataFrame:
    """Nixtla `unique_id, ds, y` → AutoRagML long-format (kolon adları korunur)."""
    out = y_df[["unique_id", "ds", "y"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna(subset=["y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


# --- 1. dalga: tablo -------------------------------------------------


def _california() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True).frame.copy(), "MedHouseVal"


def _bike_sharing() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=42712)
    target = "count" if "count" in df.columns else df.columns[-1]
    return _subsample(df), target


def _adult() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=1590)
    df["class"] = df["class"].astype(str).str.strip().str.rstrip(".")
    return _subsample(df), "class"


def _credit_g() -> tuple[pd.DataFrame, str]:
    return _fetch_openml(data_id=31), "class"


def _bank_marketing() -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=1461)
    target = "Class" if "Class" in df.columns else df.columns[-1]
    return _subsample(df), target


def _covtype() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_covtype

    return _subsample(fetch_covtype(as_frame=True).frame.copy()).reset_index(drop=True), "Cover_Type"


# --- 2. dalga: zaman serisi panel ---------------------------------


def _m3_monthly() -> tuple[pd.DataFrame, str]:
    from datasetsforecast.m3 import M3

    y_df, *_ = M3.load(directory=str(_TS_DATA_DIR), group="Monthly")
    ids = np.sort(y_df["unique_id"].unique())
    keep = set(np.random.default_rng(42).choice(ids, size=min(_M3_MAX_SERIES, len(ids)), replace=False))
    return _nixtla_long(y_df[y_df["unique_id"].isin(keep)]), "y"


def _tourism_large() -> tuple[pd.DataFrame, str]:
    from datasetsforecast.hierarchical import HierarchicalData

    y_df, *_ = HierarchicalData.load(directory=str(_TS_DATA_DIR), group="TourismLarge")
    return _nixtla_long(y_df), "y"


def _m5_subset() -> tuple[pd.DataFrame, str]:
    from datasetsforecast.m5 import M5

    y_df, *_ = M5.load(directory=str(_TS_DATA_DIR))
    ids = np.sort(y_df["unique_id"].unique())
    rng = np.random.default_rng(42)
    keep = set(rng.choice(ids, size=min(_M5_MAX_SERIES, len(ids)), replace=False))
    return _nixtla_long(y_df[y_df["unique_id"].isin(keep)]), "y"


# --- kayıt ----------------------------------------------------------

DATASETS: list[BenchmarkDataset] = [
    BenchmarkDataset(
        name="california_housing", loader=_california, task_hint="regression", naive="mean",
        notes="Temiz sayısal regresyon (20640×8).", tags=["regression", "clean"],
    ),
    BenchmarkDataset(
        name="bike_sharing", loader=_bike_sharing, task_hint="regression", naive="mean",
        notes="Sayısal + kategorik regresyon (17k).", tags=["regression", "mixed"],
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
        notes="7-sınıf orman örtüsü (60k örnek).", tags=["multiclass", "large"],
    ),
    # --- 2. dalga: forecasting (panel) ---
    BenchmarkDataset(
        name="m3_monthly", loader=_m3_monthly, task_hint="forecasting", naive="seasonal_naive",
        notes="M3 Monthly — aylık, düz talep (yüksek hacim).", tags=["forecasting", "panel", "monthly"],
        time_col="ds", group_col="unique_id", horizon=18, season_length=12, primary_metric="smape",
    ),
    BenchmarkDataset(
        name="tourism_large", loader=_tourism_large, task_hint="forecasting", naive="seasonal_naive",
        notes="Avustralya turizm hiyerarşisi — aylık, düşük-hacimli seriler karışık (~555 seri).",
        tags=["forecasting", "panel", "monthly", "hierarchical"],
        time_col="ds", group_col="unique_id", horizon=12, season_length=12, primary_metric="wmape",
    ),
    BenchmarkDataset(
        name="m5_subset", loader=_m5_subset, task_hint="forecasting", naive="seasonal_naive",
        notes=f"M5 (Walmart) — {_M5_MAX_SERIES} seri alt-küme, günlük, kesikli talep.",
        tags=["forecasting", "panel", "daily", "intermittent"],
        time_col="ds", group_col="unique_id", horizon=28, season_length=7, primary_metric="wmape",
    ),
]

BY_NAME: dict[str, BenchmarkDataset] = {d.name: d for d in DATASETS}


def naive_prediction(train_y: pd.Series, n_test: int, kind: str) -> np.ndarray:
    """Tablo naive baseline (mean/median/majority). TS naive `run._seasonal_naive`'de."""
    if kind == "mean":
        return np.full(n_test, float(pd.to_numeric(train_y, errors="coerce").mean()))
    if kind == "median":
        return np.full(n_test, float(pd.to_numeric(train_y, errors="coerce").median()))
    mode = train_y.mode()
    value = mode.iloc[0] if not mode.empty else train_y.iloc[0]
    return np.full(n_test, value)
