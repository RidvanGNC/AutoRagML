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
_M5_MAX_SERIES = 400   # M5 30k seri → seed'li alt-küme (günlük × ~1900 gün, reduction maliyeti)
_M4_MAX_SERIES = 350   # M4 Quarterly/Hourly büyük gruplar → seed'li alt-küme

# --- dev (hızlı) profil: geliştirme sırasında "iyiye mi kötüye mi" sinyali için sıkı capler.
# Model kataloğu TAM kalır (tabicl/timesfm/neural dahil) — yalnız veri küçülür.
_FAST_MODE = False
_FAST_ROW_CAP = 5_000
_FAST_SERIES_CAP = 120


def set_fast_mode(on: bool = True) -> None:
    """Dev profil: tablo satır capi `_FAST_ROW_CAP`, panel seri capi `_FAST_SERIES_CAP`."""
    global _FAST_MODE
    _FAST_MODE = on


def fast_caps() -> tuple[int, int]:
    """(tablo satır capi, panel seri capi) — dev profil raporlaması için."""
    return _FAST_ROW_CAP, _FAST_SERIES_CAP


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
    hierarchy_cols: list[str] | None = None  # ADR 0045/0047 — hiyerarşik reconciliation (group_col üstü)

    @property
    def is_timeseries(self) -> bool:
        return self.time_col is not None


# --- yardımcılar -------------------------------------------------------


def _subsample(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    cap = _FAST_ROW_CAP if _FAST_MODE else _SUBSAMPLE_CAP
    if len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=seed).reset_index(drop=True)


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
    """Nixtla `unique_id, ds, y` → AutoRagML long-format (kolon adları korunur).

    Dev profil (`_FAST_MODE`) tüm panel loader'lar için ortak çıkış noktası burada
    olduğundan seri capi de burada uygulanır (`_FAST_SERIES_CAP`, seed'li).
    """
    out = y_df[["unique_id", "ds", "y"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["y"])
    if _FAST_MODE:
        ids = np.sort(out["unique_id"].unique())
        if len(ids) > _FAST_SERIES_CAP:
            keep = np.random.default_rng(42).choice(ids, size=_FAST_SERIES_CAP, replace=False)
            out = out[out["unique_id"].isin(set(keep))]
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


# --- 1. dalga: tablo -------------------------------------------------


def _california() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_california_housing

    return _subsample(fetch_california_housing(as_frame=True).frame.copy()), "MedHouseVal"


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


# --- 3. dalga: tablo genişletme (AMLB / OpenML — akademik tarama, ADR 0040) ---


def _openml_ds(data_id: int, target: str | None = None) -> tuple[pd.DataFrame, str]:
    df = _fetch_openml(data_id=data_id)
    tgt = target or df.columns[-1]
    return _subsample(df), tgt


def _wine_quality() -> tuple[pd.DataFrame, str]:
    return _openml_ds(287, "quality")           # 6497×12, karışık, ordinal hedef


def _ailerons() -> tuple[pd.DataFrame, str]:
    return _openml_ds(296, "goal")              # 13750×41, çok-öznitelikli sayısal regresyon


def _house_16h() -> tuple[pd.DataFrame, str]:
    return _openml_ds(574, "price")             # 22784×17, sayısal regresyon


def _diamonds() -> tuple[pd.DataFrame, str]:
    return _openml_ds(42225, "price")           # 53940×10, karışık (ordinal kategorik) regresyon


def _phoneme() -> tuple[pd.DataFrame, str]:
    return _openml_ds(1489, "Class")            # 5404×6, sayısal ikili


def _kc1() -> tuple[pd.DataFrame, str]:
    return _openml_ds(1067, "defects")          # 2109×22, dengesiz yazılım-hata ikili


def _amazon_access() -> tuple[pd.DataFrame, str]:
    return _openml_ds(4135, "target")           # 32769×10, TAMAMEN yüksek-kardinalite kategorik


def _mfeat_factors() -> tuple[pd.DataFrame, str]:
    return _openml_ds(12, "class")              # 2000×217, 10-sınıf (öznitelik yoğun)


def _segment() -> tuple[pd.DataFrame, str]:
    return _openml_ds(36, "class")              # 2310×20, 7-sınıf görüntü segment


def _vehicle() -> tuple[pd.DataFrame, str]:
    return _openml_ds(54, "Class")              # 846×19, 4-sınıf küçük


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


def _tourism_hier() -> tuple[pd.DataFrame, str]:
    """Tourism — YALNIZ coğrafya ağacı: state → zone → region, tüm-amaç serileri (ADR 0047).

    Grouped (coğrafya × seyahat amacı) yapının amaç boyutu şimdilik atlanır — `hierarchy_cols`
    lineer bir ağaç. Bottom = 76 bölge (`<L><L><L>All`), üstü 27 zone / 7 eyalet / 1 toplam.
    `hierarchy_cols=["state","zone"]` + `group_col="region"` → MinTrace reconciliation.
    """
    from datasetsforecast.hierarchical import HierarchicalData

    y_df, _s_df, tags = HierarchicalData.load(directory=str(_TS_DATA_DIR), group="TourismLarge")
    region_ids = {str(x) for x in tags["Country/State/Zone/Region"]}
    df = y_df[y_df["unique_id"].astype(str).isin(region_ids)].copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    if _FAST_MODE:
        ids = np.sort(df["unique_id"].unique())
        if len(ids) > _FAST_SERIES_CAP:
            keep = np.random.default_rng(42).choice(ids, size=_FAST_SERIES_CAP, replace=False)
            df = df[df["unique_id"].isin(set(keep))]
    df["state"] = df["unique_id"].str[0]
    df["zone"] = df["unique_id"].str[:2]
    df = df.rename(columns={"unique_id": "region"})
    out = df[["state", "zone", "region", "ds", "y"]].sort_values(["region", "ds"]).reset_index(drop=True)
    return out, "y"


def _m5_subset() -> tuple[pd.DataFrame, str]:
    from datasetsforecast.m5 import M5

    y_df, *_ = M5.load(directory=str(_TS_DATA_DIR))
    ids = np.sort(y_df["unique_id"].unique())
    rng = np.random.default_rng(42)
    keep = set(rng.choice(ids, size=min(_M5_MAX_SERIES, len(ids)), replace=False))
    return _nixtla_long(y_df[y_df["unique_id"].isin(keep)]), "y"


# --- 3. dalga: forecasting frekans genişletme (M4 / LongHorizon — GIFT-Eval ilhamı) ---


def _m4_group(group: str, *, cap: int | None = None) -> tuple[pd.DataFrame, str]:
    from datasetsforecast.m4 import M4

    y_df, *_ = M4.load(directory=str(_TS_DATA_DIR), group=group)
    if cap is not None:
        ids = np.sort(y_df["unique_id"].unique())
        keep = set(np.random.default_rng(42).choice(ids, size=min(cap, len(ids)), replace=False))
        y_df = y_df[y_df["unique_id"].isin(keep)]
    return _nixtla_long(y_df), "y"


def _m4_weekly() -> tuple[pd.DataFrame, str]:
    return _m4_group("Weekly")                  # 359 seri, haftalık — yeni frekans


def _m4_hourly() -> tuple[pd.DataFrame, str]:
    return _m4_group("Hourly", cap=_M4_MAX_SERIES)  # ~414 seri, saatlik, s=24 — güçlü mevsim


def _m4_quarterly() -> tuple[pd.DataFrame, str]:
    return _m4_group("Quarterly", cap=_M4_MAX_SERIES)  # 24k seri → alt-küme, çeyreklik


def _ett_h1() -> tuple[pd.DataFrame, str]:
    """ETTh1 — elektrik trafo (enerji alanı), saatlik, 7 uzun değişken-seri (GIFT-Eval Energy)."""
    from datasetsforecast.long_horizon import LongHorizon

    y_df, *_ = LongHorizon.load(directory=str(_TS_DATA_DIR), group="ETTh1")
    return _nixtla_long(y_df), "y"


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
        name="tourism_hier", loader=_tourism_hier, task_hint="forecasting", naive="seasonal_naive",
        notes="Tourism coğrafya ağacı (76 bölge → 27 zone → 7 eyalet) — hierarchy_cols + MinTrace(ols) "
        "reconciliation (ADR 0047). Amaç boyutu atlandı (grouped → lineer).",
        tags=["forecasting", "panel", "monthly", "hierarchical"],
        time_col="ds", group_col="region", horizon=12, season_length=12, primary_metric="wmape",
        hierarchy_cols=["state", "zone"],
    ),
    BenchmarkDataset(
        name="m5_subset", loader=_m5_subset, task_hint="forecasting", naive="seasonal_naive",
        notes=f"M5 (Walmart) — {_M5_MAX_SERIES} seri alt-küme, günlük, kesikli talep.",
        tags=["forecasting", "panel", "daily", "intermittent"],
        time_col="ds", group_col="unique_id", horizon=28, season_length=7, primary_metric="wmape",
    ),
    # --- 3. dalga: tablo genişletme (AMLB/OpenML — ADR 0040) ---
    BenchmarkDataset(name="wine_quality", loader=_wine_quality, task_hint="regression", naive="mean",
        notes="Şarap kalitesi — karışık, ordinal hedef (6497×12).", tags=["regression", "mixed"]),
    BenchmarkDataset(name="ailerons", loader=_ailerons, task_hint="regression", naive="mean",
        notes="Uçak kontrol — çok-öznitelikli sayısal regresyon (13750×41).", tags=["regression", "highdim"]),
    BenchmarkDataset(name="house_16h", loader=_house_16h, task_hint="regression", naive="mean",
        notes="Konut fiyat — sayısal regresyon (22784×17).", tags=["regression", "clean"]),
    BenchmarkDataset(name="diamonds", loader=_diamonds, task_hint="regression", naive="mean",
        notes="Elmas fiyat — karışık, ordinal kategorik (53940×10).", tags=["regression", "mixed", "large"]),
    BenchmarkDataset(name="phoneme", loader=_phoneme, task_hint="binary_classification", naive="majority",
        notes="Ses fonemi — sayısal ikili (5404×6).", tags=["binary", "numeric"]),
    BenchmarkDataset(name="kc1", loader=_kc1, task_hint="binary_classification", naive="majority",
        notes="Yazılım hatası — dengesiz ikili (2109×22).", tags=["binary", "imbalanced", "small"]),
    BenchmarkDataset(name="amazon_access", loader=_amazon_access, task_hint="binary_classification",
        naive="majority", notes="Erişim izni — TAMAMEN yüksek-kardinalite kategorik (32769×10).",
        tags=["binary", "categorical", "highcardinality"]),
    BenchmarkDataset(name="mfeat_factors", loader=_mfeat_factors, task_hint="multiclass_classification",
        naive="majority", notes="El yazısı rakam öznitelikleri — 10-sınıf (2000×217).",
        tags=["multiclass", "highdim"]),
    BenchmarkDataset(name="segment", loader=_segment, task_hint="multiclass_classification",
        naive="majority", notes="Görüntü segment — 7-sınıf (2310×20).", tags=["multiclass"]),
    BenchmarkDataset(name="vehicle", loader=_vehicle, task_hint="multiclass_classification",
        naive="majority", notes="Silüet — 4-sınıf küçük (846×19).", tags=["multiclass", "small"]),
    # --- 3. dalga: forecasting frekans genişletme (M4/LongHorizon — GIFT-Eval ilhamı) ---
    BenchmarkDataset(name="m4_weekly", loader=_m4_weekly, task_hint="forecasting", naive="seasonal_naive",
        notes="M4 Weekly — 359 seri, haftalık (yeni frekans).", tags=["forecasting", "panel", "weekly"],
        time_col="ds", group_col="unique_id", horizon=13, season_length=1, primary_metric="smape"),
    BenchmarkDataset(name="m4_hourly", loader=_m4_hourly, task_hint="forecasting", naive="seasonal_naive",
        notes="M4 Hourly — ~350 seri, saatlik, güçlü günlük mevsim.", tags=["forecasting", "panel", "hourly"],
        time_col="ds", group_col="unique_id", horizon=48, season_length=24, primary_metric="smape"),
    BenchmarkDataset(name="m4_quarterly", loader=_m4_quarterly, task_hint="forecasting",
        naive="seasonal_naive", notes="M4 Quarterly — 350 seri alt-küme, çeyreklik.",
        tags=["forecasting", "panel", "quarterly"],
        time_col="ds", group_col="unique_id", horizon=8, season_length=4, primary_metric="smape"),
    BenchmarkDataset(name="ett_h1", loader=_ett_h1, task_hint="forecasting", naive="seasonal_naive",
        notes="ETTh1 — elektrik trafo (enerji), saatlik, 7 uzun değişken-seri.",
        tags=["forecasting", "panel", "hourly", "energy"],
        time_col="ds", group_col="unique_id", horizon=48, season_length=24, primary_metric="wmape"),
]

BY_NAME: dict[str, BenchmarkDataset] = {d.name: d for d in DATASETS}

# Dev (hızlı) profil — 12 dataset, tam katalog + sıkı capler (~90-120 dk).
# Kapsam: regresyon (temiz/karışık/ordinal) · ikili (kategorik/küçük/yüksek-kardinalite) ·
# çok-sınıf (büyük/küçük) · forecasting (aylık düz / düşük-hacim hiyerarşik / kesikli günlük / saatlik).
DEV_SUBSET: list[str] = [
    "california_housing", "bike_sharing", "wine_quality",
    "adult", "credit_g", "amazon_access",
    "covtype", "vehicle",
    "m3_monthly", "tourism_large", "m5_subset", "m4_hourly",
]


def naive_prediction(train_y: pd.Series, n_test: int, kind: str) -> np.ndarray:
    """Tablo naive baseline (mean/median/majority). TS naive `run._seasonal_naive`'de."""
    if kind == "mean":
        return np.full(n_test, float(pd.to_numeric(train_y, errors="coerce").mean()))
    if kind == "median":
        return np.full(n_test, float(pd.to_numeric(train_y, errors="coerce").median()))
    mode = train_y.mode()
    value = mode.iloc[0] if not mode.empty else train_y.iloc[0]
    return np.full(n_test, value)
