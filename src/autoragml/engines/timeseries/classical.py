"""Native classical forecasting — Nixtla `StatsForecast` (ADR 0023).

Klasik modeller (`family ∈ {statistical, intermittent}`) reduction pipeline'ından
geçemez (panel API, sklearn değil). Buradan geçer: `cross_validation` → OOF (rolling-origin,
leakage-safe), `fit`/`predict` → serving. GES ensemble'a v1'de girmez (cutoff-tabanlı OOF).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import SplitKind, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.logging import get_logger
from autoragml.models.estimator import resolve_class_path
from autoragml.scoring.metrics import compute_metrics
from autoragml.validators.frame_ops import OOFArrays, prediction_health

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
CLASSICAL_FAMILIES = {"statistical", "intermittent"}
_SEASON_PARAM = "season_length"
_FREQ_SEASON: dict[str, int] = {"H": 24, "D": 7, "B": 5, "W": 52, "M": 12, "Q": 4, "Y": 1, "A": 1}
_N_JOBS = -1  # statsforecast serileri kendi executor'ıyla paralelleştirir (Windows'ta güvenli)
_MAX_CV_WINDOWS = 3  # büyük panelde refit'li CV maliyeti — 3 pencere yeterli


def is_classical(candidate: Candidate) -> bool:
    return candidate.family in CLASSICAL_FAMILIES


def _season_length(profile: DataProfile, freq: str | None) -> int:
    ts = profile.timeseries
    if ts and ts.seasonality:
        return max(int(s.period) for s in ts.seasonality)
    if freq:
        return _FREQ_SEASON.get(freq[0].upper(), 1)
    return 1


def _resolve_freq(profile: DataProfile) -> str:
    ts = profile.timeseries
    return (ts.freq if ts and ts.freq else None) or "D"


def _fallback(season_length: int) -> Any:
    """Kısa/patolojik seriler için güvenli varsayılan (StatsForecast `fallback_model`)."""
    from statsforecast.models import SeasonalNaive

    return SeasonalNaive(season_length=max(1, season_length))


def _build_model(candidate: Candidate, season_length: int) -> Any:
    path = resolve_class_path(candidate.class_path, Task.FORECASTING)
    module_name, _, cls_name = path.rpartition(".")
    module = __import__(module_name, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    kwargs = dict(candidate.default_params)
    if _SEASON_PARAM in inspect.signature(cls.__init__).parameters and _SEASON_PARAM not in kwargs:
        kwargs[_SEASON_PARAM] = season_length
    return cls(**kwargs)


def _to_nixtla(frame: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    gc, tc, y = task.group_col, task.time_col, task.targets[0]
    cols = {tc: "ds", y: "y"}
    if gc:
        cols[gc] = "unique_id"
    ndf = frame[[c for c in (gc, tc, y) if c]].rename(columns=cols).copy()
    if not gc:
        ndf["unique_id"] = "series"
    ndf["unique_id"] = ndf["unique_id"].astype(str)
    ndf["ds"] = pd.to_datetime(ndf["ds"], errors="coerce")
    ndf["y"] = pd.to_numeric(ndf["y"], errors="coerce")
    return ndf.dropna(subset=["ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _horizon(task: TaskSpec, config: RunConfig) -> int:
    pol = config.split_policy
    return int(task.horizon or (pol.horizon if pol and pol.horizon else None) or 4)


def run_classical_reports(
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    candidates: list[Candidate],
) -> list[ValidationReport]:
    """Klasik adaylar → `StatsForecast.cross_validation` OOF → `ValidationReport` listesi."""
    classical = [c for c in candidates if is_classical(c)]
    if not classical:
        return []
    try:
        from statsforecast import StatsForecast
    except ImportError:  # pragma: no cover
        logger.warning("[classical] statsforecast yok — klasik modeller atlandı")
        return []

    ndf = _to_nixtla(frame, task)
    freq = _resolve_freq(profile)
    season = _season_length(profile, freq)
    h = _horizon(task, config)

    # CV pencere sayısı serilerin uzunluğuna uyarlanır; çok kısa seriler CV'den düşer.
    # Her pencerede ilk eğitim penceresi ≥ 2·season olmalı (mevsimsel modeller için).
    lengths = ndf.groupby("unique_id").size()
    season_guard = 2 * max(season, 2)
    max_folds = min(_MAX_CV_WINDOWS, max(1, config.validation.default_rolling_folds))
    n_windows, needed = 1, h + season_guard
    for w in range(max_folds, 0, -1):
        req = w * h + season_guard
        if float((lengths >= req).mean()) >= 0.6 or w == 1:
            n_windows, needed = w, req
            break
    keep = lengths[lengths >= needed].index
    dropped = len(lengths) - len(keep)
    if dropped:
        logger.info(
            "[classical] %d/%d seri CV için çok kısa (n_windows=%d, gerekli≥%d)",
            dropped, len(lengths), n_windows, needed,
        )
    cv_df = ndf[ndf["unique_id"].isin(set(keep))].reset_index(drop=True)
    if cv_df["unique_id"].nunique() < 2:
        logger.warning("[classical] CV için yeterli seri yok — klasik modeller atlandı")
        return []

    models, alias_to_cand = [], {}
    for cand in classical:
        try:
            model = _build_model(cand, season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[classical] `%s` kurulamadı: %s", cand.key, exc)
            continue
        models.append(model)
        alias_to_cand[model.alias] = cand
    if not models:
        return []

    sf = StatsForecast(models=models, freq=freq, n_jobs=_N_JOBS, fallback_model=_fallback(season))
    try:
        cv = sf.cross_validation(df=cv_df, h=h, n_windows=n_windows, step_size=h)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[classical] cross_validation başarısız: %s", exc)
        return []
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    # Heterojen tarihli panelde cutoff'lar seriye göre değişir → pencere indeksiyle grupla.
    cv["_win"] = cv.groupby("unique_id")["cutoff"].rank(method="dense").astype(int)
    windows = sorted(pd.unique(cv["_win"]))

    reports: list[ValidationReport] = []
    for alias, cand in alias_to_cand.items():
        if alias not in cv.columns:
            continue
        y_true = cv["y"].to_numpy(dtype=np.float64)
        y_pred = cv[alias].to_numpy(dtype=np.float64)
        folds = [
            FoldReport(
                fold_id=int(w),
                n_train=0,
                n_test=int((cv["_win"] == w).sum()),
                metrics=compute_metrics(
                    cv.loc[cv["_win"] == w, "y"], cv.loc[cv["_win"] == w, alias], task.task
                ),
            )
            for w in windows
        ]
        oof_metrics = compute_metrics(y_true, y_pred, task.task)
        oof_se: dict[str, float] = {}
        for k in oof_metrics:
            vals = [fr.metrics[k] for fr in folds if k in fr.metrics and np.isfinite(fr.metrics[k])]
            if len(vals) >= 2:
                oof_se[k] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        reports.append(
            ValidationReport(
                candidate_key=cand.key,
                split_kind=SplitKind.ROLLING_ORIGIN,
                folds=folds,
                oof_metrics=oof_metrics,
                oof_metric_se=oof_se,
                prediction_health=prediction_health(y_true, y_pred),
                leakage=LeakageReport(),
                nested=False,
                oof=OOFArrays(
                    y_true=y_true, y_pred=y_pred, group=cv["unique_id"].to_numpy().astype(object)
                ),
            )
        )
    logger.info("[classical] %d model doğrulandı (h=%d, %d pencere, s=%d)", len(reports), h, n_windows, season)
    return reports


class FittedClassicalForecaster:
    """Fitted `StatsForecast` + serving. `Predictor` protokolünü karşılar (ADR 0023)."""

    __slots__ = ("_alias", "_group_col", "_h", "_last", "_sf", "_target", "_time_col")

    def __init__(
        self,
        *,
        sf: Any,
        alias: str,
        horizon: int,
        train_ndf: pd.DataFrame,
        group_col: str | None,
        time_col: str,
        target: str,
    ) -> None:
        self._sf = sf
        self._alias = alias
        self._h = horizon
        self._group_col = group_col
        self._time_col = time_col
        self._target = target
        self._last = train_ndf.groupby("unique_id")["y"].last().to_dict()

    def predict(self, frame: pd.DataFrame) -> _Arr:
        fc = self._sf.predict(h=self._h).reset_index()
        fc = fc.rename(columns={self._alias: "_yhat"})[["unique_id", "ds", "_yhat"]]
        fc["ds"] = pd.to_datetime(fc["ds"])

        key = pd.DataFrame(
            {
                "unique_id": (
                    frame[self._group_col].astype(str) if self._group_col else "series"
                ),
                "ds": pd.to_datetime(frame[self._time_col], errors="coerce"),
            }
        )
        merged = key.merge(fc, on=["unique_id", "ds"], how="left")
        out = merged["_yhat"].to_numpy(dtype=np.float64)
        gap = np.isnan(out)
        if gap.any():
            fill = merged.loc[gap, "unique_id"].map(self._last).to_numpy(dtype=np.float64)
            out[gap] = np.nan_to_num(fill, nan=0.0)
        return out

    @property
    def feature_cols(self) -> list[str]:
        return []


def refit_classical(
    candidate: Candidate,
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
) -> FittedClassicalForecaster:
    """Klasik şampiyonu tüm train'de fit → `FittedClassicalForecaster`."""
    from statsforecast import StatsForecast

    ndf = _to_nixtla(frame, task)
    freq = _resolve_freq(profile)
    season = _season_length(profile, freq)
    model = _build_model(candidate, season)
    sf = StatsForecast(models=[model], freq=freq, n_jobs=_N_JOBS, fallback_model=_fallback(season))
    sf.fit(df=ndf)
    return FittedClassicalForecaster(
        sf=sf,
        alias=model.alias,
        horizon=_horizon(task, config),
        train_ndf=ndf,
        group_col=task.group_col,
        time_col=task.time_col or "ds",
        target=task.targets[0],
    )
