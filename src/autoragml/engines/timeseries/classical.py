"""Native classical forecasting — Nixtla `StatsForecast` (ADR 0023).

Klasik modeller (`family ∈ {statistical, intermittent}`) reduction pipeline'ından
geçemez (panel API, sklearn değil). Buradan geçer: `cross_validation` → OOF (rolling-origin,
leakage-safe), `fit`/`predict` → serving. GES ensemble'a v1'de girmez (cutoff-tabanlı OOF).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
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
from autoragml.ensembling.greedy import bagged_greedy_selection, greedy_selection
from autoragml.logging import get_logger
from autoragml.models.estimator import resolve_class_path
from autoragml.scoring.metrics import compute_metrics, default_primary_metric, lower_is_better
from autoragml.validators.frame_ops import OOFArrays, prediction_health

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
CLASSICAL_FAMILIES = {"statistical", "intermittent"}
CLASSICAL_ENSEMBLE_KEY = "classical_ensemble"
_SEASON_PARAM = "season_length"
_FREQ_SEASON: dict[str, int] = {"H": 24, "D": 7, "B": 5, "W": 52, "M": 12, "Q": 4, "Y": 1, "A": 1}
_N_JOBS = -1  # statsforecast serileri kendi executor'ıyla paralelleştirir (Windows'ta güvenli)
_MAX_CV_WINDOWS = 3  # büyük panelde refit'li CV maliyeti — 3 pencere yeterli


@dataclass
class ClassicalCV:
    """`run_classical_reports`'un ürettiği rolling-origin cross_validation ızgarası (ADR 0035/P2).

    `cv`: kolonlar `[unique_id, ds, cutoff, _win, y, <alias>...]`. Ortak forecasting ensemble
    (`joint_ensemble`) reduction modellerini **aynı cutoff'larda** değerlendirip bu ızgaraya
    kolon ekler → tek GES tüm aileler üzerinde.
    """

    cv: pd.DataFrame
    alias_to_key: dict[str, str]
    horizon: int
    season: int
    freq: str


def is_classical(candidate: Candidate) -> bool:
    return candidate.family in CLASSICAL_FAMILIES


def _season_length(profile: DataProfile, freq: str | None) -> int:
    """Operasyonel mevsim = freq'in doğal periyodu (günlük→7). Yıllık (365) gibi uzun
    periyotlar AutoARIMA'yı patlatır ve CV history-guard'ını şişirir → tercih edilmez.
    Tespit edilen seasonality yalnız freq bilinmiyorsa ve makul (≤ 3× taban) ise kullanılır.
    """
    base = _FREQ_SEASON.get(freq[0].upper(), 0) if freq else 0
    if base:
        return base
    ts = profile.timeseries
    if ts and ts.seasonality:
        periods = sorted(int(s.period) for s in ts.seasonality if 2 <= int(s.period) <= 60)
        if periods:
            return periods[0]
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


def _report_from_oof(
    key: str,
    y_true: _Arr,
    y_pred: _Arr,
    win: pd.Series,
    group: np.ndarray,
    task: TaskSpec,
) -> ValidationReport:
    windows = sorted(pd.unique(win))
    folds = [
        FoldReport(
            fold_id=int(w),
            n_train=0,
            n_test=int((win == w).sum()),
            metrics=compute_metrics(y_true[(win == w).to_numpy()], y_pred[(win == w).to_numpy()], task.task),
        )
        for w in windows
    ]
    oof_metrics = compute_metrics(y_true, y_pred, task.task)
    oof_se: dict[str, float] = {}
    for k in oof_metrics:
        vals = [fr.metrics[k] for fr in folds if k in fr.metrics and np.isfinite(fr.metrics[k])]
        if len(vals) >= 2:
            oof_se[k] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return ValidationReport(
        candidate_key=key,
        split_kind=SplitKind.ROLLING_ORIGIN,
        folds=folds,
        oof_metrics=oof_metrics,
        oof_metric_se=oof_se,
        prediction_health=prediction_health(y_true, y_pred),
        leakage=LeakageReport(),
        nested=False,
        oof=OOFArrays(y_true=y_true, y_pred=y_pred, group=group),
    )


def _classical_ensemble(
    cv: pd.DataFrame,
    alias_to_cand: dict[str, Candidate],
    task: TaskSpec,
    config: RunConfig,
) -> tuple[ValidationReport, Candidate] | None:
    """Klasik OOF matrisinde GES → EAT-tarzı ansambl (M3/M4 winner deseni, ADR 0024)."""
    aliases = [a for a in alias_to_cand if a in cv.columns]
    if len(aliases) < 2:
        return None
    preds = np.column_stack([cv[a].to_numpy(dtype=np.float64) for a in aliases])
    y_true = cv["y"].to_numpy(dtype=np.float64)
    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)

    def metric_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return compute_metrics(yt, yp, task.task).get(primary, float("inf"))

    ec = config.ensemble
    # Klasik havuz küçük (≤6) ve model kalitesi çok değişken (auto_ets vs croston) → bagged-GES
    # ağırlığı fazla yayıyor. Plain GES seçici kalır (M4 winner deseni: iyi modellere ağır ağırlık).
    n_bags = ec.n_bags if (ec.bagging and preds.shape[1] > 6) else 0
    if n_bags:
        w = bagged_greedy_selection(
            preds, y_true, metric_fn=metric_fn, lower_is_better=lower, max_models=ec.max_models,
            sorted_init_k=ec.sorted_init_k, n_bags=n_bags, bag_fraction=ec.bag_fraction, seed=config.seed,
        )
    else:
        w = greedy_selection(
            preds, y_true, metric_fn=metric_fn, lower_is_better=lower,
            max_models=ec.max_models, sorted_init_k=ec.sorted_init_k,
        )
    nz = np.flatnonzero(w > 1e-9)
    if nz.size < 2:
        return None
    weights = (w[nz] / w[nz].sum()).tolist()
    blend = preds[:, nz] @ (w[nz] / w[nz].sum())
    report = _report_from_oof(
        CLASSICAL_ENSEMBLE_KEY, y_true, blend, cv["_win"], cv["unique_id"].to_numpy().astype(object), task
    )
    cand = Candidate(
        key=CLASSICAL_ENSEMBLE_KEY,
        name="Classical Ensemble (EAT)",
        family="ensemble",
        class_path="__classical_ensemble__",
        modalities=[task.modality],
        tasks=[task.task],
        ensemble_members={alias_to_cand[aliases[i]].key: weights[j] for j, i in enumerate(nz)},
    )
    logger.info(
        "[classical] classical_ensemble: %d üye (%s)",
        nz.size, ", ".join(alias_to_cand[aliases[i]].key for i in nz),
    )
    return report, cand


def run_classical_reports(
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    candidates: list[Candidate],
) -> tuple[list[ValidationReport], list[Candidate], ClassicalCV | None]:
    """Klasik adaylar → `cross_validation` OOF → per-model raporlar + `classical_ensemble` +
    ortak-ızgara `ClassicalCV` (ADR 0023/0024/0035-P2)."""
    classical = [c for c in candidates if is_classical(c)]
    if not classical:
        return [], [], None
    try:
        from statsforecast import StatsForecast
    except ImportError:  # pragma: no cover
        logger.warning("[classical] statsforecast yok — klasik modeller atlandı")
        return [], [], None

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
        return [], [], None

    models: list[Any] = []
    alias_to_cand: dict[str, Candidate] = {}
    for cand in classical:
        try:
            model = _build_model(cand, season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[classical] `%s` kurulamadı: %s", cand.key, exc)
            continue
        models.append(model)
        alias_to_cand[model.alias] = cand
    if not models:
        return [], [], None

    sf = StatsForecast(models=models, freq=freq, n_jobs=_N_JOBS, fallback_model=_fallback(season))
    try:
        cv = sf.cross_validation(df=cv_df, h=h, n_windows=n_windows, step_size=h)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[classical] cross_validation başarısız: %s", exc)
        return [], [], None
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    # Heterojen tarihli panelde cutoff'lar seriye göre değişir → pencere indeksiyle grupla.
    cv["_win"] = cv.groupby("unique_id")["cutoff"].rank(method="dense").astype(int)
    y_true = cv["y"].to_numpy(dtype=np.float64)
    group = cv["unique_id"].to_numpy().astype(object)

    reports = [
        _report_from_oof(cand.key, y_true, cv[alias].to_numpy(dtype=np.float64), cv["_win"], group, task)
        for alias, cand in alias_to_cand.items()
        if alias in cv.columns
    ]
    extra: list[Candidate] = []
    ens = _classical_ensemble(cv, alias_to_cand, task, config)
    if ens is not None:
        ens_report, ens_cand = ens
        reports.append(ens_report)
        extra.append(ens_cand)
    logger.info("[classical] %d model doğrulandı (h=%d, %d pencere, s=%d)", len(reports), h, n_windows, season)
    cv_grid = ClassicalCV(
        cv=cv[["unique_id", "ds", "cutoff", "_win", "y", *[a for a in alias_to_cand if a in cv.columns]]].copy(),
        alias_to_key={a: c.key for a, c in alias_to_cand.items() if a in cv.columns},
        horizon=h, season=season, freq=freq,
    )
    return reports, extra, cv_grid


class FittedClassicalForecaster:
    """Fitted `StatsForecast` (1..n model) + serving. `Predictor` protokolü (ADR 0023/0024).

    Tek model → `aliases=[a], weights=[1.0]`. `classical_ensemble` → çok model + GES ağırlıkları;
    `predict` = model-başı forecast kolonlarının ağırlıklı ortalaması.
    """

    __slots__ = (
        "_aliases", "_freq", "_group_col", "_h", "_last", "_sf", "_time_col", "_train_end", "_weights"
    )

    def __init__(
        self,
        *,
        sf: Any,
        aliases: list[str],
        weights: list[float],
        horizon: int,
        train_ndf: pd.DataFrame,
        group_col: str | None,
        time_col: str,
        freq: str,
    ) -> None:
        self._sf = sf
        self._aliases = aliases
        self._weights = np.asarray(weights, dtype=np.float64)
        self._h = horizon
        self._group_col = group_col
        self._time_col = time_col
        self._freq = freq
        self._last = train_ndf.groupby("unique_id")["y"].last().to_dict()
        self._train_end = pd.Timestamp(pd.to_datetime(train_ndf["ds"]).max())

    def _horizon_for(self, target_max: pd.Timestamp | None) -> int:
        """İstenen tarih penceresini kapsayacak adım sayısı (ADR 0023 — serving arbitrary future).

        Şampiyon `train − holdout` üzerinde fit edildiğinde (ADR 0020 holdout carve),
        `sf.predict(h)` yalnız fit sonrası `h` adımı üretir; gerçek gelecek daha ileride olabilir.
        """
        if target_max is None or pd.isna(target_max) or target_max <= self._train_end:
            return self._h
        try:
            steps = len(pd.date_range(self._train_end, target_max, freq=self._freq)) - 1
        except (ValueError, TypeError):
            return self._h
        return int(min(max(self._h, steps), self._h * 24 + 366))

    def predict(self, frame: pd.DataFrame) -> _Arr:
        tgt_ds = pd.to_datetime(frame[self._time_col], errors="coerce")
        horizon = self._horizon_for(tgt_ds.max())
        fc = self._sf.predict(h=horizon).reset_index()
        fc["ds"] = pd.to_datetime(fc["ds"])
        blended = np.zeros(len(fc), dtype=np.float64)
        for alias, w in zip(self._aliases, self._weights, strict=True):
            blended += w * fc[alias].to_numpy(dtype=np.float64)
        fc = fc[["unique_id", "ds"]].assign(_yhat=blended)

        key = pd.DataFrame(
            {
                "unique_id": frame[self._group_col].astype(str) if self._group_col else "series",
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


def _fit_forecaster(
    models: list[Any], weights: list[float], frame: pd.DataFrame, profile: DataProfile,
    task: TaskSpec, config: RunConfig,
) -> FittedClassicalForecaster:
    from statsforecast import StatsForecast

    ndf = _to_nixtla(frame, task)
    freq = _resolve_freq(profile)
    season = _season_length(profile, freq)
    sf = StatsForecast(models=models, freq=freq, n_jobs=_N_JOBS, fallback_model=_fallback(season))
    sf.fit(df=ndf)
    return FittedClassicalForecaster(
        sf=sf,
        aliases=[m.alias for m in models],
        weights=weights,
        horizon=_horizon(task, config),
        train_ndf=ndf,
        group_col=task.group_col,
        time_col=task.time_col or "ds",
        freq=freq,
    )


def refit_classical(
    candidate: Candidate, frame: pd.DataFrame, profile: DataProfile, task: TaskSpec, config: RunConfig
) -> FittedClassicalForecaster:
    """Tek klasik şampiyonu tüm train'de fit."""
    season = _season_length(profile, _resolve_freq(profile))
    return _fit_forecaster([_build_model(candidate, season)], [1.0], frame, profile, task, config)


def refit_classical_ensemble(
    ens_candidate: Candidate,
    all_candidates: list[Candidate],
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
) -> FittedClassicalForecaster:
    """`classical_ensemble` şampiyonu — üye modelleri tek `StatsForecast`'ta fit + GES ağırlıkları."""
    members = ens_candidate.ensemble_members or {}
    by_key = {c.key: c for c in all_candidates}
    season = _season_length(profile, _resolve_freq(profile))
    models: list[Any] = []
    weights: list[float] = []
    for mkey, w in members.items():
        cand = by_key.get(mkey)
        if cand is None:
            continue
        models.append(_build_model(cand, season))
        weights.append(float(w))
    if len(models) < 2:
        msg = "classical_ensemble refit: 2'den az üye"
        raise ValueError(msg)
    w_arr = np.asarray(weights, dtype=np.float64)
    return _fit_forecaster(models, (w_arr / w_arr.sum()).tolist(), frame, profile, task, config)
