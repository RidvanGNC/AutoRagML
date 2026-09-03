"""Native nöral forecasting — Nixtla `neuralforecast` (ADR 0032).

`family == "neural_ts"` adaylar `NeuralForecast` üzerinden: `cross_validation` → OOF
(rolling-origin, leakage-safe), `fit`/`predict` → serving. Reduction pipeline'ından geçmez
(panel API). ADR 0023 (klasik) deseninin ikizi. `neural_enabled=auto` iken **yalnız GPU**.

`neural_search=True` → `Auto*` modelleri (kütüphane HPO). Serving `FittedNeuralForecaster`
joblib-picklable **değil** → `persistence.bundle` `_NEURAL_TS_DIR` sidecar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import ValidationReport
from autoragml.engines.timeseries.classical import (
    _horizon,
    _report_from_oof,
    _resolve_freq,
    _season_length,
    _to_nixtla,
)
from autoragml.logging import get_logger
from autoragml.models.estimator import resolve_class_path
from autoragml.models.torch_env import configure_torch, quiet_cwd, resolve_device

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
NEURAL_TS_FAMILY = "neural_ts"
_MULTIVARIATE = {"itransformer", "tsmixer"}  # `n_series` runtime parametresi gerekir
_MAX_CV_WINDOWS = 2  # nöral CV refit'li — 2 pencere (klasik 3'ten daha pahalı)


def is_neural_ts(candidate: Candidate) -> bool:
    return candidate.family == NEURAL_TS_FAMILY


def neural_ts_available() -> bool:
    try:
        import neuralforecast  # noqa: F401
    except ImportError:
        return False
    return True


def _input_size(h: int, season: int) -> int:
    return max(2 * h, 3 * max(season, 1), 16)


def _build_nf_model(candidate: Candidate, h: int, season: int, n_series: int, config: RunConfig) -> Any:
    path = resolve_class_path(candidate.class_path, Task.FORECASTING)
    module_name, _, cls_name = path.rpartition(".")
    cls = getattr(__import__(module_name, fromlist=[cls_name]), cls_name)
    kwargs: dict[str, Any] = {
        "h": h,
        "input_size": _input_size(h, season),
        "random_seed": config.seed,
        "accelerator": "gpu" if resolve_device(config.neural_device) == "cuda" else "cpu",
        "enable_progress_bar": False,
        "logger": False,
        **dict(candidate.default_params),
    }
    is_auto = "auto." in path or cls_name.startswith("Auto")
    if is_auto:
        budget = config.neural_search_budget_seconds
        # neuralforecast Auto: num_samples (bütçe varsa dakikaya böl, [4, 40] arası)
        kwargs = {"h": h, "num_samples": 10 if budget is None else max(4, min(40, budget // 60))}
    elif candidate.key in _MULTIVARIATE:
        kwargs["n_series"] = n_series
    try:
        return cls(**kwargs)
    except TypeError:
        # bilinmeyen kwarg (sürüm farkı) — minimal setle dene
        minimal = {"h": h} if is_auto else {"h": h, "input_size": _input_size(h, season)}
        return cls(**minimal)


def _adaptive_windows(lengths: pd.Series, h: int, season: int, max_folds: int) -> tuple[int, int]:
    guard = 2 * max(season, 2)
    for w in range(max_folds, 0, -1):
        req = w * h + guard + _input_size(h, season)
        if float((lengths >= req).mean()) >= 0.6 or w == 1:
            return w, req
    return 1, h + guard + _input_size(h, season)


def run_neural_ts_reports(
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    candidates: list[Candidate],
) -> tuple[list[ValidationReport], list[Candidate]]:
    """Nöral-TS adaylar → `NeuralForecast.cross_validation` OOF → per-model `ValidationReport`."""
    neural = [c for c in candidates if is_neural_ts(c)]
    if not neural or not neural_ts_available():
        if neural:
            logger.warning("[neural_ts] neuralforecast yok — nöral-TS modelleri atlandı")
        return [], []

    from neuralforecast import NeuralForecast

    configure_torch(config.seed, config.neural_determinism, config.neural_device)
    ndf = _to_nixtla(frame, task)
    freq = _resolve_freq(profile)
    season = _season_length(profile, freq)
    h = _horizon(task, config)

    lengths = ndf.groupby("unique_id").size()
    max_folds = min(_MAX_CV_WINDOWS, max(1, config.validation.default_rolling_folds))
    n_windows, needed = _adaptive_windows(lengths, h, season, max_folds)
    keep = set(lengths[lengths >= needed].index)
    n_short = len(lengths) - len(keep)
    if n_short:
        logger.info("[neural_ts] %d/%d seri CV için çok kısa (gerekli≥%d)", n_short, len(lengths), needed)
    cv_df = ndf[ndf["unique_id"].isin(keep)].reset_index(drop=True)
    n_series = int(cv_df["unique_id"].nunique())
    if n_series < config.neural_ts_min_series:
        logger.warning(
            "[neural_ts] %d seri < neural_ts_min_series=%d — atlandı",
            n_series, config.neural_ts_min_series,
        )
        return [], []

    models: list[Any] = []
    alias_to_cand: dict[str, Candidate] = {}
    for cand in neural:
        try:
            m = _build_nf_model(cand, h, season, n_series, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[neural_ts] `%s` kurulamadı: %s", cand.key, exc)
            continue
        models.append(m)
        alias_to_cand[getattr(m, "alias", None) or type(m).__name__] = cand
    if not models:
        return [], []

    try:
        with quiet_cwd():
            nf = NeuralForecast(models=models, freq=freq)
            cv = nf.cross_validation(df=cv_df, n_windows=n_windows, step_size=h)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[neural_ts] cross_validation başarısız: %s", exc)
        return [], []

    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cv["_win"] = cv.groupby("unique_id")["cutoff"].rank(method="dense").astype(int)
    y_true = cv["y"].to_numpy(dtype=np.float64)
    group = cv["unique_id"].to_numpy().astype(object)
    reports = [
        _report_from_oof(cand.key, y_true, cv[alias].to_numpy(dtype=np.float64), cv["_win"], group, task)
        for alias, cand in alias_to_cand.items()
        if alias in cv.columns
    ]
    logger.info("[neural_ts] %d model doğrulandı (h=%d, %d pencere, %d seri)", len(reports), h, n_windows, n_series)
    return reports, []


class FittedNeuralForecaster:
    """Fitted `NeuralForecast` (tek model) + serving. `Predictor` protokolü (ADR 0032)."""

    __slots__ = ("_freq", "_group_col", "_h", "_last", "_nf", "_time_col", "_train_end")

    def __init__(
        self, *, nf: Any, horizon: int, train_ndf: pd.DataFrame,
        group_col: str | None, time_col: str, freq: str,
    ) -> None:
        self._nf = nf
        self._h = horizon
        self._group_col = group_col
        self._time_col = time_col
        self._freq = freq
        self._last: dict[str, float] = {
            str(k): float(v) for k, v in train_ndf.groupby("unique_id")["y"].last().items()
        }
        self._train_end = pd.Timestamp(pd.to_datetime(train_ndf["ds"]).max())

    def _steps_needed(self, target_max: pd.Timestamp | None) -> int:
        if target_max is None or pd.isna(target_max) or target_max <= self._train_end:
            return self._h
        try:
            steps = len(pd.date_range(self._train_end, target_max, freq=self._freq)) - 1
        except (ValueError, TypeError):
            return self._h
        return int(min(max(self._h, steps), self._h * 24 + 366))

    def predict(self, frame: pd.DataFrame) -> _Arr:
        tgt = pd.to_datetime(frame[self._time_col], errors="coerce")
        # neuralforecast `predict` h'yi modelden alır — fit sonrası h adım. İstenen pencere daha
        # ileride ise fallback (son değer) devreye girer (v1; ADR 0032 kapsam dışı: rolling refit).
        with quiet_cwd():
            fc = self._nf.predict().reset_index()
        fc["ds"] = pd.to_datetime(fc["ds"])
        model_col = next((c for c in fc.columns if c not in {"unique_id", "ds", "index"}), None)
        fc = fc.rename(columns={model_col: "_yhat"})
        key = pd.DataFrame({
            "unique_id": frame[self._group_col].astype(str) if self._group_col else "series",
            "ds": tgt,
        })
        merged = key.merge(fc[["unique_id", "ds", "_yhat"]], on=["unique_id", "ds"], how="left")
        out = merged["_yhat"].to_numpy(dtype=np.float64)
        gap = np.isnan(out)
        if gap.any():
            fill = merged.loc[gap, "unique_id"].map(self._last).to_numpy(dtype=np.float64)
            out[gap] = np.nan_to_num(fill, nan=0.0)
        return out

    @property
    def feature_cols(self) -> list[str]:
        return []

    def save(self, directory: str) -> None:
        from pathlib import Path

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        with quiet_cwd():
            self._nf.save(path=str(p), overwrite=True, save_dataset=True)
        np.savez(
            p / "_meta.npz",
            h=self._h, freq=self._freq, group_col=self._group_col or "",
            time_col=self._time_col, train_end=self._train_end.isoformat(),
            last_keys=np.array(list(self._last), dtype=object),
            last_vals=np.array(list(self._last.values()), dtype=np.float64),
        )

    def load(self, directory: str) -> FittedNeuralForecaster:
        from pathlib import Path

        from neuralforecast import NeuralForecast

        p = Path(directory)
        with quiet_cwd():
            self._nf = NeuralForecast.load(path=str(p))
        m = np.load(p / "_meta.npz", allow_pickle=True)
        self._h = int(m["h"])
        self._freq = str(m["freq"])
        self._group_col = str(m["group_col"]) or None
        self._time_col = str(m["time_col"])
        self._train_end = pd.Timestamp(str(m["train_end"]))
        self._last = {str(k): float(v) for k, v in zip(m["last_keys"], m["last_vals"], strict=True)}
        return self


_NEURAL_TS_DIR = "champion_neural_ts"  # persistence.bundle sidecar (ADR 0032)


def refit_neural_ts(
    candidate: Candidate, frame: pd.DataFrame, profile: DataProfile, task: TaskSpec, config: RunConfig
) -> FittedNeuralForecaster:
    """Nöral-TS şampiyonu tüm train'de fit."""
    from neuralforecast import NeuralForecast

    configure_torch(config.seed, config.neural_determinism, config.neural_device)
    ndf = _to_nixtla(frame, task)
    freq = _resolve_freq(profile)
    season = _season_length(profile, freq)
    h = _horizon(task, config)
    model = _build_nf_model(candidate, h, season, ndf["unique_id"].nunique(), config)
    with quiet_cwd():
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=ndf)
    return FittedNeuralForecaster(
        nf=nf, horizon=h, train_ndf=ndf, group_col=task.group_col, time_col=task.time_col or "ds", freq=freq
    )
