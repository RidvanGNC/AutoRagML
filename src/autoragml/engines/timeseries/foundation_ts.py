"""Zero-shot foundation forecasting — Amazon `chronos` + Google `timesfm` (ADR 0033 + 0041).

`family == "foundation_ts"` adaylar bir **backend** üzerinden: **fit yok** (pretrained). "CV" =
rolling-origin pencerelerde zero-shot tahmin + skor; serving = tek forward geçiş. Reduction
pipeline'ından geçmez (panel API). `foundation_enabled=auto` iken **yalnız GPU** (kapı
`foundation_gate`'te).

Backend `candidate.default_params["backend"]` ile seçilir: `"chronos"` (varsayılan) | `"timesfm"`.
Her backend `_ForecastBackend.forecast(context_df, h) -> DataFrame[unique_id, ds, _yhat]` sunar.

Serving `FittedFoundationForecaster` joblib-picklable **değil** → `persistence.bundle`
`_FOUNDATION_TS_DIR` sidecar: backend + checkpoint + context geçmişi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
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
from autoragml.models.torch_env import configure_torch, resolve_device

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
FOUNDATION_TS_FAMILY = "foundation_ts"
# ADR 0042: 2 pencere → anlamsız SE + iyimser OOF (m3/tourism benchmark: OOF 8, holdout 17).
# Zero-shot = eğitim yok → ek pencere yalnız 1 forward geçiş; klasik ile aynı (4) pencere.
_MAX_CV_WINDOWS = 4
_FOUNDATION_TS_DIR = "champion_foundation_ts"
_QUANTILES = [0.1, 0.5, 0.9]


def is_foundation_ts(candidate: Candidate) -> bool:
    return candidate.family == FOUNDATION_TS_FAMILY


def _backend_of(candidate: Candidate) -> str:
    return str(candidate.default_params.get("backend", "chronos"))


def foundation_ts_available(backend: str = "chronos") -> bool:
    import importlib.util

    return importlib.util.find_spec("chronos" if backend == "chronos" else "timesfm") is not None


# --- backend adaptörleri ------------------------------------------------


class _ForecastBackend(Protocol):
    checkpoint: str

    def forecast(self, context: pd.DataFrame, h: int) -> pd.DataFrame: ...


class _ChronosBackend:
    """Amazon Chronos — `predict_df` (Bolt / Chronos-2 oto-ayrım)."""

    def __init__(self, pipeline: Any, checkpoint: str, device: str) -> None:
        self._p = pipeline
        self.checkpoint = checkpoint
        self.device = device

    @classmethod
    def load(cls, checkpoint: str, device: str) -> _ChronosBackend:
        from chronos import BaseChronosPipeline

        return cls(BaseChronosPipeline.from_pretrained(checkpoint, device_map=device), checkpoint, device)

    def forecast(self, context: pd.DataFrame, h: int) -> pd.DataFrame:
        try:
            out = self._p.predict_df(
                context, prediction_length=h, quantile_levels=_QUANTILES,
                id_column="unique_id", timestamp_column="ds", target="y",
            )
        except TypeError:
            out = self._p.predict_df(context, prediction_length=h)
        out = out.rename(columns={c: str(c) for c in out.columns})
        ycol = "predictions" if "predictions" in out.columns else ("0.5" if "0.5" in out.columns else None)
        if ycol is None:
            ycol = next((c for c in out.columns if c not in {"unique_id", "ds", "id", "timestamp"}), None)
        idc = "unique_id" if "unique_id" in out.columns else "id"
        tsc = "ds" if "ds" in out.columns else "timestamp"
        return pd.DataFrame({
            "unique_id": out[idc].astype(str).to_numpy(),
            "ds": pd.to_datetime(out[tsc]).to_numpy(),
            "_yhat": pd.to_numeric(out[ycol], errors="coerce").to_numpy(dtype=np.float64),
        })


class _TimesFMBackend:
    """Google TimesFM 2.5 — `forecast(horizon, inputs=[np.ndarray])` (df API'si yok, elle eşleme)."""

    def __init__(self, model: Any, checkpoint: str, device: str) -> None:
        self._m = model
        self.checkpoint = checkpoint
        self.device = device

    @classmethod
    def load(cls, checkpoint: str, device: str, *, max_horizon: int) -> _TimesFMBackend:
        import timesfm

        cls_map = {"google/timesfm-2.5-200m-pytorch": timesfm.TimesFM_2p5_200M_torch}
        model_cls = cls_map.get(checkpoint, timesfm.TimesFM_2p5_200M_torch)
        model = model_cls.from_pretrained(checkpoint)
        model.compile(timesfm.ForecastConfig(
            max_context=2048, max_horizon=max(int(max_horizon), 64),
            normalize_inputs=True, use_continuous_quantile_head=True, infer_is_positive=True,
        ))
        return cls(model, checkpoint, device)

    def forecast(self, context: pd.DataFrame, h: int) -> pd.DataFrame:
        groups = list(context.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False))
        inputs = [g["y"].to_numpy(dtype=np.float64) for _, g in groups]
        point, _q = self._m.forecast(horizon=h, inputs=inputs)  # (n, h)
        point = np.asarray(point, dtype=np.float64)
        rows_uid: list[str] = []
        rows_ds: list[pd.Timestamp] = []
        rows_y: list[float] = []
        for i, (uid, g) in enumerate(groups):
            ds = pd.to_datetime(g["ds"])
            step = ds.sort_values().diff().dropna().median()
            last = ds.max()
            for k in range(h):
                rows_uid.append(str(uid))
                rows_ds.append(last + step * (k + 1))
                rows_y.append(float(point[i, k]) if k < point.shape[1] else float(point[i, -1]))
        return pd.DataFrame({"unique_id": rows_uid, "ds": rows_ds, "_yhat": rows_y})


def _load_backend(candidate: Candidate, config: RunConfig, *, max_horizon: int) -> _ForecastBackend:
    configure_torch(config.seed, config.neural_determinism, config.foundation_device)
    device = "cuda" if resolve_device(config.foundation_device) == "cuda" else "cpu"
    backend = _backend_of(candidate)
    if backend == "timesfm":
        ckpt = str(candidate.default_params.get("checkpoint", "google/timesfm-2.5-200m-pytorch"))
        return _TimesFMBackend.load(ckpt, device, max_horizon=max_horizon)
    ckpt = str(candidate.default_params.get("checkpoint", "amazon/chronos-bolt-base"))
    return _ChronosBackend.load(ckpt, device)


def _reload_backend(backend: str, checkpoint: str, device: str, *, max_horizon: int) -> _ForecastBackend:
    if backend == "timesfm":
        return _TimesFMBackend.load(checkpoint, device, max_horizon=max_horizon)
    return _ChronosBackend.load(checkpoint, device)


# --- CV + serving -----------------------------------------------------


def _adaptive_windows(lengths: pd.Series, h: int, season: int, max_folds: int) -> tuple[int, int]:
    guard = 2 * max(season, 2)
    for w in range(max_folds, 0, -1):
        req = w * h + guard
        if float((lengths >= req).mean()) >= 0.6 or w == 1:
            return w, req
    return 1, h + guard


def run_foundation_ts_reports(
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    candidates: list[Candidate],
) -> tuple[list[ValidationReport], list[Candidate]]:
    """Foundation-TS adaylar → rolling-origin zero-shot OOF → per-model rapor."""
    fnd = [c for c in candidates if is_foundation_ts(c)]
    if not fnd:
        return [], []

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
        logger.info("[foundation_ts] %d/%d seri CV için çok kısa (gerekli≥%d)", n_short, len(lengths), needed)
    cv_df = ndf[ndf["unique_id"].isin(keep)].reset_index(drop=True)
    if cv_df.empty:
        logger.warning("[foundation_ts] CV için yeterli geçmiş yok — atlandı")
        return [], []

    reports: list[ValidationReport] = []
    for cand in fnd:
        if not foundation_ts_available(_backend_of(cand)):
            logger.warning("[foundation_ts] `%s` backend kurulu değil — atlandı", cand.key)
            continue
        try:
            backend = _load_backend(cand, config, max_horizon=h * n_windows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[foundation_ts] `%s` yüklenemedi: %s", cand.key, exc)
            continue
        oof = _rolling_oof(backend, cv_df, h, n_windows)
        if oof is None:
            continue
        y_true, y_pred, win, group = oof
        reports.append(_report_from_oof(cand.key, y_true, y_pred, pd.Series(win), group, task))
    logger.info("[foundation_ts] %d model doğrulandı (h=%d, %d pencere)", len(reports), h, n_windows)
    return reports, []


def _rolling_oof(
    backend: _ForecastBackend, cv_df: pd.DataFrame, h: int, n_windows: int
) -> tuple[_Arr, _Arr, np.ndarray, np.ndarray] | None:
    grouped = {uid: g.reset_index(drop=True) for uid, g in cv_df.groupby("unique_id")}
    yt: list[float] = []
    yp: list[float] = []
    wins: list[int] = []
    groups: list[str] = []
    for wi in range(n_windows):
        k = n_windows - wi
        ctx_parts: list[pd.DataFrame] = []
        actual_parts: list[pd.DataFrame] = []
        for uid, g in grouped.items():
            cut = len(g) - k * h
            if cut < h:
                continue
            ctx_parts.append(g.iloc[:cut])
            actual_parts.append(g.iloc[cut : cut + h].assign(unique_id=uid))
        if not ctx_parts:
            continue
        context = pd.concat(ctx_parts, ignore_index=True)
        actual = pd.concat(actual_parts, ignore_index=True)
        try:
            fc = backend.forecast(context, h)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[foundation_ts] forecast başarısız (pencere %d): %s", wi, exc)
            return None
        merged = actual[["unique_id", "ds", "y"]].merge(fc, on=["unique_id", "ds"], how="inner")
        yt.extend(merged["y"].astype(float).tolist())
        yp.extend(merged["_yhat"].astype(float).tolist())
        wins.extend([wi + 1] * len(merged))
        groups.extend(merged["unique_id"].tolist())
    if not yt:
        return None
    return (
        np.asarray(yt, dtype=np.float64),
        np.asarray(yp, dtype=np.float64),
        np.asarray(wins, dtype=int),
        np.asarray(groups, dtype=object),
    )


class FittedFoundationForecaster:
    """Fitted (zero-shot) foundation forecaster + serving. `Predictor` protokolü (ADR 0033/0041)."""

    __slots__ = (
        "_backend", "_backend_name", "_checkpoint", "_context", "_device", "_freq",
        "_group_col", "_h", "_last", "_time_col", "_train_end",
    )

    def __init__(
        self, *, backend: _ForecastBackend, backend_name: str, checkpoint: str, device: str,
        horizon: int, context: pd.DataFrame, group_col: str | None, time_col: str, freq: str,
    ) -> None:
        self._backend = backend
        self._backend_name = backend_name
        self._checkpoint = checkpoint
        self._device = device
        self._h = horizon
        self._context = context[["unique_id", "ds", "y"]].copy()
        self._group_col = group_col
        self._time_col = time_col
        self._freq = freq
        self._last = {str(k): float(v) for k, v in context.groupby("unique_id")["y"].last().items()}
        self._train_end = pd.Timestamp(pd.to_datetime(context["ds"]).max())

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
        steps = self._steps_needed(tgt.max())
        try:
            fc = self._backend.forecast(self._context, steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[foundation_ts] serving forecast başarısız: %s — fallback", exc)
            fc = pd.DataFrame(columns=["unique_id", "ds", "_yhat"])
        key = pd.DataFrame({
            "unique_id": frame[self._group_col].astype(str) if self._group_col else "series",
            "ds": tgt,
        })
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

    def save(self, directory: str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        self._context.to_parquet(p / "context.parquet", index=False)
        np.savez(
            p / "_meta.npz",
            backend=self._backend_name, checkpoint=self._checkpoint, device=self._device,
            h=self._h, freq=self._freq, group_col=self._group_col or "", time_col=self._time_col,
            train_end=self._train_end.isoformat(),
            last_keys=np.array(list(self._last), dtype=object),
            last_vals=np.array(list(self._last.values()), dtype=np.float64),
        )

    def load(self, directory: str) -> FittedFoundationForecaster:
        p = Path(directory)
        m = np.load(p / "_meta.npz", allow_pickle=True)
        self._backend_name = str(m["backend"]) if "backend" in m else "chronos"
        self._checkpoint = str(m["checkpoint"])
        self._device = str(m["device"])
        self._h = int(m["h"])
        self._freq = str(m["freq"])
        self._group_col = str(m["group_col"]) or None
        self._time_col = str(m["time_col"])
        self._train_end = pd.Timestamp(str(m["train_end"]))
        self._last = {str(k): float(v) for k, v in zip(m["last_keys"], m["last_vals"], strict=True)}
        self._context = pd.read_parquet(p / "context.parquet")
        self._backend = _reload_backend(
            self._backend_name, self._checkpoint, self._device, max_horizon=self._h * 4
        )
        return self


def refit_foundation_ts(
    candidate: Candidate, frame: pd.DataFrame, profile: DataProfile, task: TaskSpec, config: RunConfig
) -> FittedFoundationForecaster:
    """Foundation-TS şampiyonu → context'i tüm-veri sonuna kaydır (zero-shot — fit yok)."""
    h = _horizon(task, config)
    backend = _load_backend(candidate, config, max_horizon=h * 4)
    ndf = _to_nixtla(frame, task)
    return FittedFoundationForecaster(
        backend=backend,
        backend_name=_backend_of(candidate),
        checkpoint=backend.checkpoint,
        device="cuda" if resolve_device(config.foundation_device) == "cuda" else "cpu",
        horizon=h,
        context=ndf,
        group_col=task.group_col,
        time_col=task.time_col or "ds",
        freq=_resolve_freq(profile),
    )
