"""Zero-shot foundation forecasting — Amazon `chronos` (ADR 0033).

`family == "foundation_ts"` adaylar Chronos üzerinden: **fit yok** (pretrained). "CV" =
rolling-origin pencerelerde `predict_df` zero-shot tahmin + skor; serving = tek `predict_df`.
Reduction pipeline'ından geçmez (panel API). ADR 0023/0032 deseninin ikizi.
`foundation_enabled=auto` iken **yalnız GPU** (kapı `foundation_gate`'te).

Serving `FittedChronosForecaster` joblib-picklable **değil** → `persistence.bundle`
`_FOUNDATION_TS_DIR` sidecar: checkpoint adı + context geçmişi (`from_pretrained` ile geri kurulur).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
_MAX_CV_WINDOWS = 2  # zero-shot ama panel-boyu forward geçiş — 2 pencere yeterli
_FOUNDATION_TS_DIR = "champion_foundation_ts"  # persistence.bundle sidecar


def is_foundation_ts(candidate: Candidate) -> bool:
    return candidate.family == FOUNDATION_TS_FAMILY


def foundation_ts_available() -> bool:
    try:
        import chronos  # noqa: F401
    except ImportError:
        return False
    return True


def _load_pipeline(candidate: Candidate, config: RunConfig) -> Any:
    """`chronos` pipeline'ını checkpoint'ten kur (Bolt / Chronos-2 otomatik ayrımı)."""
    from chronos import BaseChronosPipeline

    configure_torch(config.seed, config.neural_determinism, config.foundation_device)
    checkpoint = str(candidate.default_params.get("checkpoint", "amazon/chronos-bolt-base"))
    device = "cuda" if resolve_device(config.foundation_device) == "cuda" else "cpu"
    return BaseChronosPipeline.from_pretrained(checkpoint, device_map=device)


def _predict_df(pipeline: Any, context: pd.DataFrame, h: int) -> pd.DataFrame:
    """`predict_df` sarımı — (unique_id, ds, _yhat) döndürür. Sürüm/imza farklarına dayanıklı."""
    try:
        out = pipeline.predict_df(
            context, prediction_length=h, quantile_levels=[0.1, 0.5, 0.9],
            id_column="unique_id", timestamp_column="ds", target="y",
        )
    except TypeError:
        out = pipeline.predict_df(context, prediction_length=h)
    out = out.rename(columns={c: str(c) for c in out.columns})
    ycol = "predictions" if "predictions" in out.columns else ("0.5" if "0.5" in out.columns else None)
    if ycol is None:
        ycol = next((c for c in out.columns if c not in {"unique_id", "ds", "id", "timestamp"}), None)
    idc = "unique_id" if "unique_id" in out.columns else "id"
    tsc = "ds" if "ds" in out.columns else "timestamp"
    res = pd.DataFrame({
        "unique_id": out[idc].astype(str).to_numpy(),
        "ds": pd.to_datetime(out[tsc]).to_numpy(),
        "_yhat": pd.to_numeric(out[ycol], errors="coerce").to_numpy(dtype=np.float64),
    })
    return res


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
    """Foundation-TS adaylar → rolling-origin zero-shot `predict_df` OOF → per-model rapor."""
    fnd = [c for c in candidates if is_foundation_ts(c)]
    if not fnd:
        return [], []
    if not foundation_ts_available():
        logger.warning("[foundation_ts] chronos yok — foundation-TS modelleri atlandı")
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
        try:
            pipeline = _load_pipeline(cand, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[foundation_ts] `%s` yüklenemedi: %s", cand.key, exc)
            continue
        oof = _rolling_oof(pipeline, cv_df, h, n_windows)
        if oof is None:
            continue
        y_true, y_pred, win, group = oof
        reports.append(_report_from_oof(cand.key, y_true, y_pred, pd.Series(win), group, task))
    logger.info("[foundation_ts] %d model doğrulandı (h=%d, %d pencere)", len(reports), h, n_windows)
    return reports, []


def _rolling_oof(
    pipeline: Any, cv_df: pd.DataFrame, h: int, n_windows: int
) -> tuple[_Arr, _Arr, np.ndarray, np.ndarray] | None:
    """Panel üstünde `n_windows` rolling-origin penceresi → OOF dizileri."""
    grouped = {uid: g.reset_index(drop=True) for uid, g in cv_df.groupby("unique_id")}
    yt: list[float] = []
    yp: list[float] = []
    wins: list[int] = []
    groups: list[str] = []
    for wi in range(n_windows):
        k = n_windows - wi  # pencere wi: seri sonundan k·h önce keser
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
            fc = _predict_df(pipeline, context, h)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[foundation_ts] predict_df başarısız (pencere %d): %s", wi, exc)
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


class FittedChronosForecaster:
    """Fitted (zero-shot) Chronos + serving. `Predictor` protokolü (ADR 0033)."""

    __slots__ = (
        "_checkpoint", "_context", "_device", "_freq", "_group_col", "_h",
        "_last", "_pipeline", "_time_col", "_train_end",
    )

    def __init__(
        self, *, pipeline: Any, checkpoint: str, device: str, horizon: int,
        context: pd.DataFrame, group_col: str | None, time_col: str, freq: str,
    ) -> None:
        self._pipeline = pipeline
        self._checkpoint = checkpoint
        self._device = device
        self._h = horizon
        self._context = context[["unique_id", "ds", "y"]].copy()
        self._group_col = group_col
        self._time_col = time_col
        self._freq = freq
        self._last = {
            str(k): float(v) for k, v in context.groupby("unique_id")["y"].last().items()
        }
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
            fc = _predict_df(self._pipeline, self._context, steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[foundation_ts] serving predict_df başarısız: %s — fallback", exc)
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
            checkpoint=self._checkpoint, device=self._device, h=self._h, freq=self._freq,
            group_col=self._group_col or "", time_col=self._time_col,
            train_end=self._train_end.isoformat(),
            last_keys=np.array(list(self._last), dtype=object),
            last_vals=np.array(list(self._last.values()), dtype=np.float64),
        )

    def load(self, directory: str) -> FittedChronosForecaster:
        from chronos import BaseChronosPipeline

        p = Path(directory)
        m = np.load(p / "_meta.npz", allow_pickle=True)
        self._checkpoint = str(m["checkpoint"])
        self._device = str(m["device"])
        self._h = int(m["h"])
        self._freq = str(m["freq"])
        self._group_col = str(m["group_col"]) or None
        self._time_col = str(m["time_col"])
        self._train_end = pd.Timestamp(str(m["train_end"]))
        self._last = {str(k): float(v) for k, v in zip(m["last_keys"], m["last_vals"], strict=True)}
        self._context = pd.read_parquet(p / "context.parquet")
        self._pipeline = BaseChronosPipeline.from_pretrained(self._checkpoint, device_map=self._device)
        return self


def refit_foundation_ts(
    candidate: Candidate, frame: pd.DataFrame, profile: DataProfile, task: TaskSpec, config: RunConfig
) -> FittedChronosForecaster:
    """Foundation-TS şampiyonu → context'i tüm-veri sonuna kaydır (zero-shot — fit yok)."""
    pipeline = _load_pipeline(candidate, config)
    ndf = _to_nixtla(frame, task)
    freq = _resolve_freq(profile)
    return FittedChronosForecaster(
        pipeline=pipeline,
        checkpoint=str(candidate.default_params.get("checkpoint", "amazon/chronos-bolt-base")),
        device="cuda" if resolve_device(config.foundation_device) == "cuda" else "cpu",
        horizon=_horizon(task, config),
        context=ndf,
        group_col=task.group_col,
        time_col=task.time_col or "ds",
        freq=freq,
    )
