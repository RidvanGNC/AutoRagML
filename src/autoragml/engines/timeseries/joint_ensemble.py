"""Klasik + reduction ortak forecasting ensemble (ADR 0035 / Parça 2).

`run_classical_reports` bir rolling-origin `cross_validation` ızgarası üretir (`ClassicalCV`).
Reduction modelleri **aynı cutoff'larda** leakage-safe değerlendirilir (her cutoff için
`train ≤ cutoff` fit + hedef pencere predict → `build_reduction_features` `shift ≥ h` sayesinde
hedef satırlar yalnız `≤ cutoff` `y` görür), OOF sütunları ızgaraya eklenir. Tüm aileler
(klasik + reduction) üzerinde **tek GES** → `joint_ensemble` adayı.

Serving: `FittedJointForecaster` — bir `FittedClassicalForecaster` (klasik üyeler + joint
ağırlıkları) + reduction `FittedModelPipeline`'ları; `predict` iki parçayı toplar. joblib-picklable.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import SplitKind
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import FoldReport, LeakageReport, ValidationReport
from autoragml.engines.model_pipeline import Predictor
from autoragml.engines.timeseries.classical import ClassicalCV
from autoragml.engines.timeseries.reduction import build_reduction_features
from autoragml.ensembling.greedy import greedy_selection
from autoragml.logging import get_logger
from autoragml.postprocessors import FittedPostprocessor
from autoragml.scoring.metrics import compute_metrics, default_primary_metric, lower_is_better
from autoragml.validators.frame_ops import OOFArrays, prediction_health

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
JOINT_ENSEMBLE_KEY = "joint_ensemble"
_JOINT_CLASS_PATH = "__joint_ensemble__"
_MIN_MEMBERS = 2


def _reduce_only(frame: pd.DataFrame, task: TaskSpec, horizon: int, season: int) -> pd.DataFrame:
    return build_reduction_features(frame, task, horizon=horizon, season=season)[0]


class FittedJointForecaster:
    """Klasik parça (`FittedClassicalForecaster`, joint ağırlıklı) + reduction parça
    (`FittedModelPipeline`'lar, ağırlıklı) toplamı. `Predictor` protokolü (ADR 0035/P2).

    Klasik forecaster kendi alias'larını joint ağırlıklarıyla blend'ler; reduction üyeleri
    ayrı ayrı ağırlıklı eklenir. Tüm joint ağırlıkları 1'e toplar. joblib-picklable.
    """

    __slots__ = ("_classical", "_postprocessor", "_reduction")

    def __init__(
        self,
        *,
        classical: Predictor | None,
        reduction: Sequence[tuple[Predictor, float]],
        postprocessor: FittedPostprocessor | None = None,
    ) -> None:
        self._classical = classical
        self._reduction = list(reduction)
        self._postprocessor = postprocessor

    def predict(self, frame: pd.DataFrame) -> _Arr:
        total = np.zeros(len(frame), dtype=np.float64)
        if self._classical is not None:
            total = total + np.asarray(self._classical.predict(frame), dtype=np.float64)
        for member, weight in self._reduction:
            total = total + weight * np.asarray(member.predict(frame), dtype=np.float64)
        if self._postprocessor is not None:
            total = self._postprocessor.apply(total)
        return total

    @property
    def feature_cols(self) -> list[str]:
        cols: list[str] = []
        for member, _w in self._reduction:
            cols.extend(getattr(member, "feature_cols", []) or [])
        return list(dict.fromkeys(cols))


def _keyed(gcol_vals: np.ndarray, ds_vals: np.ndarray) -> np.ndarray:
    left = pd.Series(gcol_vals).astype(str)
    right = pd.to_datetime(pd.Series(ds_vals)).astype(str)
    return (left + "|" + right).to_numpy()


def _reduction_cutoff_oof(
    cand: Candidate,
    choices: dict[str, str],
    raw_frame: pd.DataFrame,
    cv: pd.DataFrame,
    plan: AdaptivePlan,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    h: int,
    season: int,
) -> _Arr | None:
    """`cand`'ı **pencere indeksi başına** (klasik `_win`) leakage-safe değerlendir → `cv` OOF.

    Heterojen panelde cutoff'lar seriye göre değişir → tarih başına değil, pencere başına
    grupla: her seri kendi w'inci cutoff'una kadar train edilir (per-seri kesme), tek model
    tüm panelde fit, her serinin w'inci penceresi predict edilir. `build_reduction_features`
    `shift ≥ h` → hedef satırlar yalnız kendi cutoff'undan önceki `y`'yi görür.
    """
    from autoragml.engines.champion import _ctx, _fit_one

    gcol, tcol, tgt = task.group_col, task.time_col or "ds", task.targets[0]
    if not gcol:
        return None  # joint yalnız panel forecasting (klasik CV panel bazlı)
    ctx = _ctx(task, profile, config)
    red_partial = functools.partial(_reduce_only, task=task, horizon=h, season=season)
    raw = raw_frame.copy()
    raw[tcol] = pd.to_datetime(raw[tcol], errors="coerce")
    raw_g = raw[gcol].astype(str)

    out = np.full(len(cv), np.nan, dtype=np.float64)
    for win, block in cv.groupby("_win", sort=True):
        cut_by_uid = (
            block.groupby(block["unique_id"].astype(str))["cutoff"]
            .first().apply(pd.Timestamp).to_dict()
        )
        parts = [
            raw[(raw_g == uid) & (raw[tcol] <= cut)] for uid, cut in cut_by_uid.items()
        ]
        train_c = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        if train_c[gcol].nunique() < 2:
            continue
        pred_rows = pd.DataFrame({
            gcol: block["unique_id"].to_numpy(),
            tcol: pd.to_datetime(block["ds"]).to_numpy(),
            tgt: np.nan,
        })
        try:
            aug_c = build_reduction_features(train_c, task, horizon=h, season=season)[0]
            fitted, _ = _fit_one(
                cand, choices, {}, aug_c, plan, task, config, ctx,
                fixed_iter=None, pre_transform=red_partial,
            )
            predict_frame = pd.concat([train_c, pred_rows], ignore_index=True)
            preds_all = np.asarray(fitted.predict(predict_frame), dtype=np.float64)
            sorted_frame = red_partial(predict_frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[joint] `%s` pencere %s başarısız: %s", cand.key, win, exc)
            return None
        if len(preds_all) != len(sorted_frame):
            return None
        s = pd.Series(
            preds_all,
            index=_keyed(sorted_frame[gcol].to_numpy(), sorted_frame[tcol].to_numpy()),
        )
        s = s[~s.index.duplicated(keep="last")]
        tgt_key = _keyed(block["unique_id"].to_numpy(), block["ds"].to_numpy())
        out[block.index.to_numpy()] = s.reindex(tgt_key).to_numpy(dtype=np.float64)

    return out if not np.isnan(out).any() else None


def build_joint_forecast_ensemble(
    raw_frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    plan: AdaptivePlan,
    classical_cv: ClassicalCV,
    reduction_candidates: list[Candidate],
) -> tuple[ValidationReport, Candidate] | None:
    """Klasik + reduction OOF'unu ortak cutoff ızgarasında birleştir → tek GES → `joint_ensemble`."""
    cv = classical_cv.cv.reset_index(drop=True)
    h, season = classical_cv.horizon, classical_cv.season
    y_true = cv["y"].to_numpy(dtype=np.float64)
    group = cv["unique_id"].to_numpy().astype(object)

    choices = {g.group_name: g.default for g in plan.candidate_ops}
    cols: dict[str, _Arr] = {}
    kinds: dict[str, str] = {}

    for alias, key in classical_cv.alias_to_key.items():
        if alias in cv.columns:
            col = cv[alias].to_numpy(dtype=np.float64)
            if np.isfinite(col).all():
                cols[key] = col
                kinds[key] = "classical"

    for cand in reduction_candidates:
        oof = _reduction_cutoff_oof(cand, choices, raw_frame, cv, plan, profile, task, config, h, season)
        if oof is not None and np.isfinite(oof).all():
            cols[cand.key] = oof
            kinds[cand.key] = "reduction"

    if len(cols) < _MIN_MEMBERS or sum(1 for v in kinds.values() if v == "reduction") == 0:
        logger.info("[joint] yeterli hizalı üye yok (klasik=%d, reduction=%d) — joint atlandı",
                    sum(v == "classical" for v in kinds.values()),
                    sum(v == "reduction" for v in kinds.values()))
        return None

    keys = list(cols)
    preds = np.column_stack([cols[k] for k in keys])
    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)

    def metric_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return compute_metrics(yt, yp, task.task).get(primary, float("inf"))

    ec = config.ensemble
    w = greedy_selection(
        preds, y_true, metric_fn=metric_fn, lower_is_better=lower,
        max_models=ec.max_models, sorted_init_k=ec.sorted_init_k,
    )
    nz = np.flatnonzero(w > 1e-9)
    if nz.size < _MIN_MEMBERS:
        logger.info("[joint] GES tek üyeye indi — joint atlandı")
        return None
    weights = (w[nz] / w[nz].sum())
    member_keys = [keys[i] for i in nz]
    blend = preds[:, nz] @ weights

    win = cv["_win"]
    windows = sorted(pd.unique(win))
    folds = [
        FoldReport(
            fold_id=int(wi), n_train=0, n_test=int((win == wi).sum()),
            metrics=compute_metrics(y_true[(win == wi).to_numpy()], blend[(win == wi).to_numpy()], task.task),
        )
        for wi in windows
    ]
    oof_metrics = compute_metrics(y_true, blend, task.task)
    oof_se: dict[str, float] = {}
    if len(folds) >= 2:
        for k in oof_metrics:
            vals = [fr.metrics[k] for fr in folds if k in fr.metrics and np.isfinite(fr.metrics[k])]
            if len(vals) >= 2:
                oof_se[k] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    report = ValidationReport(
        candidate_key=JOINT_ENSEMBLE_KEY,
        split_kind=SplitKind.ROLLING_ORIGIN,
        folds=folds,
        oof_metrics=oof_metrics,
        oof_metric_se=oof_se,
        prediction_health=prediction_health(y_true, blend),
        leakage=LeakageReport(),
        nested=True,
        oof=OOFArrays(y_true=y_true, y_pred=blend, group=group),
    )
    members = {k: float(weights[j]) for j, k in enumerate(member_keys)}
    cand = Candidate(
        key=JOINT_ENSEMBLE_KEY,
        name="Joint Forecast Ensemble (klasik+reduction)",
        family="ensemble",
        class_path=_JOINT_CLASS_PATH,
        modalities=[task.modality],
        tasks=[task.task],
        ensemble_members=members,
        default_params={"member_kinds": {k: kinds[k] for k in member_keys}},
    )
    logger.info(
        "[joint] joint_ensemble: %d üye (%d klasik + %d reduction), OOF %s=%.4g",
        len(member_keys),
        sum(kinds[k] == "classical" for k in member_keys),
        sum(kinds[k] == "reduction" for k in member_keys),
        primary, oof_metrics.get(primary, float("nan")),
    )
    return report, cand
