"""Segmented champion — çekirdek pipeline segment başına + serving yönlendirme (ADR 0028).

`AdaptivePlan.structure == "per_group_champion"` + `plan.segments` → engine her segment
için `run_core_pipeline` koşar; `FittedSegmentedPipeline` her seriyi grup kimliğiyle kendi
segmentinin fitted pipeline'ına yönlendirir. `EngineResult`/`ModelBundle` sözleşmesi değişmez
(şampiyon tek bundle, `pipeline` alanı Predictor protokolü).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.scoreboard import (
    ChampionInfo,
    PromotionResult,
    ScoreBoard,
    ScoreRow,
    SelectionResult,
)
from autoragml.contracts.task_spec import TaskSpec
from autoragml.exceptions import EngineError
from autoragml.logging import get_logger

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]

# segment adı -> o segmentte tam çekirdek pipeline koşan ve EngineResult döndüren fonksiyon
PooledRunner = Callable[[pd.DataFrame, DataProfile], EngineResult]


class FittedSegmentedPipeline:
    """Segment alt-pipeline'ları + `group → segment` yönlendirme. `Predictor` protokolü."""

    __slots__ = ("_fallback", "_group_col", "_group_to_seg", "_members")

    def __init__(
        self,
        *,
        members: dict[str, Any],
        group_to_seg: dict[str, str],
        fallback: str,
        group_col: str,
    ) -> None:
        self._members = members
        self._group_to_seg = group_to_seg
        self._fallback = fallback
        self._group_col = group_col

    def predict(self, frame: pd.DataFrame) -> _Arr:
        groups = frame[self._group_col].astype(str)
        seg_of = groups.map(lambda g: self._group_to_seg.get(g, self._fallback)).to_numpy()
        out = np.full(len(frame), np.nan, dtype=np.float64)
        for name, member in self._members.items():
            mask = seg_of == name
            if not mask.any():
                continue
            preds = np.asarray(member.predict(frame.iloc[mask]), dtype=np.float64)
            out[mask] = preds
        return out

    @property
    def feature_cols(self) -> list[str]:
        cols: list[str] = []
        for member in self._members.values():
            cols.extend(getattr(member, "feature_cols", []) or [])
        return list(dict.fromkeys(cols))

    @property
    def members(self) -> dict[str, Any]:
        return dict(self._members)


def _combined_scoreboard(per_seg: dict[str, EngineResult], sizes: dict[str, int]) -> ScoreBoard:
    first = next(iter(per_seg.values())).scoreboard
    primary = first.primary_metric
    # sentetik "segmented" satırı — downstream (reporter/benchmark) tek şampiyon satırı bekler
    seg_metrics = _weighted_metrics(per_seg, sizes)
    seg_row = ScoreRow(
        model_key="segmented",
        family="segmented",
        oof_metric_mean=seg_metrics.get(primary, float("inf")),
        oof_metric_se=0.0,
        all_metrics_mean={k: v for k, v in seg_metrics.items() if not k.startswith("_")},
        selection_eligible=True,
    )
    rows: list[ScoreRow] = [seg_row]
    for name, er in per_seg.items():
        for row in er.scoreboard.rows:
            rows.append(row.model_copy(update={"model_key": f"{name}::{row.model_key}"}))
    return ScoreBoard(
        rows=rows,
        primary_metric=primary,
        noise_floor=float(np.mean([er.scoreboard.noise_floor for er in per_seg.values()])),
        n_candidates=len(rows),
        selection_bias_bound=max(er.scoreboard.selection_bias_bound for er in per_seg.values()),
        comparison_tests=None,
    )


def _weighted_metrics(per_seg: dict[str, EngineResult], sizes: dict[str, int]) -> dict[str, float]:
    total = sum(sizes.values()) or 1
    keys: set[str] = set()
    for er in per_seg.values():
        keys |= set(er.champion.metrics_oof)
    out: dict[str, float] = {}
    for k in keys:
        acc, wsum = 0.0, 0
        for name, er in per_seg.items():
            v = er.champion.metrics_oof.get(k)
            if v is not None and np.isfinite(v):
                acc += v * sizes[name]
                wsum += sizes[name]
        if wsum:
            out[k] = acc / wsum
    out["_segment_weight_total"] = float(total)
    return out


def run_segmented(
    engine_key: str,
    run_pooled: PooledRunner,
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    plan: AdaptivePlan,
    *,
    subset_profile: Callable[[DataProfile, set[str]], DataProfile],
) -> EngineResult:
    """Her segment için `run_pooled` → `FittedSegmentedPipeline` + birleşik `EngineResult`."""
    gc = task.group_col
    if not gc:
        msg = "segmented engine: group_col zorunlu"
        raise EngineError(msg)

    gcol = frame[gc].astype(str)
    per_seg: dict[str, EngineResult] = {}
    sizes: dict[str, int] = {}
    group_to_seg: dict[str, str] = {}
    messages: list[str] = []

    for seg in plan.segments:
        ids = set(seg.group_ids)
        seg_frame = frame[gcol.isin(ids)].reset_index(drop=True)
        n_series = seg_frame[gc].nunique()
        if seg_frame.empty or n_series < 2:
            messages.append(f"segment '{seg.name}' atlandı (seri yok/yetersiz) — fallback'e katıldı")
            continue
        logger.info("[segmented] '%s': %d seri, %d satır", seg.name, n_series, len(seg_frame))
        try:
            er = run_pooled(seg_frame, subset_profile(profile, ids))
        except EngineError as exc:
            messages.append(f"segment '{seg.name}' başarısız ({exc}) — atlandı")
            logger.warning("[segmented] '%s' başarısız: %s", seg.name, exc)
            continue
        per_seg[seg.name] = er
        sizes[seg.name] = len(seg_frame)
        for g in seg.group_ids:
            group_to_seg[g] = seg.name
        messages.append(
            f"segment '{seg.name}' ({n_series} seri): şampiyon {er.champion.metadata.model_key}"
        )

    if not per_seg:
        msg = "segmented engine: hiçbir segment doğrulanamadı"
        raise EngineError(msg)

    fallback = max(sizes, key=lambda k: sizes[k])
    pipeline = FittedSegmentedPipeline(
        members={name: er.champion.pipeline for name, er in per_seg.items()},
        group_to_seg=group_to_seg,
        fallback=fallback,
        group_col=gc,
    )

    largest = per_seg[fallback].champion.metadata
    metadata = BundleMetadata(
        feature_cols=pipeline.feature_cols,
        feature_set_hash=largest.feature_set_hash,
        target_col=task.targets[0],
        model_key="segmented",
        scenario=largest.scenario,
        best_iteration=None,
        adaptive_plan_summary={
            "structure": "segmented",
            "segments": {name: er.champion.metadata.model_key for name, er in per_seg.items()},
            "fallback": fallback,
        },
        params={"source": plan.segments[0].source if plan.segments else "intermittency_class"},
    )
    champion = ModelBundle(
        metadata=metadata,
        metrics_oof=_weighted_metrics(per_seg, sizes),
        pipeline=pipeline,
    )

    scoreboard = _combined_scoreboard(per_seg, sizes)
    promo_passed = all(er.selection.promotion.passed for er in per_seg.values())
    selection = SelectionResult(
        scoreboard=scoreboard,
        champion=ChampionInfo(
            model_key="segmented",
            scenario=largest.scenario,
            reason=f"{len(per_seg)} segment (intermittency sınıfı) — her biri kendi şampiyonu",
            within_1se=[],
        ),
        promotion=PromotionResult(
            passed=promo_passed,
            reasons=[] if promo_passed else ["bir veya daha çok segment promotion'ı geçemedi"],
        ),
    )

    status = (
        EngineStatus.PARTIAL
        if any(er.status is EngineStatus.PARTIAL for er in per_seg.values())
        or len(per_seg) < len(plan.segments)
        else EngineStatus.SUCCESS
    )
    return EngineResult(
        engine_key=engine_key,
        status=status,
        scoreboard=scoreboard,
        selection=selection,
        champion=champion,
        data_profile=profile,
        task_spec=task,
        adaptive_plan=plan,
        messages=[f"segmented: {len(per_seg)} segment", *messages],
    )
