"""Ortak engine akışı (ADR 0015).

dynamics → models → validators (nested CV + tuner) → scoring → şampiyon refit → EngineResult.
`tabular` ve `timeseries` engine'leri bu akışı paylaşır; TS önce reduction FE ekler.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.model_bundle import ModelBundle
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.dynamics import build_plan
from autoragml.engines.champion import refit_champion
from autoragml.engines.timeseries.classical import is_classical, run_classical_reports
from autoragml.ensembling import build_weighted_ensemble
from autoragml.exceptions import EngineError
from autoragml.fine_tuners import resolve_tuner
from autoragml.logging import get_logger
from autoragml.models import apply_model_hints, resolve_candidates
from autoragml.models.foundation_gate import prepare_foundation_candidates
from autoragml.models.neural_gate import prepare_neural_candidates
from autoragml.scoring import score_reports
from autoragml.validators import Tuner, run_validation_suite

logger = get_logger(__name__)


def run_core_pipeline(
    engine_key: str,
    frame: pd.DataFrame,
    profile: DataProfile,
    task: TaskSpec,
    config: RunConfig,
    *,
    tuner: Tuner | None = None,
    messages: list[str] | None = None,
    pre_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    raw_frame: pd.DataFrame | None = None,
    run_classical: bool = False,
    run_neural_ts: bool = False,
    run_foundation_ts: bool = False,
    recursive: bool = False,
    recursive_season: int = 1,
) -> EngineResult:
    """Standart dynamics→models→validators→scoring→refit akışı.

    `run_classical=True` (TS engine): klasik adaylar (`family∈{statistical,intermittent}`)
    `StatsForecast` native yolundan geçer, reduction adayları normal suite'ten (ADR 0023).

    `recursive=True` (ADR 0026 B): reduction adayları `shift(1)` özelliğiyle 1-adım eğitilir,
    recursive-h rolling-origin CV ile doğrulanır; `weighted_ensemble` devre dışı.
    """
    msgs = list(messages or [])
    degraded = False  # yalnız gerçek sorunlar PARTIAL yapar; ensemble/reduction bilgi amaçlı
    tuner = tuner or resolve_tuner(config)

    plan = build_plan(profile, task, config)
    if plan.structure in {"per_group_champion", "per_series_champion"} and not plan.segments:
        # segmentlenebilir tek anlamlı grup yok → pooled (planlayıcı kararı, degradasyon değil).
        # Çok-segmentli durum TimeSeriesCoreEngine tarafından `run_segmented` ile ele alınır
        # (ADR 0028 kümelenmiş / ADR 0046 seri-başı).
        msgs.append(f"{plan.structure}: tek anlamlı segment — pooled ilerleniyor.")
        logger.info("[engine] %s", msgs[-1])

    candidates = apply_model_hints(resolve_candidates(config, task), plan.model_hints)
    candidates = prepare_neural_candidates(candidates, profile, config)  # ADR 0030 kapısı
    candidates = prepare_foundation_candidates(candidates, profile, task, config)  # ADR 0033 kapısı
    if any(c.family == "neural" and "pytabkit" in c.requires for c in candidates):
        from autoragml.models.torch_env import configure_torch

        configure_torch(config.seed, config.neural_determinism, config.neural_device)
    from autoragml.engines.timeseries.foundation_ts import is_foundation_ts
    from autoragml.engines.timeseries.neural_ts import is_neural_ts

    def _native_panel(c: Candidate) -> bool:
        return (
            (run_classical and is_classical(c))
            or (run_neural_ts and is_neural_ts(c))
            or (run_foundation_ts and is_foundation_ts(c))
        )

    reduction_cands = [c for c in candidates if not _native_panel(c)]
    classical_cands = [c for c in candidates if run_classical and is_classical(c)]
    neural_ts_cands = [c for c in candidates if run_neural_ts and is_neural_ts(c)]
    foundation_ts_cands = [c for c in candidates if run_foundation_ts and is_foundation_ts(c)]
    logger.info(
        "[engine %s] %d reduction + %d klasik + %d nöral-TS + %d foundation-TS aday",
        engine_key, len(reduction_cands), len(classical_cands),
        len(neural_ts_cands), len(foundation_ts_cands),
    )

    if recursive:
        from autoragml.engines.timeseries.recursive import run_recursive_reports

        reports = run_recursive_reports(
            frame, profile, task, config, reduction_cands, plan, season=recursive_season
        )
    else:
        reports = run_validation_suite(
            reduction_cands, frame, plan, profile, task, config, tuner=tuner
        )
    classical_cv = None
    if classical_cands:
        cl_reports, cl_extra_cands, classical_cv = run_classical_reports(
            frame, profile, task, config, classical_cands
        )
        reports += cl_reports
        candidates = [*candidates, *cl_extra_cands]
    if neural_ts_cands:
        from autoragml.engines.timeseries.neural_ts import run_neural_ts_reports

        nts_reports, _ = run_neural_ts_reports(frame, profile, task, config, neural_ts_cands)
        reports += nts_reports
    if foundation_ts_cands:
        from autoragml.engines.timeseries.foundation_ts import run_foundation_ts_reports

        fts_reports, _ = run_foundation_ts_reports(frame, profile, task, config, foundation_ts_cands)
        reports += fts_reports
    if not reports:
        msg = f"{engine_key}: hiçbir aday doğrulanamadı"
        raise EngineError(msg)
    validated = {r.candidate_key for r in reports}
    if len(validated) < len(candidates):
        failed = {c.key for c in candidates} - validated
        msgs.append(f"doğrulanamayan adaylar: {sorted(failed)}")
        degraded = True

    if (
        not recursive
        and config.forecast_joint_ensemble
        and classical_cv is not None
        and reduction_cands
    ):  # ADR 0035/P2: klasik + reduction ortak cutoff ızgarasında tek GES
        from autoragml.engines.timeseries.joint_ensemble import build_joint_forecast_ensemble

        joint = build_joint_forecast_ensemble(
            raw_frame if raw_frame is not None else frame,
            profile, task, config, plan, classical_cv, reduction_cands, reports,
        )
        if joint is not None:
            j_report, j_cand = joint
            reports = [*reports, j_report]
            candidates = [*candidates, j_cand]
            msgs.append(
                f"joint_ensemble: {len(j_cand.ensemble_members or {})} üye "
                "(klasik+reduction ortak GES)"
            )

    if not recursive:  # ADR 0034: L2 stacker adayları — GES/1-SE havuzuna eklenir
        from autoragml.ensembling.stacking import build_stack_layer

        stack_layer = build_stack_layer(reports, candidates, task, config)
        for st_report, st_cand in stack_layer:
            reports = [*reports, st_report]
            candidates = [*candidates, st_cand]
        if stack_layer:
            msgs.append(f"stacking: {len(stack_layer)} L2 stacker (saf) eklendi")

    ensemble = None if recursive else build_weighted_ensemble(reports, candidates, config, task, profile)
    if ensemble is not None:
        ens_report, ens_candidate, ens_spec = ensemble
        reports = [*reports, ens_report]
        candidates = [*candidates, ens_candidate]
        msgs.append(
            f"weighted_ensemble: {len(ens_spec.member_keys)} üye / {ens_spec.base_model_count} taban "
            f"({ens_spec.method})"
        )

    selection = score_reports(reports, candidates, config, task, profile)
    _season = recursive_season if recursive else None

    def _refit(work_frame: pd.DataFrame) -> ModelBundle:
        return refit_champion(
            selection, candidates, reports, work_frame.reset_index(drop=True),
            plan, profile, task, config,
            tuner=tuner, pre_transform=pre_transform, recursive_season=_season,
        )

    champion = _refit(frame)

    def _finalize_on_full(full_frame: pd.DataFrame) -> ModelBundle:
        """ADR 0035: şampiyonu train+holdout (full ham frame) üstünde yeniden fit."""
        aug = pre_transform(full_frame) if pre_transform is not None else full_frame
        return _refit(aug)

    status = EngineStatus.PARTIAL if degraded else EngineStatus.SUCCESS
    return EngineResult(
        engine_key=engine_key,
        status=status,
        scoreboard=selection.scoreboard,
        selection=selection,
        champion=champion,
        data_profile=profile,
        task_spec=task,
        adaptive_plan=plan,
        finalize=_finalize_on_full,
        messages=msgs,
    )
