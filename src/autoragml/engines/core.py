"""Ortak engine akışı (ADR 0015).

dynamics → models → validators (nested CV + tuner) → scoring → şampiyon refit → EngineResult.
`tabular` ve `timeseries` engine'leri bu akışı paylaşır; TS önce reduction FE ekler.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.dynamics import build_plan
from autoragml.engines.champion import refit_champion
from autoragml.ensembling import build_weighted_ensemble
from autoragml.exceptions import EngineError
from autoragml.fine_tuners import resolve_tuner
from autoragml.logging import get_logger
from autoragml.models import resolve_candidates
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
) -> EngineResult:
    """Standart dynamics→models→validators→scoring→refit akışı."""
    msgs = list(messages or [])
    degraded = False  # yalnız gerçek sorunlar PARTIAL yapar; ensemble/reduction bilgi amaçlı
    tuner = tuner or resolve_tuner(config)

    plan = build_plan(profile, task, config)
    if plan.structure == "per_group_champion":
        msgs.append("per_group_champion planlandı — v1 pooled ile ilerliyor (per-group refit v1.1).")
        logger.info("[engine] %s", msgs[-1])
        degraded = True

    candidates = resolve_candidates(config, task)
    logger.info("[engine %s] %d aday doğrulanıyor", engine_key, len(candidates))

    reports = run_validation_suite(candidates, frame, plan, profile, task, config, tuner=tuner)
    if not reports:
        msg = f"{engine_key}: hiçbir aday doğrulanamadı"
        raise EngineError(msg)
    if len(reports) < len(candidates):
        failed = {c.key for c in candidates} - {r.candidate_key for r in reports}
        msgs.append(f"doğrulanamayan adaylar: {sorted(failed)}")
        degraded = True

    ensemble = build_weighted_ensemble(reports, candidates, config, task, profile)
    if ensemble is not None:
        ens_report, ens_candidate, ens_spec = ensemble
        reports = [*reports, ens_report]
        candidates = [*candidates, ens_candidate]
        msgs.append(
            f"weighted_ensemble: {len(ens_spec.member_keys)} üye / {ens_spec.base_model_count} taban "
            f"({ens_spec.method})"
        )

    selection = score_reports(reports, candidates, config, task, profile)
    champion = refit_champion(
        selection, candidates, reports, frame, plan, profile, task, config,
        tuner=tuner, pre_transform=pre_transform,
    )

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
        messages=msgs,
    )
