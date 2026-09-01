"""InProcessRunner — engine'i aynı süreçte koşturur (ADR 0006, v1 varsayılanı)."""

from __future__ import annotations

import time

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.scoreboard import (
    ChampionInfo,
    PromotionResult,
    ScoreBoard,
    SelectionResult,
)
from autoragml.contracts.task_spec import TaskSpec
from autoragml.engines.base import Engine
from autoragml.exceptions import AutoRagMLError
from autoragml.logging import get_logger
from autoragml.scoring.metrics import default_primary_metric

logger = get_logger(__name__)


def _empty_scoreboard(config: RunConfig, task: TaskSpec) -> ScoreBoard:
    return ScoreBoard(
        rows=[],
        primary_metric=config.primary_metric or default_primary_metric(task.task),
        noise_floor=0.0,
        n_candidates=0,
        selection_bias_bound=0.0,
    )


def _failed_result(
    engine_key: str, config: RunConfig, profile: DataProfile, task: TaskSpec, detail: str
) -> EngineResult:
    board = _empty_scoreboard(config, task)
    return EngineResult(
        engine_key=engine_key,
        status=EngineStatus.FAILED,
        scoreboard=board,
        selection=SelectionResult(
            scoreboard=board,
            champion=ChampionInfo(model_key="", reason="engine başarısız"),
            promotion=PromotionResult(passed=False, reasons=["engine başarısız"]),
        ),
        champion=ModelBundle(
            metadata=BundleMetadata(
                feature_cols=[], feature_set_hash="", target_col=task.targets[0], model_key=""
            )
        ),
        data_profile=profile,
        task_spec=task,
        adaptive_plan=AdaptivePlan(),
        messages=[f"engine hatası: {detail}"],
    )


class InProcessRunner:
    """Engine'i doğrudan çağırır. Çökme → `EngineResult(status=FAILED)`."""

    def run(
        self,
        engine: Engine,
        dataset: Dataset,
        config: RunConfig,
        profile: DataProfile,
        task: TaskSpec,
    ) -> EngineResult:
        start = time.perf_counter()
        try:
            result = engine.run(dataset, config, profile, task)
        except AutoRagMLError as exc:
            logger.exception("[runner] engine `%s` başarısız", engine.key)
            return _failed_result(engine.key, config, profile, task, str(exc))
        logger.info("[runner] engine `%s` %.1fs", engine.key, time.perf_counter() - start)
        return result
