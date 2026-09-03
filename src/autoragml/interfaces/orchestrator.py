"""Orchestrator — uçtan uca akış (ADR 0020).

config → io → analyzers → (holdout carve) → engine → holdout skorla (bir kez) →
persistence → reporters → tracking → RunResult. io/analyze fail-fast; engine failure graceful.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoragml.analyzers import analyze
from autoragml.contracts.config_resolution import ConfigResolution
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.enums import EngineStatus, StageStatus
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.run_manifest import TimelineEntry
from autoragml.contracts.run_result import RunResult
from autoragml.engines import InProcessRunner, select_engine
from autoragml.engines.runners.base import EngineRunner
from autoragml.interfaces.holdout import score_holdout, split_holdout
from autoragml.io import load_dataset, materialize_frame
from autoragml.logging import get_logger
from autoragml.persistence import create_run_dir, persist_run, write_manifest
from autoragml.reporters import write_reports
from autoragml.tracking import resolve_tracker
from autoragml.tracking.base import Tracker

logger = get_logger(__name__)


class Orchestrator:
    """Tek koşum akışı. `runner` enjekte edilebilir (v1: InProcess)."""

    def __init__(self, runner: EngineRunner | None = None) -> None:
        self._runner = runner or InProcessRunner()

    def run(
        self,
        data: Any,
        config: RunConfig,
        *,
        resolution: ConfigResolution | None = None,
        tracker: Tracker | None = None,
    ) -> RunResult:
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        timeline: list[TimelineEntry] = []

        paths = create_run_dir(config)
        tracker = tracker or resolve_tracker(config, run_dir=paths.root)
        tracker.start_run(paths.root.name, project=config.project_name, config=config.model_dump(mode="json"))

        with self._stage("io", timeline):
            dataset_full = load_dataset(data, config)
        with self._stage("analyze", timeline):
            profile, task = analyze(dataset_full, config)

        frame_full = materialize_frame(dataset_full)
        with self._stage("holdout_split", timeline):
            hsplit = split_holdout(frame_full, config, task)

        train_dataset = load_dataset(hsplit.train, config) if hsplit is not None else dataset_full
        with self._stage("engine", timeline):
            engine = select_engine(task, config)
            engine_result = self._runner.run(engine, train_dataset, config, profile, task)

        with self._stage("holdout_score", timeline) as st:
            if hsplit is not None and engine_result.champion.pipeline is not None:
                metrics = score_holdout(engine_result.champion.pipeline, hsplit, task)
                engine_result.champion.metrics_holdout = metrics
                logger.info("[orchestrator] nihai holdout (%d satır): %s", hsplit.n_holdout, metrics)
            else:
                st.detail = "holdout yok veya pipeline bellekte değil"

        with self._stage("finalize", timeline) as st:
            can_finalize = (
                hsplit is not None
                and config.champion_refit_full
                and engine_result.finalize is not None
                and engine_result.status is not EngineStatus.FAILED
            )
            if can_finalize:
                holdout_metrics = dict(engine_result.champion.metrics_holdout)
                try:
                    full_champ = engine_result.finalize(frame_full)
                    full_champ.metrics_holdout = holdout_metrics
                    engine_result.champion = full_champ
                    st.detail = (
                        f"şampiyon `{full_champ.metadata.model_key}` full veride "
                        f"({len(frame_full)} satır) yeniden fit edildi"
                    )
                    logger.info("[orchestrator] champion_refit_full: %s", st.detail)
                except Exception as exc:  # noqa: BLE001 — refit çökerse train-şampiyonu korunur
                    st.detail = f"refit-on-full atlandı (train-şampiyonu korundu): {exc}"
                    logger.warning("[orchestrator] champion_refit_full başarısız: %s", exc)
            else:
                st.detail = "refit-on-full yok (holdout yok / kapalı / engine desteklemiyor)"

        elapsed = time.perf_counter() - t0
        with self._stage("persist", timeline):
            _, manifest = persist_run(
                config,
                dataset_full,
                engine_result,
                paths=paths,
                timeline=timeline,
                resolution=resolution,
                realized_seconds=elapsed,
                started_at=started,
            )
        with self._stage("report", timeline):
            artifacts = write_reports(engine_result, manifest, paths)

        # persist/report aşamaları da manifest'e girsin — tam timeline ile yeniden yaz.
        manifest.timeline = list(timeline)
        manifest.artifacts = {**manifest.artifacts, **artifacts}
        write_manifest(manifest, paths.root)

        self._track_results(tracker, config, engine_result, manifest.artifacts, paths.root)
        status = "failed" if engine_result.status is EngineStatus.FAILED else "ok"
        tracker.end_run(status=status)

        return RunResult(engine_result=engine_result, manifest=manifest, reports_dir=paths.reports)

    @contextmanager
    def _stage(self, name: str, timeline: list[TimelineEntry]) -> Iterator[TimelineEntry]:
        entry = TimelineEntry(stage=name, start=datetime.now(UTC).isoformat())
        try:
            yield entry
        except Exception as exc:
            entry.end = datetime.now(UTC).isoformat()
            entry.status = StageStatus.FAILED
            entry.detail = f"{type(exc).__name__}: {exc}"
            timeline.append(entry)
            logger.error("[orchestrator] stage %s başarısız: %s", name, exc)
            raise
        entry.end = datetime.now(UTC).isoformat()
        timeline.append(entry)

    @staticmethod
    def _track_results(
        tracker: Tracker,
        config: RunConfig,
        engine_result: EngineResult,
        artifacts: dict[str, str],
        run_dir: Path,
    ) -> None:
        champ = engine_result.selection.champion
        tracker.log_params(
            {
                "engine": engine_result.engine_key,
                "champion": champ.model_key,
                "selection_rule": str(engine_result.selection.selection_rule),
                "hpo_level": str(config.hpo_level),
                "primary_metric": engine_result.scoreboard.primary_metric,
            }
        )
        metrics: dict[str, float] = {
            f"oof::{k}": v for k, v in engine_result.champion.metrics_oof.items()
        }
        metrics.update(
            {f"holdout::{k}": v for k, v in engine_result.champion.metrics_holdout.items()}
        )
        if metrics:
            tracker.log_metrics(metrics)
        for rel in artifacts:
            tracker.log_artifact(run_dir / rel, name=rel)
