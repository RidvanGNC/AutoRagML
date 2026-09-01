"""persistence — fitted `ModelBundle` serialize/load + `RunManifest` + çıktı klasör düzeni (ADR 0018).

Yan etkisi olan iki katmandan biri (diğeri `tracking`). `interfaces/Orchestrator` bu
fonksiyonları sırayla çağırır; `persist_run` hepsini tek adımda toparlayan kolaylıktır.
"""

from __future__ import annotations

from datetime import datetime

from autoragml.contracts.config_resolution import ConfigResolution
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.run_manifest import RunManifest, TimelineEntry
from autoragml.contracts.validation import ValidationReport
from autoragml.exceptions import PersistenceError
from autoragml.persistence.bundle import load_bundle, save_bundle
from autoragml.persistence.dump import (
    persist_config_snapshot,
    persist_evaluation,
    write_json,
)
from autoragml.persistence.manifest import build_manifest, write_manifest
from autoragml.persistence.paths import RunPaths, create_run_dir, make_run_id

__all__ = [
    "PersistenceError",
    "RunPaths",
    "build_manifest",
    "create_run_dir",
    "load_bundle",
    "make_run_id",
    "persist_config_snapshot",
    "persist_evaluation",
    "persist_run",
    "save_bundle",
    "write_json",
    "write_manifest",
]


def persist_run(
    config: RunConfig,
    dataset: Dataset,
    engine_result: EngineResult,
    *,
    paths: RunPaths | None = None,
    run_id: str | None = None,
    reports: list[ValidationReport] | None = None,
    holdout_metrics: dict[str, float] | None = None,
    timeline: list[TimelineEntry] | None = None,
    warnings: list[str] | None = None,
    realized_seconds: float = 0.0,
    resolution: ConfigResolution | None = None,
    started_at: datetime | None = None,
) -> tuple[RunPaths, RunManifest]:
    """Şampiyon bundle + değerlendirme + config snapshot + manifest → tek koşum dizini."""
    paths = paths or create_run_dir(config, run_id=run_id)
    bundle = engine_result.champion
    artifacts: dict[str, str] = {}

    save_bundle(bundle, paths.models / "champion.joblib")
    artifacts["models/champion.joblib"] = "champion.joblib"
    write_json(
        {
            "metadata": bundle.metadata.model_dump(mode="json"),
            "metrics_oof": dict(bundle.metrics_oof),
            "metrics_holdout": dict(bundle.metrics_holdout),
        },
        paths.models / "champion_metadata.json",
    )
    artifacts["models/champion_metadata.json"] = "champion_metadata.json"

    for rel, name in persist_evaluation(
        engine_result, paths, reports=reports, holdout_metrics=holdout_metrics
    ).items():
        artifacts[rel] = name
    for rel, name in persist_config_snapshot(config, paths, resolution=resolution).items():
        artifacts[rel] = name

    manifest = build_manifest(
        config,
        dataset,
        engine_result,
        run_id=paths.root.name,
        started_at=started_at,
        timeline=timeline,
        warnings=warnings,
        realized_seconds=realized_seconds,
        artifacts=artifacts,
    )
    write_manifest(manifest, paths.root)
    return paths, manifest
