"""RunManifest kurulumu + yazımı (ADR 0018).

Reprodüksiyon için yeterli: girdi fingerprint + config snapshot + ortam + seed + timeline.
Sır taşımaz (`RunConfig` yalnız `*_env` adları).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from autoragml import __version__
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.run_manifest import DataSnapshot, EnvInfo, RunManifest, TimelineEntry
from autoragml.persistence.dump import write_json

_ENV_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "scipy",
    "pydantic",
    "pyarrow",
    "joblib",
    "torch",
    "pytabkit",
)


def _accelerator_info(config: object) -> dict[str, str]:
    """Nöral ortam (ADR 0030) — torch yoksa boş."""
    from autoragml.models.torch_env import cuda_device_name, torch_available, torch_versions

    if not torch_available():
        return {}
    vers = torch_versions()
    info = {"torch": vers["torch"] or "?", "cuda_build": vers["cuda"] or "none"}
    dev = cuda_device_name()
    if dev:
        info["device"] = dev
    det = getattr(config, "neural_determinism", None)
    if det:
        info["determinism"] = str(det)
    return info


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _ENV_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:  # pragma: no cover
            continue
    return out


def _git_commit() -> str | None:
    from autoragml import __file__ as pkg_file

    root = Path(pkg_file).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return res.stdout.strip() or None


def _data_snapshot(dataset: Dataset, engine_result: EngineResult) -> DataSnapshot:
    stats = engine_result.data_profile.target_profile.stats
    target_summary = {
        k: float(v)
        for k, v in (
            ("min", stats.min),
            ("max", stats.max),
            ("mean", stats.mean),
            ("std", stats.std),
            ("zero_ratio", stats.zero_ratio),
        )
        if v is not None
    }
    ts = engine_result.data_profile.timeseries
    date_min, date_max = (ts.span if ts and ts.span else (None, None))
    return DataSnapshot(
        n_rows=dataset.shape.n_rows,
        n_cols=dataset.shape.n_cols,
        date_min=date_min,
        date_max=date_max,
        target_summary=target_summary,
        layout=str(dataset.layout),
    )


def build_manifest(
    config: RunConfig,
    dataset: Dataset,
    engine_result: EngineResult,
    *,
    run_id: str,
    started_at: datetime | None = None,
    timeline: list[TimelineEntry] | None = None,
    warnings: list[str] | None = None,
    realized_seconds: float = 0.0,
    artifacts: dict[str, str] | None = None,
    champion_ref: str | None = "models/champion.joblib",
) -> RunManifest:
    """Contract nesnelerinden `RunManifest` kur."""
    env = EnvInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        os=sys.platform,
        package_versions=_package_versions(),
        git_commit=_git_commit(),
        accelerator=_accelerator_info(config),
    )
    return RunManifest(
        run_id=run_id,
        created_at=(started_at or datetime.now(UTC)).isoformat(),
        project_name=config.project_name,
        autoragml_version=__version__,
        input_fingerprint=dataset.fingerprint,
        config_snapshot=config.model_dump(mode="json"),
        env=env,
        data_snapshot=_data_snapshot(dataset, engine_result),
        seed=config.seed,
        timeline=list(timeline or []),
        artifacts=dict(artifacts or {}),
        champion_ref=champion_ref,
        realized_seconds=realized_seconds,
        n_candidates=engine_result.scoreboard.n_candidates,
        warnings=list(warnings or [*engine_result.messages]),
    )


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    """`manifest.json` — koşum kök dizinine."""
    return write_json(manifest, run_dir / "manifest.json")
