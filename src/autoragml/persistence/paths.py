"""Çıktı klasör düzeni — `<output_dir>/<DDMMYYYY>_<proje>_outputs/<run_id>/` (ADR 0018).

`create_run_dir` sessizce üzerine yazmaz: aynı saniyede çakışan `run_id` → artan sonek;
dolu hedef dizin + `exist_ok=False` → `PersistenceError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from autoragml.contracts.run_config import RunConfig
from autoragml.exceptions import PersistenceError

_SUBDIRS = ("models", "evaluation", "reports", "config_snapshot", "tracking")


@dataclass(frozen=True)
class RunPaths:
    """Bir koşumun disk yolları."""

    root: Path
    models: Path
    evaluation: Path
    reports: Path
    config_snapshot: Path
    tracking: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def make_run_id(now: datetime | None = None) -> str:
    """UTC `%Y%m%dT%H%M%SZ` — sıralanabilir."""
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def _day_folder(project_name: str, now: datetime | None = None) -> str:
    return f"{(now or datetime.now(UTC)).strftime('%d%m%Y')}_{project_name}_outputs"


def create_run_dir(
    config: RunConfig, *, run_id: str | None = None, exist_ok: bool = False, now: datetime | None = None
) -> RunPaths:
    """Koşum kök dizinini + alt klasörleri oluştur, `RunPaths` döndür."""
    stamp = now or datetime.now(UTC)
    parent = Path(config.output_dir) / _day_folder(config.project_name, stamp)
    parent.mkdir(parents=True, exist_ok=True)

    base_id = run_id or make_run_id(stamp)
    root = parent / base_id
    if not exist_ok:
        suffix = 1
        while root.exists() and any(root.iterdir()):
            root = parent / f"{base_id}-{suffix:02d}"
            suffix += 1
            if suffix > 99:
                msg = f"run_id çakışması çözülemedi: {base_id!r}"
                raise PersistenceError(msg)
    elif root.exists() and not root.is_dir():
        msg = f"run kök yolu bir dizin değil: {root}"
        raise PersistenceError(msg)

    root.mkdir(parents=True, exist_ok=True)
    for name in _SUBDIRS:
        (root / name).mkdir(exist_ok=True)

    return RunPaths(
        root=root,
        models=root / "models",
        evaluation=root / "evaluation",
        reports=root / "reports",
        config_snapshot=root / "config_snapshot",
        tracking=root / "tracking",
    )
