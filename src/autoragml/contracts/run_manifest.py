"""RunManifest — bir koşumun tam kaydı. DONDU (ADR 0015).

Tekrarüretilebilirlik: girdi fingerprint + config snapshot + ortam + timeline.
Sırlar maskeli.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import StageStatus


class TimelineEntry(Contract):
    """Bir aşamanın zaman/durum kaydı — 'hangi faz kırıldı' izlenebilirliği."""

    stage: str
    start: str  # ISO 8601
    end: str | None = None
    status: StageStatus = StageStatus.OK
    detail: str | None = None


class EnvInfo(Contract):
    """Çalışma ortamı."""

    python: str
    platform: str
    os: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    git_commit: str | None = None
    # nöral (ADR 0030) — yalnız `[neural]` extra kuruluysa dolar (torch/cuda sürüm + device + determinizm)
    accelerator: dict[str, str] = Field(default_factory=dict)


class DataSnapshot(Contract):
    """Girdi verisinin özeti."""

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    date_min: str | None = None
    date_max: str | None = None
    target_summary: dict[str, float] = Field(default_factory=dict)
    layout: str = "n/a"


class RunManifest(Contract):
    """Koşum manifestosu."""

    run_id: str
    created_at: str  # ISO 8601
    project_name: str
    autoragml_version: str
    input_fingerprint: str
    config_snapshot: dict[str, object] = Field(default_factory=dict)  # sırlar maskeli
    env: EnvInfo
    data_snapshot: DataSnapshot
    seed: int = 42
    timeline: list[TimelineEntry] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    champion_ref: str | None = None
    realized_seconds: float = Field(default=0.0, ge=0.0)
    n_candidates: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
