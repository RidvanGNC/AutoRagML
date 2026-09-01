"""Tracker protokolü — deney takibi soyutlaması (ADR 0019).

NVIDIA FLARE `LogWriter` deseni: tek arayüz, çoklu writer. Çekirdek yalnız bu
protokole bağlıdır; implementasyonlar `tracking/` altında.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Deney takip sink'i. Tüm metotlar idempotent-güvenli olmalı."""

    def start_run(self, run_id: str, *, project: str, config: dict[str, Any]) -> None:
        """Koşum başlangıcı — run_id + proje + (sırsız) config snapshot."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Girdi yapılandırması / hiperparametreler."""
        ...

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Sonuç metrikleri (opsiyonel adım indeksi)."""
        ...

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """Bir çıktı dosyasını kaydet/işaretle."""
        ...

    def end_run(self, *, status: str = "ok") -> None:
        """Koşum sonu — durum: ok | failed | partial."""
        ...
