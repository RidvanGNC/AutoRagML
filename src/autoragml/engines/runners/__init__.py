"""engines.runners — engine yürütme stratejileri (ADR 0006).

`InProcessRunner` (v1 varsayılanı). Subprocess (venv izolasyonu) / Container v1.1+.
Sınır formatı Arrow/Parquet — o katmanlar geldiğinde.
"""

from __future__ import annotations

from autoragml.engines.runners.base import EngineRunner
from autoragml.engines.runners.inprocess import InProcessRunner

__all__ = ["EngineRunner", "InProcessRunner"]
