"""Ajan tool şemaları — v2 RAG/agent üst katmanı bunları çağırır (ADR 0020).

v1: yalnız **deklaratif** JSON-schema tanımı; executor yok. v2'de `llm/` + planlayıcı
bu şemaları kullanıp `Orchestrator`'ı çağıracak.
"""

from __future__ import annotations

from typing import Any

RUN_TOOL: dict[str, Any] = {
    "name": "autoragml_run",
    "description": (
        "Bir veri kümesinde uçtan uca deterministik AutoML koşumu yapar; leaderboard, "
        "şampiyon model ve çıktı dizinini döndürür."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "CSV/TSV/Parquet dosya yolu veya dizin"},
            "target": {"type": "string", "description": "Hedef kolon adı"},
            "preset": {"type": "string", "description": "Yerleşik/kullanıcı preset adı (opsiyonel)"},
            "time_col": {"type": "string", "description": "Zaman kolonu (forecasting)"},
            "group_col": {"type": "string", "description": "Seri/grup kolonu (panel)"},
            "output_dir": {"type": "string"},
        },
        "required": ["data", "target"],
    },
}

TOOLS: list[dict[str, Any]] = [RUN_TOOL]

__all__ = ["RUN_TOOL", "TOOLS"]
