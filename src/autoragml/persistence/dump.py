"""Contract → JSON dosyası + değerlendirme çıktıları (ADR 0018).

Tüm dump'lar deterministik: `sort_keys=True`, `ensure_ascii=False`, `indent=2`, sonda newline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.validation import ValidationReport
from autoragml.persistence.paths import RunPaths


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def write_json(obj: Any, path: Path) -> Path:
    """Pydantic model veya JSON-uyumlu nesneyi dosyaya yaz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_jsonable(obj), sort_keys=True, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def persist_config_snapshot(config: BaseModel, paths: RunPaths, resolution: BaseModel | None = None) -> dict[str, str]:
    """`config_snapshot/` — çözülmüş RunConfig (+ varsa ConfigResolution). `.env` KOPYALANMAZ."""
    written: dict[str, str] = {}
    p = write_json(config, paths.config_snapshot / "run_config.json")
    written["config_snapshot/run_config.json"] = p.name
    if resolution is not None:
        write_json(resolution, paths.config_snapshot / "config_resolution.json")
        written["config_snapshot/config_resolution.json"] = "config_resolution.json"
    return written


def persist_evaluation(
    engine_result: EngineResult,
    paths: RunPaths,
    *,
    reports: list[ValidationReport] | None = None,
    holdout_metrics: dict[str, float] | None = None,
) -> dict[str, str]:
    """`evaluation/` — scoreboard + selection + karşılaştırma testleri (+ opsiyonel fold/holdout)."""
    written: dict[str, str] = {}
    ev = paths.evaluation

    write_json(engine_result.scoreboard, ev / "scoreboard.json")
    written["evaluation/scoreboard.json"] = "scoreboard.json"

    write_json(engine_result.selection, ev / "selection.json")
    written["evaluation/selection.json"] = "selection.json"

    if engine_result.scoreboard.comparison_tests is not None:
        write_json(engine_result.scoreboard.comparison_tests, ev / "comparison_tests.json")
        written["evaluation/comparison_tests.json"] = "comparison_tests.json"

    if reports:
        # OOF dizileri `ValidationReport.oof` zaten `exclude=True` → model_dump'ta yok.
        payload = [r.model_dump(mode="json") for r in reports]
        write_json(payload, ev / "validation_reports.json")
        written["evaluation/validation_reports.json"] = "validation_reports.json"

    if holdout_metrics:
        write_json(dict(holdout_metrics), ev / "holdout_metrics.json")
        written["evaluation/holdout_metrics.json"] = "holdout_metrics.json"

    return written
