"""reporters — koşum raporu + model card + leaderboard (+ opsiyonel grafikler) (ADR 0019).

`write_reports(engine_result, manifest, paths, ...)` → `paths.reports/` içine dosyalar.
HTML+MD+CSV her zaman; grafikler yalnız `[report]` extra varsa. Akışı asla kırmaz.
"""

from __future__ import annotations

from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_manifest import RunManifest
from autoragml.contracts.validation import ValidationReport
from autoragml.persistence.paths import RunPaths
from autoragml.reporters.html import render_run_report_html
from autoragml.reporters.model_card import render_model_card_md
from autoragml.reporters.plots import maybe_plots
from autoragml.reporters.tables import scoreboard_to_frame

__all__ = [
    "render_model_card_md",
    "render_run_report_html",
    "scoreboard_to_frame",
    "write_reports",
]


def write_reports(
    engine_result: EngineResult,
    manifest: RunManifest,
    paths: RunPaths,
    *,
    reports: list[ValidationReport] | None = None,
) -> dict[str, str]:
    """Raporları yaz, artifacts sözlüğü döndür (rel yol → dosya adı)."""
    paths.reports.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    html = render_run_report_html(engine_result, manifest)
    (paths.reports / "run_report.html").write_text(html, encoding="utf-8")
    artifacts["reports/run_report.html"] = "run_report.html"

    card = render_model_card_md(engine_result, manifest)
    (paths.reports / "model_card.md").write_text(card, encoding="utf-8")
    artifacts["reports/model_card.md"] = "model_card.md"

    frame = scoreboard_to_frame(engine_result.scoreboard)
    frame.to_csv(paths.reports / "leaderboard.csv", index=False)
    artifacts["reports/leaderboard.csv"] = "leaderboard.csv"

    for png in maybe_plots(engine_result, paths, reports=reports):
        artifacts[f"reports/plots/{png.name}"] = png.name

    return artifacts
