"""reporters — run report HTML + model card + leaderboard (+ plots) (ADR 0019)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_manifest import RunManifest
from autoragml.engines import InProcessRunner, TabularCoreEngine
from autoragml.io import load_dataset
from autoragml.persistence import create_run_dir
from autoragml.persistence.manifest import build_manifest
from autoragml.persistence.paths import RunPaths
from autoragml.reporters import scoreboard_to_frame, write_reports


def _df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": np.abs(y), **{f"f{i}": x[:, i] for i in range(3)}})


class _Ctx(NamedTuple):
    result: EngineResult
    manifest: RunManifest
    paths: RunPaths


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> _Ctx:
    out = tmp_path_factory.mktemp("rep")
    cfg = resolve_run_config(
        target="y", overrides={"hpo_level": "none", "output_dir": str(out), "project_name": "demo"}
    ).config
    ds = load_dataset(_df(), cfg)
    profile, task = analyze(ds, cfg)
    result = InProcessRunner().run(TabularCoreEngine(), ds, cfg, profile, task)
    paths = create_run_dir(cfg)
    manifest = build_manifest(cfg, ds, result, run_id=paths.root.name, realized_seconds=2.0)
    return _Ctx(result, manifest, paths)


def test_scoreboard_to_frame_sorted_by_primary(ctx: _Ctx) -> None:
    df = scoreboard_to_frame(ctx.result.scoreboard)
    metric = ctx.result.scoreboard.primary_metric
    assert metric in df.columns
    assert df[metric].is_monotonic_increasing  # rmse: küçük iyi


def test_write_reports_always_emits_html_md_csv(ctx: _Ctx) -> None:
    arts = write_reports(ctx.result, ctx.manifest, ctx.paths)
    assert (ctx.paths.reports / "run_report.html").is_file()
    assert (ctx.paths.reports / "model_card.md").is_file()
    assert (ctx.paths.reports / "leaderboard.csv").is_file()
    assert "reports/run_report.html" in arts

    pd.read_csv(ctx.paths.reports / "leaderboard.csv")  # parse edilebilir


def test_html_is_self_contained(ctx: _Ctx) -> None:
    write_reports(ctx.result, ctx.manifest, ctx.paths)
    html = (ctx.paths.reports / "run_report.html").read_text(encoding="utf-8")
    assert ctx.manifest.project_name in html
    assert ctx.result.selection.champion.model_key in html
    for bad in ("http://", "https://", "cdn.", "src=\"//"):
        assert bad not in html


def test_model_card_has_mitchell_sections(ctx: _Ctx) -> None:
    write_reports(ctx.result, ctx.manifest, ctx.paths)
    card = (ctx.paths.reports / "model_card.md").read_text(encoding="utf-8")
    for sec in ("## Model Details", "## Intended Use", "## Training Data", "## Evaluation",
                "## Limitations", "## Ethical Considerations", "## Caveats and Recommendations"):
        assert sec in card
    assert ctx.result.champion.metadata.model_key in card
    assert ctx.manifest.input_fingerprint in card


def test_plots_generated_when_matplotlib_available(ctx: _Ctx) -> None:
    arts = write_reports(ctx.result, ctx.manifest, ctx.paths)
    pngs = [k for k in arts if k.startswith("reports/plots/")]
    assert pngs  # matplotlib dev bağımlılığında
    assert (ctx.paths.reports / "plots" / "leaderboard.png").is_file()
