"""engines — weighted_ensemble akış entegrasyonu (ADR 0021)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.engines import InProcessRunner, TabularCoreEngine
from autoragml.engines.champion import refit_champion
from autoragml.engines.ensemble_pipeline import FittedEnsemblePipeline
from autoragml.ensembling import build_weighted_ensemble
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates
from autoragml.scoring import score_reports
from autoragml.validators import run_validation_suite


def _df(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 4))
    y = x @ np.array([1.4, -1.8, 0.7, 0.3]) + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(4)}})


def _prep(**ov):
    cfg = resolve_run_config(target="y", overrides={"hpo_level": "none", **ov}).config
    df = _df()
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    return df, cfg, ds, profile, task


def test_ensemble_row_appears_in_scoreboard() -> None:
    df, cfg, ds, profile, task = _prep()
    result = InProcessRunner().run(TabularCoreEngine(), ds, cfg, profile, task)
    keys = [r.model_key for r in result.scoreboard.rows]
    assert "weighted_ensemble" in keys
    ens_row = next(r for r in result.scoreboard.rows if r.model_key == "weighted_ensemble")
    assert ens_row.family == "ensemble"
    assert any("weighted_ensemble" in m for m in result.messages)


def test_disabled_ensemble_has_no_row() -> None:
    df, cfg, ds, profile, task = _prep(ensemble={"enabled": False})
    result = InProcessRunner().run(TabularCoreEngine(), ds, cfg, profile, task)
    assert "weighted_ensemble" not in [r.model_key for r in result.scoreboard.rows]


def test_ensemble_champion_refits_to_ensemble_pipeline() -> None:
    df, cfg, ds, profile, task = _prep(ensemble={"bagging": False})
    frame = materialize_frame(ds)
    plan = build_plan(profile, task, cfg)
    candidates = resolve_candidates(cfg, task)
    reports = run_validation_suite(candidates, frame, plan, profile, task, cfg)

    built = build_weighted_ensemble(reports, candidates, cfg, task, profile)
    assert built is not None
    ens_report, ens_candidate, _ = built
    all_reports = [*reports, ens_report]
    all_cands = [*candidates, ens_candidate]

    selection = score_reports(all_reports, all_cands, cfg, task, profile)
    # şampiyonu zorla ensemble yap
    selection.champion.model_key = "weighted_ensemble"

    bundle = refit_champion(selection, all_cands, all_reports, frame, plan, profile, task, cfg)
    assert isinstance(bundle.pipeline, FittedEnsemblePipeline)
    assert bundle.metadata.model_key == "weighted_ensemble"
    assert bundle.metadata.ensemble["members"]
    assert abs(sum(bundle.metadata.ensemble["members"].values()) - 1.0) < 1e-9

    pred = bundle.pipeline.predict(_df(25))
    assert pred.shape == (25,)
    assert not np.isnan(pred).any()
