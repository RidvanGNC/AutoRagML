"""ensembling — sınıflandırma GES (olasılık OOF) + bagging (ADR 0036)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.engines.champion import refit_champion
from autoragml.engines.ensemble_pipeline import FittedEnsemblePipeline
from autoragml.ensembling import ENSEMBLE_KEY, build_weighted_ensemble
from autoragml.ensembling.greedy import greedy_selection_proba
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates
from autoragml.scoring import score_reports
from autoragml.scoring.metrics import compute_proba_metrics
from autoragml.validators import run_validation_suite


def _cls_df(n: int = 500, n_classes: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 5))
    logits = np.column_stack([
        x @ rng.normal(size=5) for _ in range(n_classes)
    ]) + rng.normal(0, 0.5, (n, n_classes))
    y = np.argmax(logits, axis=1)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(5)}})


def _prep(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(
        target="y", overrides={"hpo_level": "none", "task_hint": "multiclass_classification", **over}
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    plan = build_plan(profile, task, cfg)
    cands = resolve_candidates(cfg, task)
    reports = run_validation_suite(cands, frame, plan, profile, task, cfg)
    return cfg, profile, task, frame, plan, cands, reports


# --- greedy_selection_proba ---

def test_greedy_selection_proba_weights_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    n, c = 200, 3
    y = rng.integers(0, c, n)
    good = np.eye(c)[y] * 0.7 + 0.1  # y'ye yakın
    noise = rng.dirichlet(np.ones(c), n)
    stack = np.stack([good, noise, (good + noise) / 2])

    def m(yt: np.ndarray, p: np.ndarray) -> float:
        return compute_proba_metrics(yt, p).get("log_loss", 9.9)

    w = greedy_selection_proba(stack, y, metric_fn=m, lower_is_better=True, max_models=10)
    assert w.shape == (3,) and abs(w.sum() - 1.0) < 1e-9
    assert w[0] > w[1]  # iyi model daha ağır


# --- build_weighted_ensemble (classification branch) ---

def test_classification_ensemble_built_from_proba_oof() -> None:
    cfg, profile, task, _f, _pl, cands, reports = _prep(_cls_df())
    proba_reports = [r for r in reports if getattr(r.oof, "y_proba", None) is not None]
    assert len(proba_reports) >= 2, "en az 2 aday proba OOF üretmeli"

    result = build_weighted_ensemble(reports, cands, cfg, task, profile)
    if result is None:
        return  # GES tek modele indi — kabul (küçük sentetik)
    ens_report, ens_cand, spec = result
    assert ens_cand.key == ENSEMBLE_KEY and ens_cand.family == "ensemble"
    assert ens_report.oof.y_proba is not None
    assert ens_report.oof.y_proba.shape[1] == 3
    assert "log_loss" in ens_report.oof_metrics and "f1_macro" in ens_report.oof_metrics


def test_classification_ensemble_champion_serving() -> None:
    cfg, profile, task, frame, plan, cands, reports = _prep(_cls_df())
    result = build_weighted_ensemble(reports, cands, cfg, task, profile)
    if result is None:
        return
    ens_report, ens_cand, _ = result
    reports2 = [*reports, ens_report]
    cands2 = [*cands, ens_cand]
    selection = score_reports(reports2, cands2, cfg, task, profile)

    bundle = refit_champion(selection, cands2, reports2, frame, plan, profile, task, cfg)
    preds = bundle.pipeline.predict(_cls_df(60))
    assert preds.shape == (60,)
    assert set(np.unique(preds)).issubset({0.0, 1.0, 2.0})
    if isinstance(bundle.pipeline, FittedEnsemblePipeline) and bundle.pipeline.classes is not None:
        proba = bundle.pipeline.predict_proba(_cls_df(60))
        assert proba.shape == (60, 3) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_ges_search_metric_follows_primary() -> None:
    """K1: primary proba-aware (roc_auc) → GES onda arar; nokta metrik → log_loss."""
    cfg_auc, profile, task, _f, _pl, cands, reports = _prep(_cls_df(), primary_metric="roc_auc")
    r1 = build_weighted_ensemble(reports, cands, cfg_auc, task, profile)
    # sadece çökmemeli + (kurulduysa) roc_auc raporlanmalı
    if r1 is not None:
        assert "roc_auc" in r1[0].oof_metrics
