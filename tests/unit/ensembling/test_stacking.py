"""ensembling.stacking — saf L2 stacker katmanı (ADR 0034)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.engines.champion import refit_champion
from autoragml.engines.stack_pipeline import FittedStackPipeline
from autoragml.ensembling.stacking import STACK_PREFIX, build_stack_layer, is_stack
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates
from autoragml.scoring import score_reports
from autoragml.validators import run_validation_suite


def _reg_df(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 5))
    # doğrusal + hafif etkileşim → farklı L1 aileleri farklı hata yapar
    y = x @ np.array([1.3, -1.7, 0.6, 0.2, -0.9]) + 0.5 * x[:, 0] * x[:, 1] + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(5)}})


def _prep(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(
        target="y", overrides={"hpo_level": "none", "stacking_enabled": "on", **over}
    ).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    plan = build_plan(profile, task, cfg)
    candidates = resolve_candidates(cfg, task)
    reports = run_validation_suite(candidates, frame, plan, profile, task, cfg)
    return cfg, profile, task, frame, plan, candidates, reports


def test_is_stack_and_prefix() -> None:
    cfg, _p, task, _f, _pl, candidates, reports = _prep(_reg_df())
    layer = build_stack_layer(reports, candidates, task, cfg)
    assert layer, "en az bir L2 stacker beklenir"
    for rep, cand in layer:
        assert is_stack(cand)
        assert cand.key.startswith(STACK_PREFIX)
        assert cand.class_path == "__stack__"
        assert rep.candidate_key == cand.key
        assert rep.oof is not None and rep.oof.y_pred.shape == reports[0].oof.y_true.shape


def test_stack_layer_off() -> None:
    cfg, _p, task, _f, _pl, candidates, reports = _prep(_reg_df(), stacking_enabled="off")
    assert build_stack_layer(reports, candidates, task, cfg) == []


def test_stack_auto_gate_skips_small_data() -> None:
    # auto + 300 satır < stacking_min_rows(2000) → atlanır
    cfg, _p, task, _f, _pl, candidates, reports = _prep(_reg_df(300), stacking_enabled="auto")
    assert build_stack_layer(reports, candidates, task, cfg) == []


def test_stack_guard_requires_beating_best_l1() -> None:
    from autoragml.ensembling import _aligned_reports  # noqa: PLC2701
    from autoragml.ensembling.stacking import _EXCLUDED_FAMILIES  # noqa: PLC2701

    cfg, _p, task, _f, _pl, candidates, reports = _prep(_reg_df())
    primary = "rmse"
    by_key = {c.key: c for c in candidates}
    # build_stack_layer ile aynı taban kümesi: hizalı + dışlanmayan aileler
    base_reps = [
        r for r in _aligned_reports(reports)
        if r.candidate_key in by_key and by_key[r.candidate_key].family not in _EXCLUDED_FAMILIES
    ]
    best_l1 = min(r.oof_metrics[primary] for r in base_reps if primary in r.oof_metrics)
    layer = build_stack_layer(reports, candidates, task, cfg)
    for rep, _cand in layer:
        assert rep.oof_metrics[primary] < best_l1  # guard: her stacker L1 en iyisini geçer


def test_stack_champion_refit_roundtrip(tmp_path) -> None:
    from autoragml.persistence.bundle import load_bundle, save_bundle

    cfg, profile, task, frame, plan, candidates, reports = _prep(_reg_df())
    layer = build_stack_layer(reports, candidates, task, cfg)
    reports = [*reports, *[r for r, _ in layer]]
    candidates = [*candidates, *[c for _, c in layer]]

    selection = score_reports(reports, candidates, cfg, task, profile)
    bundle = refit_champion(selection, candidates, reports, frame, plan, profile, task, cfg)

    # şampiyon stacker ise → FittedStackPipeline; değilse en azından çöküş yok
    if bundle.metadata.model_key.startswith(STACK_PREFIX):
        assert isinstance(bundle.pipeline, FittedStackPipeline)
        preds = bundle.pipeline.predict(frame)
        assert preds.shape == (len(frame),) and np.isfinite(preds).all()
        dest = tmp_path / "champion.joblib"
        save_bundle(bundle, dest)
        got = load_bundle(dest).pipeline.predict(frame)
        assert np.allclose(got, preds, atol=1e-6)


def test_stack_forced_champion_predict() -> None:
    """stack_<x>'i zorla şampiyon yap → refit + serving tutarlı."""
    cfg, profile, task, frame, plan, candidates, reports = _prep(_reg_df())
    layer = build_stack_layer(reports, candidates, task, cfg)
    assert layer
    st_report, st_cand = layer[0]
    reports2 = [*reports, st_report]
    candidates2 = [*candidates, st_cand]
    selection = score_reports(reports2, candidates2, cfg, task, profile)
    # scoreboard'da stacker satırı olmalı
    assert any(row.model_key == st_cand.key for row in selection.scoreboard.rows)

    from autoragml.engines.champion import _stack_bundle
    from autoragml.validators import DefaultTuner

    bundle = _stack_bundle(
        st_cand, selection, candidates2, reports2, frame.reset_index(drop=True),
        plan, profile, task, cfg, DefaultTuner(), None,
    )
    assert isinstance(bundle.pipeline, FittedStackPipeline)
    preds = bundle.pipeline.predict(frame)
    assert preds.shape == (len(frame),) and np.isfinite(preds).all()
