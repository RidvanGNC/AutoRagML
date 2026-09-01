"""fine_tuners — RandomSearchTuner / OptunaTuner / resolve_tuner (ADR 0013)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.enums import HpoBackend
from autoragml.contracts.plan_context import PlanContext
from autoragml.dynamics import build_plan
from autoragml.fine_tuners import OptunaTuner, RandomSearchTuner, resolve_tuner
from autoragml.io import load_dataset
from autoragml.models import resolve_candidates
from autoragml.validators import DefaultTuner, run_validation
from autoragml.validators.frame_ops import column_roles


def _prep(**over: object):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 4))
    y = x @ np.array([1.5, -2.0, 0.5, 0.3]) + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(4)}})
    cfg = resolve_run_config(target="y", overrides={"budget": {"max_trials_per_model": 6}, **over}).config
    profile, task = analyze(load_dataset(df, cfg), cfg)
    plan = build_plan(profile, task, cfg)
    ctx = PlanContext(target="y", task=task.task, column_roles=column_roles(profile))
    return df, cfg, profile, task, plan, ctx


def _cand(cfg, task, key):
    return next(c for c in resolve_candidates(cfg, task) if c.key == key)


def test_resolve_tuner_by_level_and_backend() -> None:
    none_cfg = resolve_run_config(target="y", overrides={"hpo_level": "none"}).config
    assert isinstance(resolve_tuner(none_cfg), DefaultTuner)
    light = resolve_tuner(resolve_run_config(target="y").config)
    assert isinstance(light, RandomSearchTuner) and light.inner_folds == 1
    thorough = resolve_tuner(resolve_run_config(target="y", overrides={"hpo_level": "thorough"}).config)
    assert isinstance(thorough, RandomSearchTuner) and thorough.inner_folds == 3
    opt = resolve_tuner(resolve_run_config(target="y", overrides={"hpo_backend": "optuna"}).config)
    assert isinstance(opt, OptunaTuner)


def test_random_search_sh_path_for_fidelity_model() -> None:
    df, cfg, profile, task, plan, ctx = _prep()
    out = RandomSearchTuner().tune(_cand(cfg, task, "lightgbm"), df, plan, task, ctx, cfg)
    assert out.nested is True
    assert out.tuning_result is not None
    assert out.tuning_result.backend is HpoBackend.RANDOM_SEARCH
    assert out.tuning_result.fidelity_schedule  # SH kullanıldı
    assert len(out.tuning_result.trials) >= 6
    assert "n_estimators" in out.best_params  # fidelity son rung'a sabitlendi


def test_random_search_plain_path_no_fidelity() -> None:
    df, cfg, profile, task, plan, ctx = _prep()
    out = RandomSearchTuner().tune(_cand(cfg, task, "ridge"), df, plan, task, ctx, cfg)
    assert out.nested is True
    assert out.tuning_result is not None
    assert out.tuning_result.fidelity_schedule == []  # düz random search
    assert "alpha" in out.best_params


def test_hpo_none_falls_through() -> None:
    df, cfg, profile, task, plan, ctx = _prep(hpo_level="none")
    out = RandomSearchTuner().tune(_cand(cfg, task, "ridge"), df, plan, task, ctx, cfg)
    assert out.nested is False
    assert out.tuning_result is None


def test_no_search_surface_falls_through() -> None:
    df, cfg, profile, task, plan, ctx = _prep()
    out = RandomSearchTuner().tune(_cand(cfg, task, "linear"), df, plan, task, ctx, cfg)
    assert out.nested is False  # linear: search_space yok, candidate_ops tekil


def test_optuna_tuner_backend_tag_and_choices() -> None:
    df, cfg, profile, task, plan, ctx = _prep()
    out = OptunaTuner().tune(_cand(cfg, task, "lightgbm"), df, plan, task, ctx, cfg)
    assert out.nested is True
    assert out.tuning_result is not None
    assert out.tuning_result.backend is HpoBackend.OPTUNA
    assert all(not k.startswith("__choice__") for k in out.best_params)


def test_tuner_plugs_into_run_validation() -> None:
    df, cfg, profile, task, plan, _ = _prep()
    rep = run_validation(
        _cand(cfg, task, "ridge"), df, plan, profile, task, cfg, tuner=RandomSearchTuner()
    )
    assert rep.nested is True
    assert rep.leakage.status == "PASS"


def test_budget_deadline_stops_early() -> None:
    df, cfg, profile, task, plan, ctx = _prep(
        budget={"max_trials_per_model": 40, "per_model_max_seconds": 1}
    )
    out = RandomSearchTuner().tune(_cand(cfg, task, "ridge"), df, plan, task, ctx, cfg)
    assert out.tuning_result is not None
    assert len(out.tuning_result.trials) < 40  # deadline erken kesti
