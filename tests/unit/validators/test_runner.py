"""validators.runner — nested CV → ValidationReport (ADR 0010/6 + 0011 + 0013)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.dynamics import build_plan
from autoragml.io import load_dataset
from autoragml.models import resolve_candidates
from autoragml.validators import DefaultTuner, TunerOutcome, run_validation, run_validation_suite


def _prep(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(target="y", overrides=over or None).config
    profile, task = analyze(load_dataset(df, cfg), cfg)
    plan = build_plan(profile, task, cfg)
    return df, cfg, profile, task, plan


def _tabular(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "cat": rng.choice(list("ab"), n)})


def _cand(cfg: RunConfig, task: TaskSpec, key: str) -> Candidate:
    return next(c for c in resolve_candidates(cfg, task) if c.key == key)


def test_tabular_report_shape() -> None:
    df, cfg, profile, task, plan = _prep(_tabular())
    rep = run_validation(_cand(cfg, task, "ridge"), df, plan, profile, task, cfg)
    assert len(rep.folds) == 5
    assert rep.leakage.status == "PASS"
    assert "rmse" in rep.oof_metrics
    assert "rmse" in rep.oof_metric_se
    assert rep.nested is False  # DefaultTuner
    assert rep.realized_seconds >= 0


def test_forecasting_no_time_leakage() -> None:
    weeks = pd.date_range("2023-01-02", periods=150, freq="W-MON")
    rng = np.random.default_rng(1)
    rows = [
        {"g": g, "ds": wk, "y": 100 + 20 * np.sin(i / 52 * 6.28) + rng.normal(0, 4)}
        for g in ["A", "B", "C"]
        for i, wk in enumerate(weeks)
    ]
    df, cfg, profile, task, plan = _prep(pd.DataFrame(rows), time_col="ds", group_col="g")
    rep = run_validation(_cand(cfg, task, "lightgbm"), df, plan, profile, task, cfg)
    assert rep.split_kind.value == "rolling_origin"
    assert rep.leakage.status == "PASS"
    assert all(v.category.value != "overlap" for v in rep.leakage.violations)
    assert rep.folds[0].best_iteration is not None  # lightgbm early stopping


def test_custom_tuner_marks_nested() -> None:
    df, cfg, profile, task, plan = _prep(_tabular(300))

    class _Tuner:
        def tune(
            self,
            candidate: Candidate,
            frame: pd.DataFrame,
            plan: AdaptivePlan,
            task: TaskSpec,
            ctx: PlanContext,
            config: RunConfig,
        ) -> TunerOutcome:
            return TunerOutcome(best_params={"alpha": 2.0}, candidate_choices={}, nested=True)

    rep = run_validation(_cand(cfg, task, "ridge"), df, plan, profile, task, cfg, tuner=_Tuner())
    assert rep.nested is True


def test_suite_skips_failing_candidate() -> None:
    df, cfg, profile, task, plan = _prep(_tabular(300))
    good = _cand(cfg, task, "ridge")
    broken = good.model_copy(update={"key": "broken", "default_params": {"bad_param": 1}})
    reps = run_validation_suite([broken, good], df, plan, profile, task, cfg)
    assert [r.candidate_key for r in reps] == ["ridge"]


def test_default_tuner_uses_plan_defaults() -> None:
    df, cfg, profile, task, plan = _prep(_tabular(200))
    ctx = PlanContext(target="y", task=task.task)
    outcome = DefaultTuner().tune(_cand(cfg, task, "ridge"), df, plan, task, ctx, cfg)
    assert outcome.nested is False
    for group in plan.candidate_ops:
        assert outcome.candidate_choices[group.group_name] == group.default
