"""Sözleşme smoke testleri — her model kurulabiliyor, doğrulama çalışıyor,
round-trip (dump → validate) kimliği koruyor.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoragml import contracts as c
from autoragml.contracts.enums import Modality, Task


def test_runconfig_minimal_defaults() -> None:
    cfg = c.RunConfig(target="y")
    assert cfg.seed == 42
    assert cfg.budget.max_trials_per_model == 15
    assert cfg.selection_rule == c.enums.SelectionRule.ONE_STD_ERR
    assert cfg.tracking.backend == c.enums.TrackingBackend.JSONL


def test_runconfig_forbids_unknown_field() -> None:
    with pytest.raises(ValidationError):
        c.RunConfig(target="y", nonsense=1)  # type: ignore[call-arg]


def test_runconfig_forecasting_requires_time_col() -> None:
    with pytest.raises(ValidationError):
        c.RunConfig(target="y", task_hint=Task.FORECASTING)
    ok = c.RunConfig(target="y", task_hint=Task.FORECASTING, time_col="ds")
    assert ok.time_col == "ds"


def test_runconfig_autopilot_rejected_in_v1() -> None:
    with pytest.raises(ValidationError):
        c.RunConfig(target="y", autopilot=True)


def test_budget_trial_bounds() -> None:
    with pytest.raises(ValidationError):
        c.BudgetConfig(min_trials_per_model=20, max_trials_per_model=10)


def test_runconfig_roundtrip() -> None:
    cfg = c.RunConfig(target="siparis_adeti", time_col="ds", group_col="urun_grubu")
    restored = c.RunConfig.model_validate(cfg.model_dump())
    assert restored == cfg


def test_plancontext_is_frozen() -> None:
    ctx = c.PlanContext(target="y", task=Task.REGRESSION)
    assert ctx.provenance == "train"
    with pytest.raises(ValidationError):
        ctx.target = "z"  # type: ignore[misc]


def test_plancontext_provenance_locked_to_train() -> None:
    with pytest.raises(ValidationError):
        c.PlanContext(target="y", task=Task.REGRESSION, provenance="test")  # type: ignore[arg-type]


def test_taskspec_forecasting_needs_timeseries_modality() -> None:
    with pytest.raises(ValidationError):
        c.TaskSpec(
            task=Task.FORECASTING,
            modality=Modality.TABULAR,
            targets=["y"],
            time_col="ds",
        )
    ok = c.TaskSpec(
        task=Task.FORECASTING,
        modality=Modality.TIMESERIES,
        targets=["y"],
        time_col="ds",
        horizon=4,
    )
    assert ok.horizon == 4


def test_dataset_roundtrip() -> None:
    ds = c.Dataset(
        source=c.DataSource(kind=c.enums.SourceKind.CSV, ref="a.csv"),
        dtypes={"y": "float64", "ds": "datetime64[ns]"},
        shape=c.DatasetShape(n_rows=10, n_cols=2),
        materialization=c.enums.Materialization.EAGER,
        fingerprint="deadbeef",
        fingerprint_spec="strict/multiset-v1",
    )
    dumped = ds.model_dump()
    assert c.Dataset.model_validate(dumped).dtypes == ds.dtypes
    assert c.Dataset.model_validate(dumped).relations is None


def test_scoreboard_composition() -> None:
    row = c.ScoreRow(model_key="lightgbm", oof_metric_mean=12.3, oof_metric_se=0.4)
    board = c.ScoreBoard(
        rows=[row],
        primary_metric="smape",
        noise_floor=0.4,
        n_candidates=1,
        selection_bias_bound=0.0,
    )
    sel = c.SelectionResult(
        scoreboard=board,
        champion=c.ChampionInfo(model_key="lightgbm", reason="tek uygun aday"),
        promotion=c.PromotionResult(passed=True),
    )
    assert sel.champion.model_key == "lightgbm"
    assert sel.selection_rule == c.enums.SelectionRule.ONE_STD_ERR


def test_all_exported_names_importable() -> None:
    for name in c.__all__:
        assert hasattr(c, name), name
