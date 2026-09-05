"""config.resolve_run_config — uçtan uca katmanlı çözümleme (ADR 0016)."""

from __future__ import annotations

import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import HpoLevel, SplitKind
from autoragml.exceptions import ConfigError


def test_minimal_target_only() -> None:
    res = resolve_run_config(target="y")
    assert res.config.target == "y"
    assert res.config.seed == 42
    # target kwarg'ı overrides katmanı olarak uygulanır
    assert res.layers == ["defaults", "overrides"]
    assert res.provenance["target"] == "overrides"
    assert res.provenance["seed"] == "default"


def test_missing_target_raises() -> None:
    with pytest.raises(ConfigError, match="target zorunlu"):
        resolve_run_config(preset="tabular_fast")


def test_preset_then_override() -> None:
    res = resolve_run_config(
        target="y",
        preset="tabular_fast",
        overrides={"split_policy": {"n_folds": 8}},
    )
    assert res.config.split_policy is not None
    assert res.config.split_policy.kind is SplitKind.KFOLD  # preset
    assert res.config.split_policy.n_folds == 8  # override
    assert res.provenance["split_policy.kind"] == "preset:tabular_fast"
    assert res.provenance["split_policy.n_folds"] == "overrides"
    assert "preset:tabular_fast" in res.layers


def test_extends_chain_in_layers() -> None:
    res = resolve_run_config(
        target="siparis_adeti", preset="demandsensing", overrides={"time_col": "ds"}
    )
    assert res.config.project_name == "demandsensing"
    assert res.config.hpo_level is HpoLevel.LIGHT
    assert res.layers == [
        "defaults",
        "preset:timeseries_rolling",
        "preset:demandsensing",
        "overrides",
    ]
    # project_name yaprak preset'ten, split_policy kök preset'ten
    assert res.provenance["project_name"] == "preset:demandsensing"
    assert res.provenance["split_policy.kind"] == "preset:timeseries_rolling"


def test_timeseries_preset_requires_time_col() -> None:
    # task_hint=forecasting (preset) + time_col yok -> RunConfig validator hata
    with pytest.raises(ConfigError, match="doğrulaması başarısız"):
        resolve_run_config(target="y", preset="timeseries_rolling")


def test_target_kwarg_conflicts_with_overrides() -> None:
    with pytest.raises(ConfigError, match="Çelişkili target"):
        resolve_run_config(target="a", overrides={"target": "b"})


# --- hierarchy_cols (ADR 0045) -------------------------------------------------


def test_hierarchy_cols_requires_group_col() -> None:
    with pytest.raises(ConfigError, match="doğrulaması başarısız"):
        resolve_run_config(target="y", overrides={"time_col": "ds", "hierarchy_cols": ["state"]})


def test_hierarchy_cols_rejects_group_col_duplicate() -> None:
    with pytest.raises(ConfigError, match="doğrulaması başarısız"):
        resolve_run_config(
            target="y",
            overrides={"time_col": "ds", "group_col": "region", "hierarchy_cols": ["state", "region"]},
        )


def test_hierarchy_cols_rejects_repeated_cols() -> None:
    with pytest.raises(ConfigError, match="doğrulaması başarısız"):
        resolve_run_config(
            target="y",
            overrides={"time_col": "ds", "group_col": "region", "hierarchy_cols": ["state", "state"]},
        )


def test_hierarchy_cols_accepted_with_group_col() -> None:
    res = resolve_run_config(
        target="y",
        overrides={"time_col": "ds", "group_col": "region", "hierarchy_cols": ["state", "zone"]},
    )
    assert res.config.hierarchy_cols == ["state", "zone"]


def test_hierarchy_reconcile_method_default_ols() -> None:
    """ADR 0047: varsayılan reconciliation yöntemi `ols`."""
    assert resolve_run_config(target="y").config.hierarchy_reconcile_method == "ols"
    res = resolve_run_config(target="y", overrides={"hierarchy_reconcile_method": "wls_struct"})
    assert res.config.hierarchy_reconcile_method == "wls_struct"
    with pytest.raises(ConfigError, match="doğrulaması başarısız"):
        resolve_run_config(target="y", overrides={"hierarchy_reconcile_method": "mint_shrink"})


def test_config_file_layer(tmp_path: object) -> None:
    cfg = tmp_path / "run.yaml"  # type: ignore[operator]
    cfg.write_text("seed: 7\nproject_name: proj\n", encoding="utf-8")
    res = resolve_run_config(target="y", config_file=cfg)
    assert res.config.seed == 7
    assert res.config.project_name == "proj"
    assert res.provenance["seed"].startswith("file:")


def test_unknown_field_rejected() -> None:
    with pytest.raises(ConfigError, match="doğrulaması başarısız"):
        resolve_run_config(target="y", overrides={"bogus_field": 1})
