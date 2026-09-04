"""models.registry — katalog yükleme + Candidate çözümleme (ADR 0012)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import CandidateSource, Modality, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.models import build_candidates, load_catalog, resolve_candidates
from autoragml.models.registry import ModelCatalogError


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides=over or None).config


def _reg_task(modality: Modality = Modality.TABULAR, task: Task = Task.REGRESSION) -> TaskSpec:
    kw: dict[str, object] = {"task": task, "modality": modality, "targets": ["y"]}
    if task is Task.FORECASTING:
        kw.update(time_col="ds", horizon=4)
    return TaskSpec(**kw)  # type: ignore[arg-type]


def test_builtin_catalog_loads() -> None:
    catalog = load_catalog()
    assert {"linear", "ridge", "lightgbm", "random_forest", "dummy_mean"} <= set(catalog)


def test_lightgbm_resolved_xgboost_skipped() -> None:
    keys = {c.key for c in build_candidates(_cfg())}
    assert "lightgbm" in keys  # çekirdek bağımlılık
    assert "xgboost" not in keys  # opsiyonel, kurulu değil
    assert "catboost" not in keys  # enabled: false


def test_resolve_filters_by_task_and_modality() -> None:
    reg = {c.key for c in resolve_candidates(_cfg(), _reg_task())}
    assert "linear" in reg
    assert "logistic" not in reg  # sınıflandırma modeli
    assert "dummy_prior" not in reg

    clf = {c.key for c in resolve_candidates(_cfg(), _reg_task(task=Task.BINARY_CLASSIFICATION))}
    assert "logistic" in clf
    assert "linear" not in clf


def test_forecasting_uses_reduction_regressors() -> None:
    keys = {c.key for c in resolve_candidates(_cfg(), _reg_task(Modality.TIMESERIES, Task.FORECASTING))}
    assert "lightgbm" in keys
    assert "random_forest" in keys


def test_adr0040_academic_additions_resolve() -> None:
    """ADR 0040: klasik forecasting + EBM/KNN eklemeleri (kurulu extra'larla)."""
    import importlib.util as u

    fc = {c.key for c in resolve_candidates(_cfg(), _reg_task(Modality.TIMESERIES, Task.FORECASTING))}
    assert {"auto_ces", "dynamic_theta", "imapa", "adida"} <= fc  # statsforecast dep var
    assert "knn" in fc  # sklearn — reduction yolu

    reg = {c.key for c in resolve_candidates(_cfg(), _reg_task())}
    assert "knn" in reg
    assert ("ebm" in reg) == (u.find_spec("interpret") is not None)
    # opt-in (enabled:false) → varsayılan havuzda YOK
    assert not {"svr", "svc", "auto_tbats", "ngboost", "tabicl", "tabpfn"} & reg
    assert "auto_tbats" not in fc


def test_no_match_raises() -> None:
    with pytest.raises(ModelCatalogError, match="uygun model adayı yok"):
        resolve_candidates(_cfg(), _reg_task(task=Task.MULTILABEL_CLASSIFICATION))


def test_user_override_disable_and_add(tmp_path: Path) -> None:
    override = tmp_path / "models.yaml"
    override.write_text(
        "linear:\n  enabled: false\n"
        "my_gbm:\n"
        "  name: MyGBM\n"
        "  family: gbdt\n"
        "  class_path: {regression: sklearn.ensemble.GradientBoostingRegressor}\n"
        "  modalities: [tabular]\n"
        "  tasks: [regression]\n",
        encoding="utf-8",
    )
    cands = {c.key: c for c in build_candidates(_cfg(model_catalog_override=[str(override)]))}
    assert "linear" not in cands
    assert cands["my_gbm"].source is CandidateSource.USER_CATALOG
    assert cands["ridge"].source is CandidateSource.BUILTIN_CATALOG


def test_user_override_param_change(tmp_path: Path) -> None:
    override = tmp_path / "m.yaml"
    override.write_text("ridge:\n  default_params: {alpha: 5.0}\n", encoding="utf-8")
    cands = {c.key: c for c in build_candidates(_cfg(model_catalog_override=[str(override)]))}
    assert cands["ridge"].default_params["alpha"] == 5.0


def test_search_space_parsed() -> None:
    lgbm = next(c for c in build_candidates(_cfg()) if c.key == "lightgbm")
    assert "learning_rate" in lgbm.search_space
    assert lgbm.search_space["learning_rate"].type == "loguniform"
    assert lgbm.fidelity == "n_estimators"
    assert lgbm.supports_early_stopping is True
