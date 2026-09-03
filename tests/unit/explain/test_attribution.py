"""explain.attribution — öznitelik atıfı (ADR 0037)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.dynamics import build_plan
from autoragml.engines.champion import refit_champion
from autoragml.explain import explain_champion
from autoragml.io import load_dataset, materialize_frame
from autoragml.models import resolve_candidates
from autoragml.scoring import score_reports
from autoragml.validators import run_validation_suite

_HAS_SHAP = importlib.util.find_spec("shap") is not None


def _reg_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 4))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.1 * x[:, 2] + rng.normal(0, 0.3, n)  # x3 önemsiz
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(4)}})


def _champion(df: pd.DataFrame, **over: object):
    cfg = resolve_run_config(target="y", overrides={"hpo_level": "none", **over}).config
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    frame = materialize_frame(ds)
    plan = build_plan(profile, task, cfg)
    cands = resolve_candidates(cfg, task)
    reports = run_validation_suite(cands, frame, plan, profile, task, cfg)
    selection = score_reports(reports, cands, cfg, task, profile)
    bundle = refit_champion(selection, cands, reports, frame, plan, profile, task, cfg)
    return bundle, task


def test_permutation_importance_ranks_signal_features() -> None:
    bundle, task = _champion(_reg_df())
    expl = explain_champion(bundle, _reg_df(150), task, method="permutation")
    assert expl.method == "permutation"
    assert set(expl.feature_names) == {"f0", "f1", "f2", "f3"}
    top2 = {fs.feature for fs in expl.global_importance[:2]}
    assert top2 == {"f0", "f1"}  # en güçlü iki sinyal
    assert expl.global_importance[-1].feature == "f3"  # gürültü en altta
    assert all(fs.importance >= 0 for fs in expl.global_importance)


def test_explain_requires_data_for_feature_model() -> None:
    bundle, task = _champion(_reg_df())
    with pytest.raises(ValueError, match="temsili"):
        explain_champion(bundle, None, task)


def test_explain_unavailable_for_classical_forecaster() -> None:
    from autoragml.contracts.enums import Modality, Task
    from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
    from autoragml.contracts.task_spec import TaskSpec
    from autoragml.engines.timeseries.classical import FittedClassicalForecaster

    fake = FittedClassicalForecaster.__new__(FittedClassicalForecaster)
    bundle = ModelBundle(
        metadata=BundleMetadata(
            feature_cols=[], feature_set_hash="h", target_col="y", model_key="auto_ets",
            params={"family": "statistical"},
        ),
        pipeline=fake,
    )
    task = TaskSpec(task=Task.FORECASTING, modality=Modality.TIMESERIES, targets=["y"], time_col="ds")
    expl = explain_champion(bundle, None, task)
    assert expl.method == "unavailable"
    assert any("öznitelik-tabanlı değil" in n for n in expl.notes)


def test_runresult_explain_merges_structural_and_attribution(tmp_path) -> None:
    from autoragml.interfaces import Orchestrator

    resolution = resolve_run_config(
        target="y", overrides={"hpo_level": "none", "output_dir": str(tmp_path), "project_name": "x"}
    )
    result = Orchestrator().run(_reg_df(), resolution.config, resolution=resolution)
    # data yok → yapısal özet + attribution "skipped"
    out0 = result.explain()
    assert "champion" in out0 and out0["attribution"]["method"] == "skipped"
    # data ile → attribution dolu
    out1 = result.explain(_reg_df(120))
    assert out1["attribution"]["method"] in {"permutation", "shap_tree", "shap_linear", "shap"}
    assert out1["attribution"]["global_importance"]


@pytest.mark.skipif(not _HAS_SHAP, reason="shap kurulu değil")
def test_shap_path_when_available() -> None:
    bundle, task = _champion(_reg_df(), model_catalog_override=[])
    expl = explain_champion(bundle, _reg_df(120), task, method="auto")
    # tek model + ağaç/linear ise shap_*; ensemble/bag ise permutation
    assert expl.method in {"shap_tree", "shap_linear", "shap", "permutation"}
    assert expl.global_importance
