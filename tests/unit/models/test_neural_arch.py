"""models.neural_arch — TabularModelEstimator + bundle round-trip (ADR 0031).

pytorch_tabular yoksa tüm test'ler atlanır.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_HAS_PT = importlib.util.find_spec("pytorch_tabular") is not None
pytestmark = pytest.mark.skipif(not _HAS_PT, reason="pytorch_tabular kurulu değil")
_FILTER = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")


@pytest.fixture
def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(240, 4))
    y = np.sin(x[:, 0]) * 2 + x[:, 1] ** 2 + rng.normal(0, 0.2, 240)
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(4)]), y


@_FILTER
def test_layer_widths_scaling() -> None:
    from autoragml.models.neural_arch import _layer_widths

    assert _layer_widths(3, 256, "const") == [256, 256, 256]
    assert _layer_widths(3, 256, "pyramid") == [256, 128, 64]
    assert len(_layer_widths(4, 512, "funnel")) == 4


@_FILTER
@pytest.mark.parametrize("family", ["mlp", "gandalf", "ft_transformer"])
def test_estimator_fits_and_predicts(family: str, _data: tuple[pd.DataFrame, np.ndarray]) -> None:
    from autoragml.models.neural_arch import TabularModelEstimator

    x, y = _data
    est = TabularModelEstimator(family=family, task_kind="regression", n_layers=2,
                                layer_width=64, max_epochs=6, seed=1)
    est.fit(x, y)
    pred = est.predict(x.iloc[:20])
    assert pred.shape == (20,)
    assert not np.isnan(pred).any()


@_FILTER
def test_build_estimator_routes_neural_arch() -> None:
    from autoragml.contracts.candidate import Candidate
    from autoragml.contracts.enums import Modality, Task
    from autoragml.models import build_estimator
    from autoragml.models.neural_arch import TabularModelEstimator

    cand = Candidate(key="neural_arch_search", name="nas", family="neural",
                     class_path="__neural_arch__", modalities=[Modality.TABULAR],
                     tasks=[Task.REGRESSION], default_params={"max_epochs": 5})
    est = build_estimator(cand, Task.REGRESSION, {"family": "mlp", "n_layers": 1})
    assert isinstance(est, TabularModelEstimator)
    assert est.task_kind == "regression" and est.family == "mlp"


@_FILTER
def test_bundle_round_trip_neural_sidecar(tmp_path, _data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """ADR 0031: save_bundle → champion_neural/ sidecar; load_bundle geri koyar, tahminler eşleşir."""
    from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
    from autoragml.engines.model_pipeline import FittedModelPipeline
    from autoragml.models.neural_arch import TabularModelEstimator
    from autoragml.persistence.bundle import _NEURAL_DIR, load_bundle, save_bundle
    from autoragml.preprocessors import TargetTransform
    from autoragml.preprocessors.pipeline import FittedFeaturePipeline

    x, y = _data
    est = TabularModelEstimator(family="mlp", n_layers=1, layer_width=32, max_epochs=6, seed=2)
    est.fit(x, y)
    ref = est.predict(x.iloc[:30])

    pipe = FittedModelPipeline(
        feature_pipeline=FittedFeaturePipeline(steps=[]),
        estimator=est,
        target_transform=TargetTransform("none").fit(y),
        feature_cols=list(x.columns),
        reserved=set(),
    )
    bundle = ModelBundle(
        metadata=BundleMetadata(
            feature_cols=list(x.columns), feature_set_hash="h", target_col="y",
            model_key="neural_arch_search", params={"family": "mlp", "n_layers": 1},
        ),
        pipeline=pipe,
    )
    dest = tmp_path / "champion.joblib"
    save_bundle(bundle, dest)
    assert (tmp_path / _NEURAL_DIR).is_dir()
    # bellekteki bundle hâlâ çalışır (save estimator'ı geri koydu)
    assert np.allclose(pipe.predict(x.iloc[:30]), ref, atol=1e-4)

    reloaded = load_bundle(dest)
    got = reloaded.pipeline.predict(x.iloc[:30])
    assert np.allclose(got, ref, atol=1e-4)
