"""fine_tuners.arch_search — nöral mimari arama (ADR 0031).

pytorch_tabular kurulu OLMASA da geçer: koşullu uzay örnekleme, tuner çözümleme, kapı.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.candidate import SearchDim
from autoragml.fine_tuners import resolve_tuner
from autoragml.fine_tuners.arch_search import ArchitectureSearchTuner, _load_space, make_arch_tuner
from autoragml.fine_tuners.space import sample_params
from autoragml.validators.runner import DefaultTuner

_HAS_PT = importlib.util.find_spec("pytorch_tabular") is not None


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides=over or None).config


# --- koşullu arama uzayı (SearchDim.condition) ---

def test_conditional_dim_sampled_only_when_condition_met() -> None:
    space = {
        "family": SearchDim(type="categorical", choices=["gandalf", "mlp"]),
        "gflu_stages": SearchDim(type="int", low=2, high=12, condition={"param": "family", "eq": "gandalf"}),
        "n_layers": SearchDim(type="int", low=1, high=4),
    }
    rng = np.random.default_rng(0)
    gandalf_seen = mlp_seen = False
    for _ in range(40):
        p = sample_params(space, rng)
        assert "n_layers" in p  # koşulsuz her zaman
        if p["family"] == "gandalf":
            assert "gflu_stages" in p
            gandalf_seen = True
        else:
            assert "gflu_stages" not in p  # koşul sağlanmadı → atlandı
            mlp_seen = True
    assert gandalf_seen and mlp_seen


def test_condition_operators() -> None:
    rng = np.random.default_rng(1)
    for op, val, pos, neg in [("eq", 2, 2, 3), ("ne", 2, 3, 2), ("ge", 2, 3, 1)]:
        space = {
            "n": SearchDim(type="categorical", choices=[pos, neg]),
            "dep": SearchDim(type="int", low=0, high=1, condition={"param": "n", op: val}),
        }
        got_pos = got_neg = False
        for _ in range(30):
            p = sample_params(space, rng)
            if p["n"] == pos:
                assert "dep" in p
                got_pos = True
            else:
                got_neg = True
        assert got_pos and got_neg


def test_load_space_yaml() -> None:
    small = _load_space("small")
    assert {"n_layers", "layer_width", "dropout", "learning_rate"} <= set(small)
    full = _load_space("full")
    assert full["gflu_stages"].condition == {"param": "family", "eq": "gandalf"}


# --- tuner çözümleme ---

def test_resolve_tuner_no_search_returns_base() -> None:
    t = resolve_tuner(_cfg(neural_search=False))
    assert not isinstance(t, ArchitectureSearchTuner)


def test_make_arch_tuner_gated() -> None:
    base = DefaultTuner()
    assert make_arch_tuner(_cfg(neural_search=False), base) is base
    t = make_arch_tuner(_cfg(neural_search=True), base)
    assert isinstance(t, ArchitectureSearchTuner)


@pytest.mark.skipif(not _HAS_PT, reason="pytorch_tabular kurulu değil")
def test_resolve_tuner_with_search() -> None:
    t = resolve_tuner(_cfg(neural_search=True, hpo_level="none"))
    assert isinstance(t, ArchitectureSearchTuner)


def test_arch_tuner_delegates_non_arch_candidate() -> None:
    """`neural_arch_search` dışındaki adaylar fallback tuner'a gider (arama yok)."""
    from autoragml.contracts.candidate import Candidate
    from autoragml.contracts.enums import Modality, Task

    sentinel = object()

    class _FakeFallback:
        def tune(self, *_a: object, **_k: object) -> object:
            return sentinel

    lgbm = Candidate(
        key="lightgbm", name="lgbm", family="gbdt", class_path={"regression": "x.R"},
        modalities=[Modality.TABULAR], tasks=[Task.REGRESSION],
    )
    tuner = ArchitectureSearchTuner(fallback=_FakeFallback())  # type: ignore[arg-type]
    assert tuner.tune(lgbm, None, None, None, None, None) is sentinel  # type: ignore[arg-type]
