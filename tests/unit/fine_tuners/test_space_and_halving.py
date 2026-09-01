"""fine_tuners.space + halving (ADR 0013)."""

from __future__ import annotations

import numpy as np
import pytest

from autoragml.contracts.adaptive_plan import CandidateOpGroup
from autoragml.contracts.candidate import SearchDim
from autoragml.fine_tuners.halving import build_schedule
from autoragml.fine_tuners.space import SpaceError, sample_candidate_choices, sample_params, sample_value


def test_sample_value_types() -> None:
    rng = np.random.default_rng(0)
    assert isinstance(sample_value(SearchDim(type="int", low=1, high=5), rng), int)
    f = sample_value(SearchDim(type="float", low=0.0, high=1.0), rng)
    assert 0.0 <= f <= 1.0
    lg = sample_value(SearchDim(type="loguniform", low=1e-3, high=1e0), rng)
    assert 1e-3 <= lg <= 1e0
    c = sample_value(SearchDim(type="categorical", choices=["a", "b", "c"]), rng)
    assert c in {"a", "b", "c"}


def test_loguniform_is_log_distributed() -> None:
    rng = np.random.default_rng(1)
    dim = SearchDim(type="loguniform", low=1e-4, high=1e2)
    samples = np.array([sample_value(dim, rng) for _ in range(2000)])
    # yarıdan fazlası geometrik ortanın (1e-1) altında olmalı
    assert np.mean(samples < 1e-1) > 0.4


def test_sample_value_errors() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(SpaceError):
        sample_value(SearchDim(type="bogus"), rng)
    with pytest.raises(SpaceError):
        sample_value(SearchDim(type="loguniform", low=-1.0, high=1.0), rng)


def test_sample_params_and_choices() -> None:
    rng = np.random.default_rng(2)
    space = {"lr": SearchDim(type="loguniform", low=0.01, high=0.3), "depth": SearchDim(type="int", low=3, high=10)}
    params = sample_params(space, rng)
    assert set(params) == {"lr", "depth"}
    groups = [CandidateOpGroup(group_name="g1", columns=["x"], choices=["none", "log1p"], default="none")]
    ch = sample_candidate_choices(groups, rng)
    assert ch["g1"] in {"none", "log1p"}


def test_build_schedule_shrinks_by_eta() -> None:
    sched = build_schedule(15, 10, 1000, eta=3.0)
    assert sched[0].keep == 5
    assert sched[-1].fidelity == 1000
    assert [r.fidelity for r in sched] == sorted(r.fidelity for r in sched)


def test_build_schedule_degenerate() -> None:
    assert build_schedule(1, 10, 100) == build_schedule(1, 10, 100)
    single = build_schedule(1, 10, 100)
    assert len(single) == 1 and single[0].fidelity == 100
    assert len(build_schedule(10, 500, 500)) == 1  # min == max
