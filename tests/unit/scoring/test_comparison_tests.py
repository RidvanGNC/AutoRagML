"""scoring.comparison_tests — MCB + Diebold-Mariano (ADR 0014)."""

from __future__ import annotations

from autoragml.scoring.comparison_tests import build_comparison_tests, diebold_mariano, mcb_ranks
from tests.unit.scoring._fixtures import make_report


def test_mcb_ranks_best_model_has_lowest_rank() -> None:
    reports = [
        make_report("good", smape=8.0, se=0.5),
        make_report("mid", smape=15.0, se=0.5),
        make_report("bad", smape=30.0, se=0.5),
    ]
    ranks = mcb_ranks(reports, "smape")
    assert ranks["good"] < ranks["mid"] < ranks["bad"]
    assert 1.0 <= ranks["good"] <= 3.0


def test_diebold_mariano_pvalue_range() -> None:
    champ = make_report("champ", smape=8.0, se=0.3)
    other = make_report("other", smape=25.0, se=0.3)
    p = diebold_mariano(champ, other, "smape")
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_dm_none_when_too_few_folds() -> None:
    a = make_report("a", smape=10.0, n_folds=2)
    b = make_report("b", smape=20.0, n_folds=2)
    assert diebold_mariano(a, b, "smape") is None


def test_build_comparison_tests_none_below_min_folds() -> None:
    reports = [make_report("a", smape=10.0, n_folds=2), make_report("b", smape=12.0, n_folds=2)]
    assert build_comparison_tests(reports, "a", "smape") is None


def test_build_comparison_tests_populated() -> None:
    reports = [make_report("a", smape=8.0), make_report("b", smape=20.0), make_report("c", smape=25.0)]
    ct = build_comparison_tests(reports, "a", "smape")
    assert ct is not None
    assert set(ct.mcb_ranks) == {"a", "b", "c"}
    assert "a" not in ct.dm_pvalues  # şampiyon kendine karşı test edilmez
