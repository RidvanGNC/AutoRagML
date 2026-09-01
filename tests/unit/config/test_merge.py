"""config.merge — derin merge + provenance semantiği (ADR 0016)."""

from __future__ import annotations

from autoragml.config.merge import deep_merge, flatten_paths, merge_with_provenance


def test_deep_merge_recurses_dicts() -> None:
    base = {"budget": {"a": 1, "b": 2}, "x": 1}
    overlay = {"budget": {"b": 99, "c": 3}}
    assert deep_merge(base, overlay) == {"budget": {"a": 1, "b": 99, "c": 3}, "x": 1}


def test_deep_merge_replaces_scalars_and_lists() -> None:
    assert deep_merge({"s": [1, 2]}, {"s": [3]}) == {"s": [3]}
    assert deep_merge({"s": 1}, {"s": None}) == {"s": None}


def test_merge_with_provenance_last_layer_wins() -> None:
    layers = [
        ("preset", {"seed": 1, "budget": {"max_trials_per_model": 10}}),
        ("overrides", {"budget": {"max_trials_per_model": 40}}),
    ]
    merged, prov = merge_with_provenance(layers)
    assert merged == {"seed": 1, "budget": {"max_trials_per_model": 40}}
    assert prov["seed"] == "preset"
    assert prov["budget.max_trials_per_model"] == "overrides"


def test_merge_partial_nested_object() -> None:
    layers = [
        ("preset", {"split_policy": {"kind": "rolling_origin", "horizon": 4, "n_folds": 4}}),
        ("overrides", {"split_policy": {"n_folds": 6}}),
    ]
    merged, prov = merge_with_provenance(layers)
    assert merged["split_policy"] == {"kind": "rolling_origin", "horizon": 4, "n_folds": 6}
    assert prov["split_policy.kind"] == "preset"
    assert prov["split_policy.n_folds"] == "overrides"


def test_flatten_paths_lists_are_leaves() -> None:
    paths = flatten_paths({"a": {"b": 1, "c": [1, 2]}, "d": 3})
    assert set(paths) == {"a.b", "a.c", "d"}
