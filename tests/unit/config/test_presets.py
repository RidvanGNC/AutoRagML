"""config.presets — yerleşik preset yükleme + extends + döngü (ADR 0016)."""

from __future__ import annotations

import pytest

from autoragml.config.presets import list_presets, preset_chain, resolve_preset_layers
from autoragml.exceptions import PresetError


def test_builtin_presets_present() -> None:
    names = list_presets()
    assert {"tabular_fast", "timeseries_rolling", "demandsensing"} <= set(names)


def test_resolve_layers_strips_meta_keys() -> None:
    layers = resolve_preset_layers("tabular_fast")
    assert len(layers) == 1
    name, data = layers[0]
    assert name == "preset:tabular_fast"
    assert not ({"description", "preset", "extends"} & data.keys())


def test_extends_chain_root_to_leaf() -> None:
    layers = resolve_preset_layers("demandsensing")
    assert [n for n, _ in layers] == ["preset:timeseries_rolling", "preset:demandsensing"]
    # kök katman
    assert layers[0][1]["split_policy"]["kind"] == "rolling_origin"
    # yaprak katman kendi ezmesi
    assert layers[1][1]["project_name"] == "demandsensing"
    assert preset_chain("demandsensing") == ["timeseries_rolling", "demandsensing"]


def test_unknown_preset_raises() -> None:
    with pytest.raises(PresetError, match="bulunamadı"):
        resolve_preset_layers("yok_boyle_bir_sey")
