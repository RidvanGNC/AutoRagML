"""Yerleşik preset çözümleme + `extends` zinciri (ADR 0016).

Presetler pakete gömülü: `autoragml/config/presets/*.yaml`. `importlib.resources` ile
okunur (wheel'de de çalışır). Her preset kendi merge katmanı olarak döner —
`extends` üzerinden gelen alanlar provenance'ta doğru preset'e atfedilir.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from autoragml.config.loaders import load_yaml_text
from autoragml.exceptions import PresetError

JsonDict = dict[str, Any]

_META_KEYS = frozenset({"preset", "extends", "description"})
_PRESET_PKG = "autoragml.config._presets"


def list_presets() -> list[str]:
    """Kullanılabilir yerleşik preset adları."""
    names: list[str] = []
    for entry in resources.files(_PRESET_PKG).iterdir():
        name = entry.name
        if name.endswith(".yaml") and entry.is_file():
            names.append(name.removesuffix(".yaml"))
    return sorted(names)


def _read_preset(name: str) -> JsonDict:
    resource = resources.files(_PRESET_PKG) / f"{name}.yaml"
    if not resource.is_file():
        available = ", ".join(list_presets()) or "(yok)"
        msg = f"Preset bulunamadı: {name!r}. Mevcut: {available}"
        raise PresetError(msg)
    return load_yaml_text(resource.read_text(encoding="utf-8"), source=f"preset:{name}")


def _strip_meta(data: JsonDict) -> JsonDict:
    return {k: v for k, v in data.items() if k not in _META_KEYS}


def resolve_preset_layers(name: str) -> list[tuple[str, JsonDict]]:
    """`extends` zincirini kökten yaprağa merge katmanları olarak çöz.

    Döner: `[("preset:kök", {...}), ..., ("preset:<name>", {...})]`.
    Meta anahtarlar soyulur. `extends` döngüsü → `PresetError`.
    """
    seen: set[str] = set()
    stack: list[str] = []
    current: str | None = name

    while current is not None:
        if current in seen:
            cycle = " -> ".join([*stack, current])
            msg = f"Preset `extends` döngüsü: {cycle}"
            raise PresetError(msg)
        seen.add(current)
        stack.append(current)
        raw = _read_preset(current)
        parent = raw.get("extends")
        if parent is not None and not isinstance(parent, str):
            msg = f"Preset {current!r}: `extends` bir string olmalı"
            raise PresetError(msg)
        current = parent

    return [(f"preset:{n}", _strip_meta(_read_preset(n))) for n in reversed(stack)]


def preset_chain(name: str) -> list[str]:
    """Sadece zincir adları (kökten yaprağa), tanı/log için."""
    return [layer_name.removeprefix("preset:") for layer_name, _ in resolve_preset_layers(name)]
