"""Katmanlı derin merge + alan-düzeyi provenance (ADR 0016).

Kurallar:
- `dict + dict` → özyinelemeli birleşme
- `scalar | list | None + herhangi` → değiştir (açık `None` de ezer)
- provenance yalnız yaprak (scalar/list/None) düzeyinde tutulur; dict'ler özyinelenir
"""

from __future__ import annotations

from typing import Any

JsonDict = dict[str, Any]


def deep_merge(base: JsonDict, overlay: JsonDict) -> JsonDict:
    """İki sözlüğü özyinelemeli birleştir (overlay kazanır)."""
    out: JsonDict = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            out[key] = deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _merge_layer(
    merged: JsonDict,
    provenance: dict[str, str],
    layer: JsonDict,
    layer_name: str,
    prefix: str,
) -> JsonDict:
    out: JsonDict = dict(merged)
    for key, value in layer.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            base = out.get(key)
            base_dict: JsonDict = base if isinstance(base, dict) else {}
            out[key] = _merge_layer(base_dict, provenance, value, layer_name, f"{path}.")
        else:
            out[key] = value
            provenance[path] = layer_name
    return out


def merge_with_provenance(
    layers: list[tuple[str, JsonDict]],
) -> tuple[JsonDict, dict[str, str]]:
    """Sıralı katmanları birleştir; her yaprak alanın kaynak katmanını kaydet."""
    merged: JsonDict = {}
    provenance: dict[str, str] = {}
    for name, layer in layers:
        merged = _merge_layer(merged, provenance, layer, name, prefix="")
    return merged, provenance


def flatten_paths(data: JsonDict, prefix: str = "") -> list[str]:
    """Bir sözlüğün tüm yaprak yollarını nokta-ayrımlı döndür (list'ler yaprak)."""
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(flatten_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths
