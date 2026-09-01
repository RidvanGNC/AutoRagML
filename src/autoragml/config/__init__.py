"""config — `RunConfig` katmanlı çözümleme (ADR 0016).

defaults ← preset (`extends` zinciri, her biri ayrı katman) ← kullanıcı dosyası ←
runtime override. `resolve_run_config` doğrulanmış `RunConfig` + alan-düzeyi provenance
döndürür.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autoragml.config.loaders import load_yaml_file
from autoragml.config.merge import flatten_paths, merge_with_provenance
from autoragml.config.presets import list_presets, preset_chain, resolve_preset_layers
from autoragml.config.settings import Settings, parse_dotenv
from autoragml.contracts.config_resolution import ConfigResolution
from autoragml.contracts.run_config import RunConfig
from autoragml.exceptions import ConfigError

JsonDict = dict[str, Any]

__all__ = [
    "ConfigError",
    "Settings",
    "list_presets",
    "parse_dotenv",
    "preset_chain",
    "resolve_run_config",
]


def resolve_run_config(
    *,
    target: str | None = None,
    preset: str | None = None,
    config_file: str | Path | None = None,
    overrides: JsonDict | None = None,
) -> ConfigResolution:
    """Katmanları birleştirip doğrulanmış `RunConfig` üret.

    `target` kwarg'ı `overrides["target"]` için kısayoldur; ikisi çelişirse hata.
    """
    override_layer: JsonDict = dict(overrides or {})
    if target is not None:
        existing = override_layer.get("target")
        if existing is not None and existing != target:
            msg = f"Çelişkili target: kwarg={target!r} vs overrides={existing!r}"
            raise ConfigError(msg)
        override_layer["target"] = target

    layers: list[tuple[str, JsonDict]] = []
    if preset is not None:
        layers.extend(resolve_preset_layers(preset))
    if config_file is not None:
        layers.append((f"file:{config_file}", load_yaml_file(config_file)))
    if override_layer:
        layers.append(("overrides", override_layer))

    merged, provenance = merge_with_provenance(layers)

    if "target" not in merged:
        msg = "target zorunlu — kwarg, config_file veya overrides ile verin (ADR 0008/3)"
        raise ConfigError(msg)

    try:
        config = RunConfig.model_validate(merged)
    except ValidationError as exc:
        msg = f"RunConfig doğrulaması başarısız:\n{exc}"
        raise ConfigError(msg) from exc

    for path in flatten_paths(config.model_dump(mode="json")):
        provenance.setdefault(path, "default")

    return ConfigResolution(
        config=config,
        provenance=provenance,
        layers=["defaults", *(name for name, _ in layers)],
    )
