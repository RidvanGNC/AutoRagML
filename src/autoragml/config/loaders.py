"""YAML yükleme yardımcıları (ADR 0016)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autoragml.exceptions import ConfigError

JsonDict = dict[str, Any]


def load_yaml_text(text: str, *, source: str) -> JsonDict:
    """YAML metnini sözlüğe çevir. Boş → {}. Sözlük olmayan kök → hata."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # noqa: TRY003 - bağlam önemli
        msg = f"YAML ayrıştırma hatası ({source}): {exc}"
        raise ConfigError(msg) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"YAML kökü sözlük olmalı ({source}), gelen: {type(data).__name__}"
        raise ConfigError(msg)
    return data


def load_yaml_file(path: str | Path) -> JsonDict:
    """Diskteki bir YAML dosyasını yükle."""
    p = Path(path)
    if not p.is_file():
        msg = f"Config dosyası bulunamadı: {p}"
        raise ConfigError(msg)
    return load_yaml_text(p.read_text(encoding="utf-8"), source=str(p))
