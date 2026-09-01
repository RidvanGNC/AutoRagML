"""Paket geneli hata tipleri."""

from __future__ import annotations


class AutoRagMLError(Exception):
    """Tüm AutoRagML hatalarının tabanı."""


class ConfigError(AutoRagMLError):
    """Config çözümleme / doğrulama hatası (ADR 0016)."""


class PresetError(ConfigError):
    """Preset bulunamadı, bozuk, veya `extends` döngüsü (ADR 0016)."""
