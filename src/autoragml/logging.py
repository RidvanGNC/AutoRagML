"""Paket geneli logging yardımcısı.

Kütüphane kök logger'ı yapılandırmaz — yalnız isimli logger döndürür.
Uygulama/CLI kendi handler'ını kurar.
"""

from __future__ import annotations

import logging

_ROOT = "autoragml"


def get_logger(name: str) -> logging.Logger:
    """`autoragml.<name>` altında isimli logger döndür."""
    suffix = name.removeprefix("autoragml.").removeprefix("autoragml")
    full = _ROOT if not suffix else f"{_ROOT}.{suffix.lstrip('.')}"
    return logging.getLogger(full)
