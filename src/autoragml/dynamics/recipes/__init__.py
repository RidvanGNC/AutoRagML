"""Custom transform registry (ADR 0015).

Kaynaklar (katmanlı):
1. `@register_recipe("ad")` ile bu pakete / import edilmiş modüllere kayıtlı sınıflar
2. `RunConfig.dynamics` `recipe_paths` — proje-yerel dizinden yüklenen `.py` dosyaları
3. entry-points grubu `autoragml.recipes`

Tek registry; isim çakışması → `RecipeError` (sessizce ezme yok).
Bir recipe `autoragml.transform.Transform` protokolüne uyar.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path

from autoragml.exceptions import AutoRagMLError
from autoragml.logging import get_logger
from autoragml.transform import Transform

logger = get_logger(__name__)

_REGISTRY: dict[str, type[Transform]] = {}
_ENTRY_POINT_GROUP = "autoragml.recipes"
_paths_loaded: set[str] = set()
_entry_points_loaded = False


class RecipeError(AutoRagMLError):
    """Recipe bulunamadı, çakıştı, veya protokole uymuyor."""


def register_recipe(name: str) -> Callable[[type[Transform]], type[Transform]]:
    """`@register_recipe("ad")` — bir Transform sınıfını registry'ye ekle."""

    def _decorator(cls: type[Transform]) -> type[Transform]:
        _register(name, cls, source="decorator")
        return cls

    return _decorator


def _register(name: str, cls: type, *, source: str) -> None:
    if not (hasattr(cls, "fit") and callable(cls.fit)):
        msg = f"Recipe {name!r} ({source}): `fit(frame, ctx)` metodu yok — Transform protokolüne uymuyor"
        raise RecipeError(msg)
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        msg = (
            f"Recipe adı çakışması: {name!r} zaten {existing.__module__}.{existing.__qualname__} "
            f"için kayıtlı ({source} ile {cls.__module__}.{cls.__qualname__} eklenemez)"
        )
        raise RecipeError(msg)
    _REGISTRY[name] = cls
    logger.debug("Recipe kaydedildi: %s (%s)", name, source)


def load_recipe_paths(paths: Iterable[str | Path]) -> None:
    """Verilen dizin/dosyalardan `.py` recipe modüllerini içe aktar."""
    for raw in paths:
        p = Path(raw)
        key = str(p.resolve())
        if key in _paths_loaded:
            continue
        files = sorted(p.glob("*.py")) if p.is_dir() else [p]
        for file in files:
            if not file.is_file() or file.name.startswith("_"):
                continue
            mod_name = f"autoragml_recipe_{file.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, file)
            if spec is None or spec.loader is None:  # pragma: no cover
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        _paths_loaded.add(key)


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    try:
        eps = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - eski importlib.metadata davranışı
        return
    for ep in eps:
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recipe entry-point %r yüklenemedi: %s", ep.name, exc)
            continue
        _register(ep.name, cls, source="entry_point")


def get_recipe(name: str) -> type[Transform]:
    """İsimle recipe sınıfı çöz. Bulunamazsa `RecipeError`."""
    _load_entry_points()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(yok)"
        msg = f"Recipe bulunamadı: {name!r}. Kayıtlı: {available}"
        raise RecipeError(msg)
    return cls


def list_recipes() -> list[str]:
    """Kayıtlı tüm recipe adları."""
    _load_entry_points()
    return sorted(_REGISTRY)


def validate_recipes(names: Iterable[str]) -> None:
    """Verilen adların tümü çözülebiliyor mu — plan zamanında fail-fast."""
    for name in names:
        get_recipe(name)
