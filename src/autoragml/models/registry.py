"""Model kataloğu → `Candidate` çözümleme (ADR 0012).

Yerleşik katalog `models/catalog/*.yaml` (pakete gömülü). Kullanıcı override YAML'ları
(`RunConfig.model_catalog_override`) key bazında deep-merge edilir. Eksik `requires` veya
importable olmayan `class_path` → entry atlanır (tek seferlik WARNING).
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from autoragml.config.loaders import load_yaml_file, load_yaml_text
from autoragml.config.merge import deep_merge
from autoragml.contracts.candidate import Candidate, SearchDim
from autoragml.contracts.enums import CandidateSource, Modality, PredictKind, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.exceptions import AutoRagMLError
from autoragml.logging import get_logger

logger = get_logger(__name__)

_CATALOG_PKG = "autoragml.models.catalog"
_ENTRY_POINT_GROUP = "autoragml.models"
_META_KEYS = frozenset({"description"})
_warned: set[str] = set()

JsonDict = dict[str, Any]


class ModelCatalogError(AutoRagMLError):
    """Katalog yükleme / entry doğrulama hatası."""


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("[models] %s", message)


def _builtin_catalog() -> JsonDict:
    merged: JsonDict = {}
    for entry in sorted(resources.files(_CATALOG_PKG).iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".yaml"):
            continue
        data = load_yaml_text(entry.read_text(encoding="utf-8"), source=f"catalog:{entry.name}")
        merged = deep_merge(merged, {k: v for k, v in data.items() if k not in _META_KEYS})
    return merged


def load_catalog(override_paths: list[Path] | None = None) -> JsonDict:
    """Yerleşik katalog + kullanıcı override'ları (key bazında deep-merge)."""
    catalog = _builtin_catalog()
    for path in override_paths or []:
        user = load_yaml_file(path)
        catalog = deep_merge(catalog, {k: v for k, v in user.items() if k not in _META_KEYS})
    return catalog


def _class_exists(class_path: str) -> bool:
    if class_path.startswith("__") and class_path.endswith("__"):
        return True  # sentinel (ör. __neural_arch__) — build_estimator özel işler
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        return False
    if importlib.util.find_spec(module_path.split(".")[0]) is None:
        return False
    try:
        module = importlib.import_module(module_path)
    except Exception:  # noqa: BLE001
        return False
    return hasattr(module, class_name)


def _requires_available(requires: list[str]) -> list[str]:
    return [r for r in requires if importlib.util.find_spec(r) is None]


def _import_hint(paths: list[str]) -> str:
    """Paketi kurulu ama import'u patlıyorsa (ör. macOS'ta libomp'suz lightgbm) gerçek hatayı yüzeye çıkar."""
    for path in paths:
        top = path.split(".")[0]
        if importlib.util.find_spec(top) is None:
            continue
        try:
            importlib.import_module(path.rpartition(".")[0] or top)
        except Exception as exc:  # noqa: BLE001
            return f" — `{top}` kurulu ama import edilemedi: {exc}"
    return ""


def _build_candidate(key: str, entry: JsonDict, *, source: CandidateSource) -> Candidate | None:
    if entry.get("enabled", True) is False:
        return None

    requires = [str(r) for r in entry.get("requires", [])]
    missing = _requires_available(requires)
    if missing:
        _warn_once(key, f"model `{key}` atlandı — eksik paket: {', '.join(missing)}")
        return None

    raw_cp = entry.get("class_path")
    if isinstance(raw_cp, str):
        class_path: str | dict[str, str] = raw_cp
        paths = [raw_cp]
    elif isinstance(raw_cp, dict):
        class_path = {str(k): str(v) for k, v in raw_cp.items()}
        paths = list(class_path.values())
    else:
        _warn_once(key, f"model `{key}` atlandı — `class_path` yok/geçersiz")
        return None

    if not any(_class_exists(p) for p in paths):
        _warn_once(key, f"model `{key}` atlandı — class_path importable değil: {paths}{_import_hint(paths)}")
        return None

    try:
        search_space = {
            str(name): SearchDim(**spec) for name, spec in (entry.get("search_space") or {}).items()
        }
        return Candidate(
            key=key,
            name=str(entry.get("name", key)),
            family=str(entry.get("family", "ml")),
            class_path=class_path,
            modalities=[Modality(m) for m in entry.get("modalities", ["tabular"])],
            tasks=[Task(t) for t in entry.get("tasks", ["regression"])],
            predict_kind=[PredictKind(p) for p in entry.get("predict_kind", ["point"])],
            default_params=dict(entry.get("default_params") or {}),
            search_space=search_space,
            fidelity=entry.get("fidelity"),
            supports_early_stopping=bool(entry.get("supports_early_stopping", False)),
            early_stopping_rounds=entry.get("early_stopping_rounds"),
            requires=requires,
            wrap=bool(entry.get("wrap", False)),
            scale=bool(entry.get("scale", False)),
            source=source,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"model `{key}` entry'si geçersiz: {exc}"
        raise ModelCatalogError(msg) from exc


def _entry_point_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    try:
        eps = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001
        return out
    for ep in eps:
        try:
            entry = ep.load()
        except Exception as exc:  # noqa: BLE001
            _warn_once(f"ep:{ep.name}", f"model entry-point `{ep.name}` yüklenemedi: {exc}")
            continue
        if callable(entry):
            entry = entry()
        cand = _build_candidate(ep.name, dict(entry), source=CandidateSource.ENTRY_POINT)
        if cand is not None:
            out.append(cand)
    return out


def build_candidates(config: RunConfig) -> list[Candidate]:
    """Config'e göre tüm kullanılabilir `Candidate`'ları çöz (task filtresi YOK)."""
    builtin_keys = set(_builtin_catalog())
    catalog = load_catalog(config.model_catalog_override)
    out: list[Candidate] = []
    for key, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        source = (
            CandidateSource.BUILTIN_CATALOG if key in builtin_keys else CandidateSource.USER_CATALOG
        )
        cand = _build_candidate(key, entry, source=source)
        if cand is not None:
            out.append(cand)
    out.extend(_entry_point_candidates())
    return out


def resolve_candidates(config: RunConfig, task: TaskSpec) -> list[Candidate]:
    """Modalite + görev uyumlu adayları döndür."""
    resolved = [
        c
        for c in build_candidates(config)
        if task.modality in c.modalities and task.task in c.tasks
    ]
    if not resolved:
        msg = (
            f"{task.modality}/{task.task} için uygun model adayı yok. "
            "Katalog override veya eksik `requires` kontrol edin."
        )
        raise ModelCatalogError(msg)
    logger.info("[models] %d aday çözüldü: %s", len(resolved), [c.key for c in resolved])
    return resolved
