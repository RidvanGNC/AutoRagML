"""ModelBundle ↔ disk — joblib tek dosya (ADR 0018).

**GÜVENLİK:** `joblib.load` pickle çözer → keyfi kod çalıştırır. Yalnız güvendiğiniz
bundle'ları yükleyin. sklearn pickle'ları sürümler arası garanti değildir — yüklemede
sürüm sapması WARNING'lenir.
"""

from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib

from autoragml import __version__
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.exceptions import PersistenceError
from autoragml.logging import get_logger

logger = get_logger(__name__)

_FORMAT_VERSION = 1
_ENV_PACKAGES = ("scikit-learn", "lightgbm", "numpy", "scipy", "pandas")
_VERSION_SENSITIVE = ("scikit-learn", "lightgbm")
_load_security_warned = False
_NEURAL_DIR = "champion_neural"  # ADR 0031: pytorch_tabular model dizini (joblib yanı sıra)


def _neural_estimator(pipeline: Any) -> Any | None:
    """Pipeline'ın (tek-model) `TabularModelEstimator`'ını bul — yoksa None (ADR 0031)."""
    est = getattr(pipeline, "_estimator", None)
    return est if type(est).__name__ == "TabularModelEstimator" else None


def _env_snapshot() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in _ENV_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:  # pragma: no cover
            continue
    return out


def _minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def _warn_env_drift(saved: dict[str, str]) -> None:
    current = _env_snapshot()
    for pkg in _VERSION_SENSITIVE:
        old, new = saved.get(pkg), current.get(pkg)
        if old and new and _minor(old) != _minor(new):
            logger.warning(
                "[persistence] bundle `%s` %s ile kaydedildi, ortamda %s var — "
                "pickle uyumsuzluğu olabilir",
                pkg,
                old,
                new,
            )


def save_bundle(bundle: ModelBundle, path: str | Path, *, compress: int = 3) -> Path:
    """`ModelBundle`'ı joblib tek dosyaya yaz (canlı `pipeline` dahil)."""
    if bundle.pipeline is None:
        msg = "save_bundle: bundle.pipeline None — kaydedilecek fitted nesne yok"
        raise PersistenceError(msg)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # ADR 0031: nöral mimari arama şampiyonu — TabularModel joblib-picklable değil.
    # Estimator'ı sidecar dizine yaz, pickle'da None bırak, yüklemede geri koy.
    neural_est = _neural_estimator(bundle.pipeline)
    neural_saved = False
    if neural_est is not None:
        neural_est.save(dest.parent / _NEURAL_DIR)
        bundle.pipeline._estimator = None  # noqa: SLF001
        neural_saved = True

    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "autoragml_version": __version__,
        "saved_env": _env_snapshot(),
        "metadata": bundle.metadata,
        "metrics_oof": dict(bundle.metrics_oof),
        "metrics_holdout": dict(bundle.metrics_holdout),
        "pipeline": bundle.pipeline,
        "neural_sidecar": neural_saved,
    }
    try:
        joblib.dump(payload, dest, compress=compress)
    except Exception as exc:  # noqa: BLE001
        msg = f"bundle serialize edilemedi ({dest}): {exc}"
        raise PersistenceError(msg) from exc
    finally:
        if neural_saved:
            bundle.pipeline._estimator = neural_est  # noqa: SLF001 - bellekteki bundle sağlam kalsın
    return dest


def load_bundle(path: str | Path) -> ModelBundle:
    """Diskteki joblib bundle'ını `ModelBundle` olarak yükle (sürüm kontrollü)."""
    global _load_security_warned
    src = Path(path)
    if not src.is_file():
        msg = f"bundle bulunamadı: {src}"
        raise PersistenceError(msg)
    if not _load_security_warned:
        _load_security_warned = True
        logger.warning(
            "[persistence] joblib.load keyfi kod çalıştırabilir — yalnız güvenilen bundle'ları yükleyin"
        )
    try:
        payload = joblib.load(src)
    except Exception as exc:  # noqa: BLE001
        msg = f"bundle yüklenemedi ({src}): {exc}"
        raise PersistenceError(msg) from exc

    if not isinstance(payload, dict) or payload.get("format_version") != _FORMAT_VERSION:
        got = payload.get("format_version") if isinstance(payload, dict) else "?"
        msg = f"desteklenmeyen bundle format_version={got} (beklenen {_FORMAT_VERSION})"
        raise PersistenceError(msg)

    _warn_env_drift(payload.get("saved_env", {}))

    md = payload["metadata"]
    if not isinstance(md, BundleMetadata):
        md = BundleMetadata(**md)

    pipeline = payload.get("pipeline")
    if payload.get("neural_sidecar") and pipeline is not None:  # ADR 0031
        from autoragml.models.neural_arch import TabularModelEstimator

        neural_dir = src.parent / _NEURAL_DIR
        if not neural_dir.is_dir():
            msg = f"nöral sidecar dizini yok: {neural_dir}"
            raise PersistenceError(msg)
        pipeline._estimator = TabularModelEstimator().load(neural_dir)  # noqa: SLF001

    return ModelBundle(
        metadata=md,
        metrics_oof=payload.get("metrics_oof", {}),
        metrics_holdout=payload.get("metrics_holdout", {}),
        artifact_path=str(src),
        pipeline=pipeline,
    )
