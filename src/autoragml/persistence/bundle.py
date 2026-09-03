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
_NEURAL_DIR = "champion_neural"        # ADR 0031: pytorch_tabular model dizini (joblib yanı sıra)
_NEURAL_TS_DIR = "champion_neural_ts"  # ADR 0032: neuralforecast model dizini
_FOUNDATION_DIR = "champion_foundation"        # ADR 0033: TabPFN context sidecar
_FOUNDATION_TS_DIR = "champion_foundation_ts"  # ADR 0033: Chronos context sidecar

_SIDECAR_ESTIMATORS = {"TabularModelEstimator", "FoundationTabEstimator"}
_SIDECAR_PIPELINES = {"FittedNeuralForecaster", "FittedChronosForecaster"}


def _sidecar_estimator(pipeline: Any) -> Any | None:
    """Pipeline'ın (tek-model) torch-tabanlı estimator'ını bul — joblib-picklable değil."""
    est = getattr(pipeline, "_estimator", None)
    return est if type(est).__name__ in _SIDECAR_ESTIMATORS else None


def _sidecar_pipeline(pipeline: Any) -> Any | None:
    """Pipeline'ın kendisi torch-tabanlı forecaster ise onu döndür (joblib-picklable değil)."""
    return pipeline if type(pipeline).__name__ in _SIDECAR_PIPELINES else None


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

    # ADR 0031/0032/0033: torch modelleri joblib-picklable değil → sidecar dizin, pickle'da None,
    # yüklemede geri koy. (estimator sidecar: FittedModelPipeline._estimator; pipeline sidecar:
    # pipeline'ın kendisi.)
    side_est = _sidecar_estimator(bundle.pipeline)
    side_pipe = _sidecar_pipeline(bundle.pipeline)
    kind = ""
    saved_pipeline: Any = bundle.pipeline
    if side_est is not None:
        est_name = type(side_est).__name__
        if est_name == "FoundationTabEstimator":  # ADR 0033
            side_est.save(dest.parent / _FOUNDATION_DIR)
            kind = "foundation_tab"
        else:  # TabularModelEstimator — ADR 0031
            side_est.save(dest.parent / _NEURAL_DIR)
            kind = "arch"
        bundle.pipeline._estimator = None  # noqa: SLF001
    elif side_pipe is not None:
        if type(side_pipe).__name__ == "FittedChronosForecaster":  # ADR 0033
            side_pipe.save(str(dest.parent / _FOUNDATION_TS_DIR))
            kind = "foundation_ts"
        else:  # FittedNeuralForecaster — ADR 0032
            side_pipe.save(str(dest.parent / _NEURAL_TS_DIR))
            kind = "ts"
        saved_pipeline = None  # tüm pipeline sidecar'dan yüklenir

    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "autoragml_version": __version__,
        "saved_env": _env_snapshot(),
        "metadata": bundle.metadata,
        "metrics_oof": dict(bundle.metrics_oof),
        "metrics_holdout": dict(bundle.metrics_holdout),
        "pipeline": saved_pipeline,
        "neural_sidecar": kind or False,
    }
    try:
        joblib.dump(payload, dest, compress=compress)
    except Exception as exc:  # noqa: BLE001
        msg = f"bundle serialize edilemedi ({dest}): {exc}"
        raise PersistenceError(msg) from exc
    finally:
        if kind in {"arch", "foundation_tab"}:
            bundle.pipeline._estimator = side_est  # noqa: SLF001 - bellekteki bundle sağlam kalsın
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
    sidecar = payload.get("neural_sidecar")
    if sidecar == "arch" and pipeline is not None:  # ADR 0031
        from autoragml.models.neural_arch import TabularModelEstimator

        neural_dir = src.parent / _NEURAL_DIR
        if not neural_dir.is_dir():
            raise PersistenceError(f"nöral sidecar dizini yok: {neural_dir}")
        pipeline._estimator = TabularModelEstimator().load(neural_dir)  # noqa: SLF001
    elif sidecar == "foundation_tab" and pipeline is not None:  # ADR 0033
        from autoragml.models.foundation_tab import FoundationTabEstimator

        fnd_dir = src.parent / _FOUNDATION_DIR
        if not fnd_dir.is_dir():
            raise PersistenceError(f"foundation sidecar dizini yok: {fnd_dir}")
        pipeline._estimator = FoundationTabEstimator().load(fnd_dir)  # noqa: SLF001
    elif sidecar == "foundation_ts":  # ADR 0033
        from autoragml.engines.timeseries.foundation_ts import FittedChronosForecaster

        fts_dir = src.parent / _FOUNDATION_TS_DIR
        if not fts_dir.is_dir():
            raise PersistenceError(f"foundation-TS sidecar dizini yok: {fts_dir}")
        pipeline = FittedChronosForecaster.__new__(FittedChronosForecaster).load(str(fts_dir))
    elif sidecar == "ts" or (sidecar and pipeline is None):  # ADR 0032
        from autoragml.engines.timeseries.neural_ts import FittedNeuralForecaster

        ts_dir = src.parent / _NEURAL_TS_DIR
        if not ts_dir.is_dir():
            raise PersistenceError(f"nöral-TS sidecar dizini yok: {ts_dir}")
        pipeline = FittedNeuralForecaster.__new__(FittedNeuralForecaster).load(str(ts_dir))

    return ModelBundle(
        metadata=md,
        metrics_oof=payload.get("metrics_oof", {}),
        metrics_holdout=payload.get("metrics_holdout", {}),
        artifact_path=str(src),
        pipeline=pipeline,
    )
