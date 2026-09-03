"""Torch ortam yapılandırması — nöral modeller için determinizm + cihaz (ADR 0030).

Çekirdek torch'suz (ADR 0003). Bu modül yalnız `[neural]` extra kuruluyken iş görür;
her fonksiyon torch yoksa **güvenli/boş** döner. `configure_torch` idempotenttir.
"""

from __future__ import annotations

import contextlib
import os
import random
import tempfile
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

from autoragml.logging import get_logger

logger = get_logger(__name__)

DeterminismMode = Literal["strict", "best_effort", "off"]


@contextlib.contextmanager
def quiet_cwd() -> Iterator[None]:
    """pytorch_tabular / neuralforecast CWD'ye `lightning_logs/` `.pt_tmp/` yazar — geçici dizine
    yönlendir (kullanıcı proje dizini kirlenmesin). ADR 0030/0031/0032."""
    prev = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="autoragml_nn_") as d:
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(prev)

_configured = False


@lru_cache(maxsize=1)
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def has_cuda() -> bool:
    """torch kurulu + CUDA cihazı görünür mü."""
    if not torch_available():
        return False
    import torch

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - sürücü/driver sorunları fatal değil
        return False


def cuda_device_name() -> str | None:
    if not has_cuda():
        return None
    import torch

    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return None


def torch_versions() -> dict[str, str | None]:
    """RunManifest.env için: torch + cuda derleme sürümleri (yoksa None)."""
    if not torch_available():
        return {"torch": None, "cuda": None}
    import torch

    return {"torch": str(torch.__version__), "cuda": torch.version.cuda}


def resolve_device(device: str) -> str:
    """`auto`/`cuda`/`cpu` → gerçekte kullanılacak cihaz."""
    if device in {"auto", "cuda"} and has_cuda():
        return "cuda"
    return "cpu"


def configure_torch(seed: int, mode: DeterminismMode = "best_effort", device: str = "auto") -> str:
    """Seed + determinizm + cudnn ayarları. Döner: kullanılacak cihaz (`cuda`/`cpu`).

    İlk çağrı yapılandırır; sonrakiler yalnız cihaz döndürür (süreç-genelinde bir kez).
    torch yoksa hiçbir şey yapmaz, `cpu` döner.
    """
    resolved = resolve_device(device)
    if not torch_available():
        return "cpu"

    import numpy as np
    import torch

    global _configured
    if _configured:
        return resolved

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda():
        torch.cuda.manual_seed_all(seed)
        # Tensor Core'lu GPU'larda (RTX 40xx) matmul hızlandırması — determinizmi bozmaz
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

    if mode != "off":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(mode == "strict", warn_only=(mode == "best_effort"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[torch] use_deterministic_algorithms(%s): %s", mode, exc)

    _configured = True
    logger.info(
        "[torch] seed=%d mode=%s device=%s (%s)",
        seed, mode, resolved, cuda_device_name() or "cpu",
    )
    return resolved


def _reset_for_tests() -> None:  # pragma: no cover - yalnız test
    global _configured
    _configured = False
    has_cuda.cache_clear()
    torch_available.cache_clear()
