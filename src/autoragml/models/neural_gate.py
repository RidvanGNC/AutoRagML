"""Nöral aday kapısı (ADR 0030).

`resolve_candidates` katalog + `requires` filtresini yapar; burada **çalışma-zamanı** kapısı:
- `neural_enabled`: `auto` → yalnız GPU varken; `on` → her zaman; `off` → hiç.
- satır sayısı bandı (`neural_min_rows`..`neural_max_rows`; GPU'da tavan 4× esner).
- pytabkit adayları havuza girdiğinde sklearn `mlp` düşürülür (RealMLP > MLP).
- `neural_device` override'ı adayların `default_params`'ına yazılır.
"""

from __future__ import annotations

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.run_config import RunConfig
from autoragml.logging import get_logger
from autoragml.models.torch_env import has_cuda

logger = get_logger(__name__)

_PYTABKIT = "pytabkit"
_ARCH_KEY = "neural_arch_search"


def prepare_neural_candidates(
    candidates: list[Candidate], profile: DataProfile, config: RunConfig
) -> list[Candidate]:
    """Nöral adayları çalışma-zamanı kapısından geçir; `mlp` çakışmasını çöz; cihazı enjekte et."""
    # ADR 0031: mimari arama adayı yalnız neural_search=True iken havuzda
    if not config.neural_search:
        candidates = [c for c in candidates if c.key != _ARCH_KEY]

    torch_neural = [c for c in candidates if _PYTABKIT in c.requires]
    arch = [c for c in candidates if c.key == _ARCH_KEY]
    if not torch_neural and not arch:
        return candidates

    gpu = has_cuda()
    mode = config.neural_enabled
    active = mode == "on" or (mode == "auto" and gpu)

    n_rows = profile.n_rows
    lo = config.neural_min_rows
    hi = config.neural_max_rows
    if hi is None:
        hi = None if gpu else max(lo * 200, 200_000)  # CPU'da güvenli tavan
    elif gpu:
        hi = hi * 4
    row_ok = n_rows >= lo and (hi is None or n_rows <= hi)

    dropped = {c.key for c in [*torch_neural, *arch]}
    if not (active and row_ok):
        reason = (
            "GPU yok / neural_enabled!=on" if not active else f"n_rows={n_rows} bant [{lo}, {hi}] dışı"
        )
        logger.info("[neural] adaylar atlandı (%s): %s", reason, sorted(dropped))
        return [c for c in candidates if c.key not in dropped]

    has_pytabkit = bool(torch_neural)
    logger.info(
        "[neural] %s havuzda (GPU=%s, n_rows=%d)%s",
        sorted(dropped), gpu, n_rows, "; sklearn `mlp` düşürüldü" if has_pytabkit else "",
    )
    device = config.neural_device
    out: list[Candidate] = []
    for c in candidates:
        if c.key == "mlp" and has_pytabkit:
            continue  # RealMLP/TabM havuzdayken sklearn MLP gereksiz
        if (_PYTABKIT in c.requires or c.key == _ARCH_KEY) and device != "auto":
            out.append(c.model_copy(update={"default_params": {**c.default_params, "device": device}}))
        else:
            out.append(c)
    return out
