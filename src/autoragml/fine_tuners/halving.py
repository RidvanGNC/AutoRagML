"""Successive Halving zamanlayıcısı (ADR 0013).

Her rung'da bütçe `eta` katına çıkar, hayatta kalanlar `1/eta`'ya iner
(Karnin-Koren-Somekh / Hyperband deseni; doğrulandı — bkz. proje notları).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rung:
    """Bir SH aşaması: bu fidelity'de değerlendir, `keep` hayatta kalanı bırak."""

    fidelity: int
    keep: int


def build_schedule(n_configs: int, min_fidelity: int, max_fidelity: int, *, eta: float = 3.0) -> list[Rung]:
    """`n_configs` adaydan başlayıp `max_fidelity`'e kadar SH takvimi kur."""
    min_fidelity = max(1, min_fidelity)
    max_fidelity = max(min_fidelity, max_fidelity)
    if n_configs <= 1 or min_fidelity >= max_fidelity or eta <= 1.0:
        return [Rung(fidelity=max_fidelity, keep=max(1, n_configs))]

    s_max = max(0, int(math.floor(math.log(max_fidelity / min_fidelity, eta))))
    rungs: list[Rung] = []
    n = n_configs
    for i in range(s_max + 1):
        fidelity = min(max_fidelity, int(round(min_fidelity * (eta**i))))
        keep = 1 if i == s_max else max(1, int(math.ceil(n / eta)))
        rungs.append(Rung(fidelity=fidelity, keep=keep))
        n = keep
    if rungs[-1].fidelity != max_fidelity:
        rungs[-1] = Rung(fidelity=max_fidelity, keep=rungs[-1].keep)
    return rungs
