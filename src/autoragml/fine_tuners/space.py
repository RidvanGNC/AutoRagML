"""Arama uzayı örnekleme — `SearchDim` → somut değer (ADR 0013)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from autoragml.contracts.adaptive_plan import CandidateOpGroup
from autoragml.contracts.candidate import SearchDim
from autoragml.exceptions import AutoRagMLError


class SpaceError(AutoRagMLError):
    """Geçersiz arama uzayı tanımı."""


def sample_value(dim: SearchDim, rng: np.random.Generator) -> Any:
    """Tek bir `SearchDim`'den örnekle."""
    if dim.type == "int":
        if dim.low is None or dim.high is None:
            msg = "int boyutu için low/high zorunlu"
            raise SpaceError(msg)
        return int(rng.integers(int(dim.low), int(dim.high) + 1))
    if dim.type == "float":
        if dim.low is None or dim.high is None:
            msg = "float boyutu için low/high zorunlu"
            raise SpaceError(msg)
        return float(rng.uniform(dim.low, dim.high))
    if dim.type == "loguniform":
        if not dim.low or not dim.high or dim.low <= 0:
            msg = "loguniform boyutu için pozitif low/high zorunlu"
            raise SpaceError(msg)
        lo, hi = math.log(dim.low), math.log(dim.high)
        return float(math.exp(rng.uniform(lo, hi)))
    if dim.type == "categorical":
        if not dim.choices:
            msg = "categorical boyutu için choices zorunlu"
            raise SpaceError(msg)
        return dim.choices[int(rng.integers(0, len(dim.choices)))]
    msg = f"Bilinmeyen SearchDim.type: {dim.type!r}"
    raise SpaceError(msg)


def _condition_met(condition: dict[str, object], sampled: dict[str, Any]) -> bool:
    """`{"param": ad, "eq"|"ne"|"ge"|"in": değer}` — koşul sağlandı mı (ADR 0031)."""
    param = condition.get("param")
    if not isinstance(param, str) or param not in sampled:
        return False
    val = sampled[param]
    if "eq" in condition:
        return bool(val == condition["eq"])
    if "ne" in condition:
        return bool(val != condition["ne"])
    if "ge" in condition:
        try:
            return bool(val >= condition["ge"])
        except TypeError:
            return False
    if "in" in condition:
        allowed = condition["in"]
        return isinstance(allowed, (list, tuple, set)) and val in allowed
    return False


def sample_params(search_space: dict[str, SearchDim], rng: np.random.Generator) -> dict[str, Any]:
    """Bir konfigürasyon örnekle. Koşullu boyutlar (ADR 0031) sonra — koşul sağlanmazsa atlanır."""
    out: dict[str, Any] = {}
    pending: list[tuple[str, SearchDim]] = []
    for name, dim in search_space.items():
        if dim.condition:
            pending.append((name, dim))
        else:
            out[name] = sample_value(dim, rng)
    for _ in range(len(pending) + 1):  # koşul zincirleri için birkaç geçiş
        progressed = False
        for name, dim in list(pending):
            assert dim.condition is not None
            cond_param = dim.condition.get("param")
            if _condition_met(dim.condition, out):
                out[name] = sample_value(dim, rng)
                pending.remove((name, dim))
                progressed = True
            elif isinstance(cond_param, str) and cond_param in out:  # değerlendirildi, sağlanmadı
                pending.remove((name, dim))
                progressed = True
        if not progressed:
            break
    return out


def sample_candidate_choices(
    groups: list[CandidateOpGroup], rng: np.random.Generator
) -> dict[str, str]:
    """Her `candidate_ops` grubu için bir seçenek örnekle."""
    out: dict[str, str] = {}
    for group in groups:
        idx = int(rng.integers(0, len(group.choices)))
        out[group.group_name] = group.choices[idx]
    return out
