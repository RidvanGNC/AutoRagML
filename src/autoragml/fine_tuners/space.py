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


def sample_params(search_space: dict[str, SearchDim], rng: np.random.Generator) -> dict[str, Any]:
    """Tüm arama uzayından bir konfigürasyon örnekle."""
    return {name: sample_value(dim, rng) for name, dim in search_space.items()}


def sample_candidate_choices(
    groups: list[CandidateOpGroup], rng: np.random.Generator
) -> dict[str, str]:
    """Her `candidate_ops` grubu için bir seçenek örnekle."""
    out: dict[str, str] = {}
    for group in groups:
        idx = int(rng.integers(0, len(group.choices)))
        out[group.group_name] = group.choices[idx]
    return out
