"""fine_tuners — HPO (ensemble-öncelikli, multi-fidelity, nested) — ADR 0013.

`resolve_tuner(config) -> Tuner` (`validators.Tuner` protokolünü gerçekler):
- `hpo_level=none` → `DefaultTuner` (plan varsayılanları, arama yok)
- `hpo_level=light` → `RandomSearchTuner` (tek iç holdout, ≤15 deneme)
- `hpo_level=thorough` → `RandomSearchTuner`/`OptunaTuner` (3-fold iç CV, tam bütçe)

`Candidate.fidelity` varsa Successive Halving zamanlaması (RandomSearch); Optuna
backend'i sabit fidelity + TPE (`[hpo]` extra, opsiyonel).
"""

from __future__ import annotations

from autoragml.contracts.enums import HpoBackend, HpoLevel
from autoragml.contracts.run_config import RunConfig
from autoragml.fine_tuners.halving import Rung, build_schedule
from autoragml.fine_tuners.inner_eval import build_inner_splits, evaluate_trial, resolve_primary_metric
from autoragml.fine_tuners.optuna_backend import OptunaMissingError, OptunaTuner
from autoragml.fine_tuners.random_search import RandomSearchTuner
from autoragml.fine_tuners.space import SpaceError, sample_candidate_choices, sample_params
from autoragml.validators.runner import DefaultTuner, Tuner

__all__ = [
    "DefaultTuner",
    "OptunaMissingError",
    "OptunaTuner",
    "RandomSearchTuner",
    "Rung",
    "SpaceError",
    "Tuner",
    "build_inner_splits",
    "build_schedule",
    "evaluate_trial",
    "resolve_primary_metric",
    "resolve_tuner",
    "sample_candidate_choices",
    "sample_params",
]


def resolve_tuner(config: RunConfig) -> Tuner:
    """`RunConfig.hpo_level` + `hpo_backend` → hazır `Tuner`."""
    if config.hpo_level is HpoLevel.NONE:
        return DefaultTuner()
    # ADR 0022: tek iç fold → yüksek varyanslı config seçimi (val'a aşırı-uyum).
    # `light` bile en az 2 iç fold kullanır; `thorough` 3.
    inner_folds = 2 if config.hpo_level is HpoLevel.LIGHT else 3
    if config.hpo_backend is HpoBackend.OPTUNA:
        return OptunaTuner(inner_folds=inner_folds)
    return RandomSearchTuner(inner_folds=inner_folds)
