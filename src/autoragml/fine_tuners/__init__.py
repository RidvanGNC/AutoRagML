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
    """`RunConfig.hpo_level` + `hpo_backend` → hazır `Tuner`.

    `neural_search=True` (ADR 0031): nöral `neural_arch_search` adayına mimari arama, diğer
    adaylara aşağıdaki tuner (heterojen — `ArchitectureSearchTuner` içeride devreder).
    `hpo_level=none` iken bile mimari arama koşar (kullanıcı açıkça istedi).
    """
    if config.hpo_level is HpoLevel.NONE:
        base: Tuner = DefaultTuner()
    else:
        # ADR 0022: tek iç fold → yüksek varyanslı config seçimi. `light` ≥2 fold, `thorough` 3.
        inner_folds = 2 if config.hpo_level is HpoLevel.LIGHT else 3
        base = (
            OptunaTuner(inner_folds=inner_folds)
            if config.hpo_backend is HpoBackend.OPTUNA
            else RandomSearchTuner(inner_folds=inner_folds)
        )
    if config.neural_search:
        from autoragml.fine_tuners.arch_search import make_arch_tuner

        return make_arch_tuner(config, base)
    return base
