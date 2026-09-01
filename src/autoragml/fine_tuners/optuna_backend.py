"""OptunaTuner — opsiyonel HPO backend'i (`[hpo]` extra, ADR 0013).

TPE sampler ile `candidate.search_space` + `candidate_ops` seçimini birlikte arar.
**Not (v1 sınırı):** ara-adım (`trial.report`) entegrasyonu yok — `HyperbandPruner`
etkin fidelity budama yapmaz (rapor çağrısı olmadan pruner hiç tetiklenmez); v1'de
sabit fidelity ile TPE. Gerçek multi-fidelity pruning callback entegrasyonu v1.1.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate, SearchDim
from autoragml.contracts.enums import HpoBackend, HpoLevel
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.tuning import Trial, TuningResult
from autoragml.exceptions import AutoRagMLError
from autoragml.fine_tuners.inner_eval import build_inner_splits, evaluate_trial
from autoragml.fine_tuners.random_search import _fidelity_bounds, _has_search_surface, _n_trials
from autoragml.validators.runner import DefaultTuner, TunerOutcome


class OptunaMissingError(AutoRagMLError):
    """`optuna` kurulu değil — `pip install 'autoragml[hpo]'`."""


def _suggest(trial: Any, name: str, dim: SearchDim) -> Any:
    if dim.type == "int":
        return trial.suggest_int(name, int(dim.low or 0), int(dim.high or 1))
    if dim.type == "float":
        return trial.suggest_float(name, float(dim.low or 0.0), float(dim.high or 1.0))
    if dim.type == "loguniform":
        return trial.suggest_float(name, float(dim.low or 1e-6), float(dim.high or 1.0), log=True)
    if dim.type == "categorical":
        return trial.suggest_categorical(name, list(dim.choices or []))
    msg = f"Bilinmeyen SearchDim.type: {dim.type!r}"
    raise ValueError(msg)


class OptunaTuner:
    """TPE tabanlı opsiyonel HPO backend'i."""

    def __init__(self, *, inner_folds: int = 1) -> None:
        self.inner_folds = inner_folds

    def tune(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
    ) -> TunerOutcome:
        if config.hpo_level is HpoLevel.NONE or not _has_search_surface(candidate, plan):
            return DefaultTuner().tune(candidate, frame, plan, task, ctx, config)

        try:
            import optuna
        except ModuleNotFoundError as exc:
            msg = "OptunaTuner için `optuna` gerekli: pip install 'autoragml[hpo]'"
            raise OptunaMissingError(msg) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_splits = build_inner_splits(frame, task, config, self.inner_folds)
        fidelity_bounds = _fidelity_bounds(candidate)
        fixed_fidelity = fidelity_bounds[1] if fidelity_bounds else None

        trials: list[Trial] = []
        start = time.perf_counter()

        def objective(trial: optuna.Trial) -> float:
            params = {name: _suggest(trial, name, dim) for name, dim in candidate.search_space.items()}
            choices = {
                g.group_name: trial.suggest_categorical(f"__choice__{g.group_name}", g.choices)
                for g in plan.candidate_ops
            }
            t0 = time.perf_counter()
            score = evaluate_trial(
                candidate, frame, plan, task, ctx, config, params, choices, inner_splits,
                fidelity_value=fixed_fidelity,
            )
            trials.append(
                Trial(
                    number=trial.number, params=params, value=score,
                    fidelity=float(fixed_fidelity) if fixed_fidelity else None,
                    elapsed_seconds=round(time.perf_counter() - t0, 4),
                )
            )
            return score

        sampler = optuna.samplers.TPESampler(seed=config.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=_n_trials(config),
            timeout=config.budget.per_model_max_seconds,
            show_progress_bar=False,
        )

        best = study.best_trial
        best_params = {k: v for k, v in best.params.items() if not k.startswith("__choice__")}
        best_choices = {
            k[len("__choice__") :]: v for k, v in best.params.items() if k.startswith("__choice__")
        }
        if fixed_fidelity and candidate.fidelity:
            best_params[candidate.fidelity] = fixed_fidelity

        elapsed_total = time.perf_counter() - start
        result = TuningResult(
            candidate_key=candidate.key,
            best_params=best_params,
            trials=trials,
            spent_budget={"trials": float(len(trials)), "seconds": elapsed_total},
            realized_seconds=round(elapsed_total, 4),
            fidelity_schedule=[float(fixed_fidelity)] if fixed_fidelity else [],
            backend=HpoBackend.OPTUNA,
            hpo_level=config.hpo_level,
        )
        return TunerOutcome(
            best_params=best_params, candidate_choices=best_choices, nested=True, tuning_result=result
        )
