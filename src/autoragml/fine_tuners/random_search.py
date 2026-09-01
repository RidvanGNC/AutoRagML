"""RandomSearchTuner — çekirdek HPO backend'i (ADR 0013).

Ensemble-öncelikli felsefe: HPO ikincil rafinasyon. `Candidate.fidelity` varsa
Successive Halving zamanlaması (ADR 0013 kaynak: multi-fidelity HPO); yoksa düz random
search. Bütçe: `RunConfig.budget` — kooperatif zorlama (tam kill `engines/runners` işi,
ADR 0014/6); ilk deneme sonrası projeksiyon uyarısı (ADR 0008/1).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import HpoBackend, HpoLevel
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.tuning import Trial, TuningResult
from autoragml.fine_tuners.halving import Rung, build_schedule
from autoragml.fine_tuners.inner_eval import IdxPair, build_inner_splits, evaluate_trial
from autoragml.fine_tuners.space import sample_candidate_choices, sample_params
from autoragml.logging import get_logger
from autoragml.validators.runner import DefaultTuner, TunerOutcome

logger = get_logger(__name__)

_MIN_FIDELITY_FRACTION = 0.1
Config = tuple[dict[str, Any], dict[str, str]]


def _has_search_surface(candidate: Candidate, plan: AdaptivePlan) -> bool:
    has_choices = any(len(g.choices) > 1 for g in plan.candidate_ops)
    return bool(candidate.search_space) or has_choices


def _fidelity_bounds(candidate: Candidate) -> tuple[int, int] | None:
    if not candidate.fidelity:
        return None
    max_fid = candidate.default_params.get(candidate.fidelity)
    if not isinstance(max_fid, (int, float)) or max_fid <= 1:
        return None
    max_fid = int(max_fid)
    min_fid = max(10, int(max_fid * _MIN_FIDELITY_FRACTION))
    return (min_fid, max_fid) if min_fid < max_fid else None


def _n_trials(config: RunConfig) -> int:
    budget = config.budget
    ceiling = 15 if config.hpo_level is HpoLevel.LIGHT else budget.max_trials_per_model
    return max(budget.min_trials_per_model, min(budget.max_trials_per_model, ceiling))


class RandomSearchTuner:
    """Çekirdek HPO: random search + (varsa) Successive Halving."""

    def __init__(self, *, inner_folds: int = 1, eta: float = 3.0) -> None:
        self.inner_folds = inner_folds
        self.eta = eta

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

        rng = np.random.default_rng(config.seed)
        n_trials = _n_trials(config)
        inner_splits = build_inner_splits(frame, task, config, self.inner_folds)
        configs: list[Config] = [
            (sample_params(candidate.search_space, rng), sample_candidate_choices(plan.candidate_ops, rng))
            for _ in range(n_trials)
        ]

        bounds = _fidelity_bounds(candidate)
        start = time.perf_counter()
        trials: list[Trial] = []

        if bounds is not None:
            schedule = build_schedule(len(configs), bounds[0], bounds[1], eta=self.eta)
            best_idx = self._run_halving(
                candidate, frame, plan, task, ctx, config, configs, inner_splits, schedule, trials
            )
        else:
            schedule = []
            best_idx = self._run_plain(
                candidate, frame, plan, task, ctx, config, configs, inner_splits, start, trials
            )

        best_params, best_choices = configs[best_idx]
        if schedule and candidate.fidelity:
            best_params = {**best_params, candidate.fidelity: schedule[-1].fidelity}

        elapsed = time.perf_counter() - start
        result = TuningResult(
            candidate_key=candidate.key,
            best_params=best_params,
            trials=trials,
            spent_budget={"trials": float(len(trials)), "seconds": elapsed},
            realized_seconds=round(elapsed, 4),
            fidelity_schedule=[float(r.fidelity) for r in schedule],
            backend=HpoBackend.RANDOM_SEARCH,
            hpo_level=config.hpo_level,
        )
        return TunerOutcome(
            best_params=best_params, candidate_choices=best_choices, nested=True, tuning_result=result
        )

    def _run_plain(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
        configs: list[Config],
        inner_splits: list[IdxPair],
        start: float,
        trials: list[Trial],
    ) -> int:
        scores: dict[int, float] = {}
        deadline = config.budget.per_model_max_seconds
        for i, (params, choices) in enumerate(configs):
            t0 = time.perf_counter()
            scores[i] = evaluate_trial(
                candidate, frame, plan, task, ctx, config, params, choices, inner_splits
            )
            dt = time.perf_counter() - t0
            trials.append(Trial(number=i, params=params, value=scores[i], elapsed_seconds=round(dt, 4)))

            if i == 0 and dt * len(configs) > config.budget.runtime_projection_warn_seconds:
                logger.warning(
                    "[fine_tuners] `%s`: tahmini toplam HPO süresi ~%.0fs (%d deneme).",
                    candidate.key,
                    dt * len(configs),
                    len(configs),
                )
            if deadline and (time.perf_counter() - start) > deadline:
                logger.info(
                    "[fine_tuners] `%s`: per_model_max_seconds doldu (%d/%d).",
                    candidate.key,
                    i + 1,
                    len(configs),
                )
                break
        return min(scores, key=lambda k: scores[k])

    def _run_halving(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
        configs: list[Config],
        inner_splits: list[IdxPair],
        schedule: list[Rung],
        trials: list[Trial],
    ) -> int:
        survivors = list(range(len(configs)))
        last_scores: dict[int, float] = {}
        trial_no = 0
        for rung in schedule:
            rung_scores: list[tuple[int, float]] = []
            for i in survivors:
                params, choices = configs[i]
                t0 = time.perf_counter()
                score = evaluate_trial(
                    candidate, frame, plan, task, ctx, config, params, choices, inner_splits,
                    fidelity_value=rung.fidelity,
                )
                trials.append(
                    Trial(
                        number=trial_no,
                        params=params,
                        value=score,
                        fidelity=float(rung.fidelity),
                        elapsed_seconds=round(time.perf_counter() - t0, 4),
                    )
                )
                trial_no += 1
                rung_scores.append((i, score))
            rung_scores.sort(key=lambda t: t[1])
            survivors = [i for i, _ in rung_scores[: rung.keep]]
            last_scores.update(dict(rung_scores))
        return min(survivors, key=lambda i: last_scores[i])
