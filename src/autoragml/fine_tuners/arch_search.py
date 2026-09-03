"""ArchitectureSearchTuner — nöral mimari arama (ADR 0031).

İki aşama: **A)** aile taraması (MLP/GANDALF/FT-Transformer, kısa epoch) → en iyi ≤2 aile;
**B)** kazanan aile(ler) üzerinde koşullu mimari+HP arama, Successive Halving (epoch = fidelity).
Yalnız `candidate.key == "neural_arch_search"` için; diğer adaylarda `fallback` tuner'a devreder.
Nested CV korunur (ADR 0010/6): tüm arama dış-fold train'in iç resample'ında.
"""

from __future__ import annotations

import time
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from autoragml.config.loaders import load_yaml_text
from autoragml.contracts.adaptive_plan import AdaptivePlan
from autoragml.contracts.candidate import Candidate, SearchDim
from autoragml.contracts.enums import HpoBackend
from autoragml.contracts.plan_context import PlanContext
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.tuning import Trial, TuningResult
from autoragml.fine_tuners.halving import build_schedule
from autoragml.fine_tuners.inner_eval import build_inner_splits, evaluate_trial
from autoragml.fine_tuners.space import sample_params
from autoragml.logging import get_logger
from autoragml.models.neural_arch import FAMILIES
from autoragml.validators.runner import DefaultTuner, Tuner, TunerOutcome

logger = get_logger(__name__)

_ARCH_KEY = "neural_arch_search"
_SWEEP_EPOCHS = 20      # Aşama A: hızlı aile taraması
_MIN_EPOCHS = 15        # Aşama B: SH taban fidelity
_DEFAULT_BUDGET_S = 3600
_SPACE_PKG = "autoragml.fine_tuners._spaces"


def _load_space(name: str) -> dict[str, SearchDim]:
    text = resources.files(_SPACE_PKG).joinpath(f"neural_arch_{name}.yaml").read_text("utf-8")
    raw = load_yaml_text(text, source=f"neural_arch_{name}")
    return {k: SearchDim(**v) for k, v in raw.items()}


def _n_configs(space_name: str) -> int:
    return 12 if space_name == "small" else 24


class ArchitectureSearchTuner:
    """`Tuner` protokolü — nöral mimari arama; diğer adaylarda fallback."""

    def __init__(self, *, fallback: Tuner, inner_folds: int = 2) -> None:
        self._fallback = fallback
        self.inner_folds = inner_folds
        self._cache: dict[str, TunerOutcome] = {}

    def tune(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
    ) -> TunerOutcome:
        if candidate.key != _ARCH_KEY:
            return self._fallback.tune(candidate, frame, plan, task, ctx, config)

        # Mimari arama **bir kez** koşar (ilk dış fold'da); sonraki dış fold'lar aynı mimariyi
        # değerlendirir. Auto-PyTorch deseni — mimari meta-seçimdir, fold'a duyarsız; nested CV
        # dış fold'ların testi aramaya girmediği için sızıntı yok (ADR 0031).
        if candidate.key in self._cache:
            logger.info("[arch_search] önbellekten mimari (dış fold tekrarı)")
            return self._cache[candidate.key]

        start = time.perf_counter()
        budget = config.neural_search_budget_seconds or _DEFAULT_BUDGET_S
        rng = np.random.default_rng(config.seed)
        splits = build_inner_splits(frame, task, config, self.inner_folds)
        choices = {g.group_name: g.default for g in plan.candidate_ops}
        trials: list[Trial] = []
        _me = candidate.default_params.get("max_epochs", 120)
        max_epochs = int(_me) if isinstance(_me, (int, float, str)) else 120

        # --- Aşama A: aile taraması ---
        fam_scores: list[tuple[str, float]] = []
        for fam in FAMILIES:
            t0 = time.perf_counter()
            score = self._eval(candidate, frame, plan, task, ctx, config,
                               {"family": fam}, choices, splits, _SWEEP_EPOCHS)
            fam_scores.append((fam, score))
            trials.append(Trial(number=len(trials), params={"family": fam}, value=score,
                                fidelity=float(_SWEEP_EPOCHS), elapsed_seconds=round(time.perf_counter() - t0, 2)))
            if time.perf_counter() - start > budget * 0.4:
                logger.info("[arch_search] Aşama A bütçesi doldu — %d aile tarandı", len(fam_scores))
                break
        fam_scores.sort(key=lambda t: t[1])
        winners = [f for f, _ in fam_scores[:2]]
        logger.info("[arch_search] kazanan aile(ler): %s (skorlar: %s)", winners, [round(s, 3) for _, s in fam_scores])

        # --- Aşama B: kazanan aile(ler)de mimari+HP arama, SH ---
        space = _load_space(config.neural_search_space)
        n_cfg = _n_configs(config.neural_search_space)
        configs: list[dict[str, Any]] = []
        for _ in range(n_cfg):
            fam = winners[int(rng.integers(0, len(winners)))]
            p = sample_params(space, rng)
            p["family"] = fam
            configs.append(p)

        schedule = build_schedule(len(configs), _MIN_EPOCHS, max_epochs, eta=3.0)
        survivors = list(range(len(configs)))
        last: dict[int, float] = {}
        for rung in schedule:
            for i in list(survivors):
                if time.perf_counter() - start > budget:
                    logger.warning("[arch_search] toplam bütçe (%ds) doldu — arama kesildi", budget)
                    survivors = [j for j in survivors if j in last] or survivors[:1]
                    break
                t0 = time.perf_counter()
                s = self._eval(candidate, frame, plan, task, ctx, config,
                               configs[i], choices, splits, rung.fidelity)
                last[i] = s
                trials.append(Trial(number=len(trials), params=configs[i], value=s,
                                    fidelity=float(rung.fidelity),
                                    elapsed_seconds=round(time.perf_counter() - t0, 2)))
            ranked = sorted((i for i in survivors if i in last), key=lambda i: last[i])
            survivors = ranked[: rung.keep] or ranked[:1]

        best_i = min((i for i in last), key=lambda i: last[i]) if last else 0
        best_params: dict[str, object] = (
            {**configs[best_i], "max_epochs": max_epochs} if configs else {"family": winners[0]}
        )
        elapsed = time.perf_counter() - start
        logger.info(
            "[arch_search] en iyi: family=%s n_layers=%s width=%s (%.1fs, %d deneme)",
            best_params.get("family"), best_params.get("n_layers"), best_params.get("layer_width"),
            elapsed, len(trials),
        )
        result = TuningResult(
            candidate_key=candidate.key,
            best_params=best_params,
            trials=trials,
            spent_budget={"trials": float(len(trials)), "seconds": elapsed},
            realized_seconds=round(elapsed, 3),
            fidelity_schedule=[float(r.fidelity) for r in schedule],
            backend=HpoBackend.RANDOM_SEARCH,
            hpo_level=config.hpo_level,
        )
        outcome = TunerOutcome(
            best_params=best_params, candidate_choices=choices, nested=True, tuning_result=result
        )
        self._cache[candidate.key] = outcome
        return outcome

    def _eval(
        self,
        candidate: Candidate,
        frame: pd.DataFrame,
        plan: AdaptivePlan,
        task: TaskSpec,
        ctx: PlanContext,
        config: RunConfig,
        params: dict[str, Any],
        choices: dict[str, str],
        splits: Any,
        fidelity: float,
    ) -> float:
        try:
            return evaluate_trial(
                candidate, frame, plan, task, ctx, config, params, choices, splits,
                fidelity_value=fidelity,
            )
        except Exception as exc:  # noqa: BLE001 - bir mimari patlarsa arama sürsün
            logger.warning("[arch_search] config başarısız (%s): %s", params.get("family"), exc)
            return float("inf")


def make_arch_tuner(config: RunConfig, fallback: Tuner) -> Tuner:
    """`neural_search` iken `ArchitectureSearchTuner`, değilse fallback."""
    if not config.neural_search:
        return fallback
    inner = 2 if config.hpo_level.value != "thorough" else 3
    return ArchitectureSearchTuner(fallback=fallback or DefaultTuner(), inner_folds=inner)
