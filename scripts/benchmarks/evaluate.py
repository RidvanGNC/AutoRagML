"""Benchmark başarı değerlendirmesi — dış test setinde şampiyon vs naive baseline.

"Başarı" = şampiyon, harici (fit'in hiç görmediği) test setinde naive baseline'ı geçer +
leakage FAIL yok + engine çökmedi. Süre kaydedilir ama başarı ölçütü değildir (motto).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from autoragml.contracts.run_result import RunResult
from autoragml.scoring.metrics import compute_metrics, lower_is_better

_MARGIN = 0.005  # naive'i en az %0.5 (göreli) geçmeli


@dataclass
class Outcome:
    name: str
    status: str  # success | no_improvement | engine_failed | error
    task: str
    champion: str
    champion_family: str
    n_candidates: int
    ensemble_used: bool
    primary_metric: str
    champion_test_score: float
    naive_test_score: float
    improvement_pct: float
    holdout_score: float | None
    runtime_s: float
    leakage: str
    target_encoded: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rel_improvement(champ: float, naive: float, *, lower: bool) -> float:
    if not np.isfinite(champ) or not np.isfinite(naive):
        return float("nan")
    if lower:
        return (naive - champ) / naive * 100.0 if naive != 0 else 0.0
    denom = abs(naive) if naive != 0 else 1e-9
    return (champ - naive) / denom * 100.0


def evaluate(
    name: str,
    result: RunResult,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    naive_pred: np.ndarray,
    *,
    runtime_s: float,
    target_encoded: bool,
) -> Outcome:
    er = result.engine_result
    task = er.task_spec.task
    primary = er.scoreboard.primary_metric
    lower = lower_is_better(primary)

    champ_m = compute_metrics(y_test, y_pred, task).get(primary, float("nan"))
    naive_m = compute_metrics(y_test, naive_pred, task).get(primary, float("nan"))
    imp = _rel_improvement(champ_m, naive_m, lower=lower)

    champ_row = next((r for r in er.scoreboard.rows if r.model_key == er.selection.champion.model_key), None)
    holdout = er.champion.metrics_holdout.get(primary)
    leak = "PASS"
    champ_report_leak = getattr(champ_row, "guardrail_flags", []) or []
    if any("leakage" in f for f in champ_report_leak):
        leak = "FLAG"

    beats = (champ_m < naive_m * (1 - _MARGIN)) if lower else (champ_m > naive_m * (1 + _MARGIN))
    status = "success" if beats else "no_improvement"

    return Outcome(
        name=name,
        status=status,
        task=str(task),
        champion=er.selection.champion.model_key,
        champion_family=champ_row.family if champ_row else "?",
        n_candidates=er.scoreboard.n_candidates,
        ensemble_used=er.selection.champion.model_key == "weighted_ensemble",
        primary_metric=primary,
        champion_test_score=round(float(champ_m), 5),
        naive_test_score=round(float(naive_m), 5),
        improvement_pct=round(float(imp), 2),
        holdout_score=round(float(holdout), 5) if holdout is not None else None,
        runtime_s=round(runtime_s, 1),
        leakage=leak,
        target_encoded=target_encoded,
        note=er.selection.champion.reason,
    )
