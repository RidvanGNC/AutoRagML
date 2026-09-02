"""scoring — dürüst model seçimi (ADR 0014).

`score_reports(reports, candidates, config, task, profile) -> SelectionResult`:
1. guardrail/quarantine → `ScoreRow` (eligible/quarantined)
2. class-weighted skor (forecasting + `metric_by_class`)
3. `noise_floor` (fold'lar arası SE) + `selection_bias_bound = σ·√(2 ln K)`
4. **1-SE kuralı** (varsayılan) → en basit/ucuz aday · promotion (mutlak eşikler)
5. MCB / Diebold-Mariano (forecasting, ≥3 fold, opsiyonel)

Seçim **yalnız OOF metriklerinden** — dış test'e dokunulmaz (ADR 0014/1).
"""

from __future__ import annotations

import math

import numpy as np

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.scoreboard import ScoreBoard, ScoreRow, SelectionResult
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import ValidationReport
from autoragml.logging import get_logger
from autoragml.postprocessors.steps import resolve_clip_lower
from autoragml.scoring.comparison_tests import build_comparison_tests
from autoragml.scoring.guardrails import evaluate_guardrails
from autoragml.scoring.metrics import default_primary_metric, lower_is_better
from autoragml.scoring.selection import class_weighted_score, select_champion

logger = get_logger(__name__)

__all__ = ["build_scoreboard", "score_reports"]

_INF = float("inf")
_REGRESSION_TASKS = {
    Task.REGRESSION,
    Task.FORECASTING,
    Task.QUANTILE_REGRESSION,
    Task.ORDINAL_REGRESSION,
}


def _serving_clip_lower(config: RunConfig, task: TaskSpec, target_min: float | None) -> float | None:
    """Serving'de uygulanacak negatif-olmayan kırpma tabanı (ADR 0027)."""
    pp = config.postprocess
    if not pp.enabled:
        return None
    return resolve_clip_lower(
        pp.clip.lower,
        auto_nonneg=pp.clip.auto_nonneg,
        is_regression=task.task in _REGRESSION_TASKS,
        target_min=target_min,
    )


def _best_fold_iteration(report: ValidationReport) -> int | None:
    iters = [fr.best_iteration for fr in report.folds if fr.best_iteration is not None]
    return int(np.median(iters)) if iters else None


def _noise_floor(rows: list[ScoreRow]) -> float:
    ses = [r.oof_metric_se for r in rows if r.oof_metric_se > 0]
    if not ses:
        return 0.0
    return float(np.median(ses))


def build_scoreboard(
    reports: list[ValidationReport],
    candidates: list[Candidate],
    config: RunConfig,
    task: TaskSpec,
    profile: DataProfile,
) -> ScoreBoard:
    """`ValidationReport` listesi → sıralı `ScoreBoard` (henüz şampiyon yok)."""
    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)
    family_by_key = {c.key: c.family for c in candidates}
    target_min = profile.target_profile.stats.min
    serving_clip_lower = _serving_clip_lower(config, task, target_min)

    rows: list[ScoreRow] = []
    for report in reports:
        flags = evaluate_guardrails(
            report, config, task, target_min=target_min, serving_clip_lower=serving_clip_lower
        )
        metric_value = report.oof_metrics.get(primary)
        rows.append(
            ScoreRow(
                model_key=report.candidate_key,
                family=family_by_key.get(report.candidate_key, "ml"),
                scenario=report.scenario,
                oof_metric_mean=metric_value if metric_value is not None and math.isfinite(metric_value)
                else (_INF if lower else -_INF),
                oof_metric_se=report.oof_metric_se.get(primary, 0.0),
                all_metrics_mean=dict(report.oof_metrics),
                guardrail_flags=flags,
                is_quarantined=bool(flags),
                selection_eligible=not flags,
                class_weighted_score=class_weighted_score(report, profile, config, task),
                realized_seconds=report.realized_seconds,
                best_iteration=_best_fold_iteration(report),
            )
        )

    noise_floor = _noise_floor(rows)
    k = len(rows)
    bias_bound = noise_floor * math.sqrt(2.0 * math.log(max(k, 2)))

    rows.sort(
        key=lambda r: (
            not r.selection_eligible,
            r.oof_metric_mean if lower else -r.oof_metric_mean,
        )
    )

    return ScoreBoard(
        rows=rows,
        primary_metric=primary,
        noise_floor=noise_floor,
        n_candidates=k,
        selection_bias_bound=bias_bound,
        comparison_tests=None,
    )


def score_reports(
    reports: list[ValidationReport],
    candidates: list[Candidate],
    config: RunConfig,
    task: TaskSpec,
    profile: DataProfile,
) -> SelectionResult:
    """Uçtan uca: guardrail → skor tablosu → şampiyon + promotion + karşılaştırma testleri."""
    if not reports:
        msg = "score_reports: boş ValidationReport listesi"
        raise ValueError(msg)

    board = build_scoreboard(reports, candidates, config, task, profile)
    champion, promotion = select_champion(
        board.rows, reports, config, task, noise_floor=board.noise_floor
    )

    if task.task is Task.FORECASTING:
        board.comparison_tests = build_comparison_tests(reports, champion.model_key, board.primary_metric)

    logger.info(
        "[scoring] şampiyon=%s (%s) | %s=%.4g | K=%d | σ√(2lnK)=%.3g | promotion=%s",
        champion.model_key,
        champion.reason,
        board.primary_metric,
        next((r.oof_metric_mean for r in board.rows if r.model_key == champion.model_key), 0.0),
        board.n_candidates,
        board.selection_bias_bound,
        "PASS" if promotion.passed else f"FAIL({'; '.join(promotion.reasons)})",
    )
    return SelectionResult(
        scoreboard=board,
        selection_rule=config.selection_rule,
        champion=champion,
        promotion=promotion,
    )
