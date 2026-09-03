"""Şampiyon seçimi — 1-SE kuralı + class-weighted skor + promotion (ADR 0014).

Seçim yalnız OOF metriklerinden. `one_std_err` (varsayılan): en iyinin 1 SE'si içindeki
en **basit/ucuz** model. `best`: sadece en iyi metrik. Class-weighted skor DemandSensing
`primary_metric_by_class` desenidir (forecasting).
"""

from __future__ import annotations

import math

import numpy as np

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import SelectionRule, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.scoreboard import ChampionInfo, PromotionResult, ScoreRow
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import ValidationReport
from autoragml.scoring.metrics import compute_metrics, default_primary_metric, lower_is_better

# Basitlik sırası — 1-SE kuralında eşitlik bozucu (küçük = daha basit).
_FAMILY_COMPLEXITY: dict[str, int] = {
    "baseline": 0,
    "linear": 1,
    "statistical": 1,
    "intermittent": 1,
    "distance": 2,
    "forest": 3,
    "gbdt": 3,
    "ml": 3,
    "neural": 4,
    "ensemble": 5,  # tek model eşitse tek model kazanır (ADR 0021)
    "stack": 6,  # en karmaşık — L2 stacker eşitse L1/ensemble kazanır (ADR 0034)
}


def class_weighted_score(
    report: ValidationReport, profile: DataProfile, config: RunConfig, task: TaskSpec
) -> float | None:
    """Talep-sınıfı ağırlıklı skor (minimize yönü). Uygulanamıyorsa `None`."""
    if (
        task.task is not Task.FORECASTING
        or report.oof is None
        or report.oof.group is None
        or profile.timeseries is None
        or not config.metric_by_class
    ):
        return None

    per_series = profile.timeseries.per_series
    if not per_series:
        return None
    group_to_class = {sp.group: sp.intermittency_class.value for sp in per_series}
    total = len(per_series)
    class_weight: dict[str, float] = {}
    for sp in per_series:
        class_weight[sp.intermittency_class.value] = class_weight.get(sp.intermittency_class.value, 0.0) + 1.0 / total

    y_true = report.oof.y_true
    y_pred = report.oof.y_pred
    classes = np.array([group_to_class.get(str(g), "unknown") for g in report.oof.group])

    acc = 0.0
    used_weight = 0.0
    for cls, weight in class_weight.items():
        mask = classes == cls
        if not mask.any():
            continue
        metric_name = config.metric_by_class.get(cls, "wmape")
        metrics = compute_metrics(y_true[mask], y_pred[mask], task.task)
        value = metrics.get(metric_name)
        if value is None or not math.isfinite(value):
            continue
        score = value if lower_is_better(metric_name) else -value
        acc += weight * score
        used_weight += weight
    return acc / used_weight if used_weight > 0 else None


def _rank_value(row: ScoreRow, lower: bool) -> float:
    """Sıralama tek eksende: birincil metrik. `class_weighted_score` v1'de yalnız
    bilgilendirme (per-class SE yok → 1-SE bandına giremez); tam class-weighted seçim v1.1."""
    return row.oof_metric_mean if lower else -row.oof_metric_mean


def _within_one_se(pool: list[ScoreRow], best: ScoreRow, noise_floor: float, lower: bool) -> list[ScoreRow]:
    if noise_floor <= 0:
        return [best]
    if lower:
        return [r for r in pool if r.oof_metric_mean <= best.oof_metric_mean + noise_floor]
    return [r for r in pool if r.oof_metric_mean >= best.oof_metric_mean - noise_floor]


def select_champion(
    rows: list[ScoreRow],
    reports: list[ValidationReport],
    config: RunConfig,
    task: TaskSpec,
    *,
    noise_floor: float,
) -> tuple[ChampionInfo, PromotionResult]:
    """Sıralı `ScoreRow` listesinden şampiyon + promotion sonucu."""
    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)

    eligible = [r for r in rows if r.selection_eligible]
    pool = eligible or rows
    fallback_warning = not eligible

    pool_sorted = sorted(pool, key=lambda r: _rank_value(r, lower))
    best = pool_sorted[0]
    within = _within_one_se(pool_sorted, best, noise_floor, lower)

    if config.selection_rule is SelectionRule.ONE_STD_ERR:
        champ = min(
            within,
            key=lambda r: (_FAMILY_COMPLEXITY.get(r.family, 3), r.realized_seconds),
        )
        reason = (
            f"1-SE kuralı: en iyinin {noise_floor:.3g} SE'si içindeki {len(within)} adaydan "
            f"en basiti ({champ.family})"
        )
    else:
        champ = best
        reason = f"En iyi {primary} = {best.oof_metric_mean:.4g}"
    if fallback_warning:
        reason += " [tüm adaylar guardrail'e takıldı — ham sıralamadan seçildi]"

    ties = [r.model_key for r in within if r.model_key != champ.model_key]
    within_keys = [r.model_key for r in within]

    champ_report = next((r for r in reports if r.candidate_key == champ.model_key), None)
    promotion = _evaluate_promotion(champ, champ_report, config)

    return (
        ChampionInfo(
            model_key=champ.model_key,
            scenario=champ.scenario,
            reason=reason,
            within_1se=within_keys,
            statistical_ties=ties,
        ),
        promotion,
    )


def _evaluate_promotion(
    champ: ScoreRow, report: ValidationReport | None, config: RunConfig
) -> PromotionResult:
    p = config.promotion
    reasons: list[str] = []
    m = champ.all_metrics_mean

    # `smape_max` aslında bir "yüzde-hata tavanı" (ADR 0014, DemandSensing). Kesikli talepte sMAPE
    # y≈0'da patlar → wMAPE koşumunda "smape 145 > 35" yanlış/anlamsız. Tavanı **primary metriğe**
    # uygula (sMAPE-benzeri yüzde metrikler): sMAPE / wMAPE / MAPE. Diğer primary → tavan atlanır.
    pct_metric = (config.primary_metric or "smape").lower()
    if p.smape_max is not None and pct_metric in {"smape", "wmape", "mape"}:
        v = m.get(pct_metric)
        if v is not None and v > p.smape_max:
            reasons.append(f"{pct_metric} {v:.2f} > {p.smape_max}")
    if p.abs_bias_max is not None and (v := m.get("abs_bias")) is not None and v > p.abs_bias_max:
        reasons.append(f"abs_bias {v:.2f} > {p.abs_bias_max}")
    if p.rmse_max is not None and (v := m.get("rmse")) is not None and v > p.rmse_max:
        reasons.append(f"rmse {v:.2f} > {p.rmse_max}")
    if report is not None:
        if len(report.folds) < p.min_folds:
            reasons.append(f"fold sayısı {len(report.folds)} < {p.min_folds}")
        if p.require_leakage_pass and report.leakage.status != "PASS":
            reasons.append("leakage kontrolü PASS değil")
    if champ.is_quarantined:
        reasons.append("guardrail karantinası")

    return PromotionResult(passed=not reasons, reasons=reasons)
