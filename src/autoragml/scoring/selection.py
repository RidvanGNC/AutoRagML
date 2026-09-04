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
    "glassbox": 2,        # EBM — yorumlanabilir GAM; eşitlikte GBDT'ye tercih (ADR 0040)
    "forest": 3,
    "gbdt": 3,
    "ml": 3,
    "neural": 4,
    "neural_ts": 4,       # ADR 0039
    "foundation": 2,      # in-context / zero-shot tablo — fit yok, operasyonel basit (ADR 0039)
    "foundation_ts": 4,   # ADR 0042: neural_ts ile eşit — GPU + opak pretrained; 1-SE bandında klasiği tercih et
    "ensemble": 5,  # tek model eşitse tek model kazanır (ADR 0021)
    "stack": 6,  # en karmaşık — L2 stacker eşitse L1/ensemble kazanır (ADR 0034)
}

# ADR 0042: OOF'u kırılgan aileler + minimum fold eşiği.
# - foundation_ts (Chronos/TimesFM): ön-eğitim korpusu M3/M4/tourism benchmark'larını içerir →
#   rolling-origin OOF "ezberden" iyimser (m3: OOF 8 → holdout 17; tourism: OOF 18 → holdout 27).
# - neural_ts: yalnız 2 pencerede eğitilir → ince kanıt.
# Bu ailelerden bir aday ancak en iyi "güvenilir" adayı (bu kümede değil + n_folds≥3) `max(SE)`
# marjıyla belirgin geçerse şampiyon; aksi halde güvenilir rakip seçilir. `foundation` (sentetik-veri
# ön-eğitimli tablo: TabPFN/TabICL) kontaminasyon riski taşımaz → kümede DEĞİL.
_THIN_OOF_FAMILIES = frozenset({"foundation_ts", "neural_ts"})
_RELIABLE_FOLDS = 3

# ADR 0039: 1-SE basitlik ikamesi en fazla bu kadar göreli OOF maliyetine izin verir.
# m5 smooth: chronos 51.7 → auto_theta 53.6 (%3.6) — basitlik bunu feda etmemeli.
_SIMPLICITY_MAX_COST = 0.03


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


_SE_BAND_MULT = 2.0  # ADR 0038: kararsız-CV filtresi — r.se > mult·band ise bantta değil


def _untrusted_oof(row: ScoreRow) -> bool:
    """OOF'u gerçek holdout'u tahmin etmeyebilir: kontamine-ön-eğitimli forecaster ya da ince kanıt."""
    return row.family in _THIN_OOF_FAMILIES or row.n_folds < _RELIABLE_FOLDS


def _reliable_best(pool_sorted: list[ScoreRow], raw_best: ScoreRow) -> ScoreRow:
    """ADR 0042 — ince-kanıt / kontamine-OOF marj guard'ı.

    OOF-en-iyi aday `_untrusted_oof` ise, en iyi **güvenilir** adayı (`_untrusted_oof` değil)
    `max(kendi SE, rakip SE)` marjıyla **belirgin** geçerse `raw_best` kalır. Aksi halde o
    güvenilir rakip `best` olur — 1-SE kuralı oradan devam eder; ince-kanıtlı aday `within`
    bandına girse de %3 basitlik tavanında elenir.

    Benchmark: tourism timesfm OOF 18.21 vs joint_ensemble 19.64 (marj 1.43 < SE 1.70) → düşürülür
    (v4 auto_ets +%15.9 davranışına döner). m5 (marj 7.36 > 5.98) ve m4_hourly (9.2 ≫ 0.03) → korunur.
    """
    if not _untrusted_oof(raw_best):
        return raw_best
    rival = next((r for r in pool_sorted if not _untrusted_oof(r)), None)
    if rival is None:
        return raw_best  # kıyaslanacak güvenilir aday yok — ham en iyiyi koru
    margin = abs(raw_best.oof_metric_mean - rival.oof_metric_mean)
    need = max(raw_best.oof_metric_se, rival.oof_metric_se)
    return raw_best if margin > need else rival


def _within_one_se(pool: list[ScoreRow], best: ScoreRow, noise_floor: float, lower: bool) -> list[ScoreRow]:
    """1-SE bandı: en iyinin `max(best.SE, noise_floor)` toleransı içindeki **güvenilir** adaylar.

    - Band = yalnız en iyi modelin SE'si (medyan-SE tabanıyla). ADR 0035/K3 "aday-başı SE"
      (`max(best.se, r.se)`) geri alındı — gürültülü aday kendini içeri alıyordu.
    - **Kararsız-CV filtresi (ADR 0038):** `r.oof_metric_se > _SE_BAND_MULT·band` olan aday
      banttan dışlanır (`best` hariç). m3 benchmark: `lightgbm` sMAPE 12.25 ± 1.76 (band 0.59),
      klasik 11.7 ± 0.03 ile "eşdeğer" sayılıp seçiliyordu → gerçek holdout sMAPE 44.
    """
    band = max(best.oof_metric_se, noise_floor)
    if band <= 0:
        return [best]
    b = best.oof_metric_mean
    max_se = _SE_BAND_MULT * band
    within: list[ScoreRow] = []
    for r in pool:
        if r.model_key != best.model_key and r.oof_metric_se > max_se:
            continue  # CV tahmini bandın çok ötesinde belirsiz → güvenilir biçimde eşdeğer değil
        ok = r.oof_metric_mean <= b + band if lower else r.oof_metric_mean >= b - band
        if ok:
            within.append(r)
    return within or [best]


def select_champion(
    rows: list[ScoreRow],
    reports: list[ValidationReport],
    config: RunConfig,
    task: TaskSpec,
    *,
    noise_floor: float,
    sparse_frac: float = 0.0,
) -> tuple[ChampionInfo, PromotionResult]:
    """Sıralı `ScoreRow` listesinden şampiyon + promotion sonucu."""
    primary = config.primary_metric or default_primary_metric(task.task)
    lower = lower_is_better(primary)

    eligible = [r for r in rows if r.selection_eligible]
    pool = eligible or rows
    fallback_warning = not eligible

    pool_sorted = sorted(pool, key=lambda r: _rank_value(r, lower))
    raw_best = pool_sorted[0]
    best = _reliable_best(pool_sorted, raw_best)  # ADR 0042: ince-kanıt marj guard'ı
    demoted = best.model_key != raw_best.model_key
    within = _within_one_se(pool_sorted, best, noise_floor, lower)

    if config.selection_rule is SelectionRule.ONE_STD_ERR:
        # Basitlik (aile karmaşıklığı) → süre. (ADR 0035/K3 `-n_folds` tie-break GERİ ALINDI.)
        # ADR 0039: basitlik ikamesi yalnız en iyiden ≤ %3 göreli maliyetliyse — bariz-daha-iyi
        # foundation/nöral OOF'u feda etme.
        b = best.oof_metric_mean
        cap = abs(b) * _SIMPLICITY_MAX_COST if math.isfinite(b) and b != 0 else float("inf")
        elig = [r for r in within if r.model_key == best.model_key or abs(r.oof_metric_mean - b) <= cap]
        champ = min(
            elig,
            key=lambda r: (_FAMILY_COMPLEXITY.get(r.family, 3), r.realized_seconds),
        )
        reason = (
            f"1-SE kuralı: en iyinin ~{max(best.oof_metric_se, noise_floor):.3g} SE'si içindeki "
            f"{len(within)} adaydan (≤%{_SIMPLICITY_MAX_COST * 100:.0f} maliyetli {len(elig)}) "
            f"en basiti ({champ.family})"
        )
    else:
        champ = best
        reason = f"En iyi {primary} = {best.oof_metric_mean:.4g}"
    if demoted:
        reason += (
            f" [ADR 0042: OOF-en-iyi `{raw_best.model_key}` yalnız {raw_best.n_folds} pencerede "
            f"doğrulandı, `{best.model_key}` ({best.n_folds}+ fold) belirgin marjla geçilmedi]"
        )
    if fallback_warning:
        reason += " [tüm adaylar guardrail'e takıldı — ham sıralamadan seçildi]"

    ties = [r.model_key for r in within if r.model_key != champ.model_key]
    within_keys = [r.model_key for r in within]

    champ_report = next((r for r in reports if r.candidate_key == champ.model_key), None)
    promotion = _evaluate_promotion(champ, champ_report, config, sparse_frac)

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
    champ: ScoreRow, report: ValidationReport | None, config: RunConfig, sparse_frac: float = 0.0
) -> PromotionResult:
    p = config.promotion
    reasons: list[str] = []
    m = champ.all_metrics_mean

    # `smape_max` aslında bir "yüzde-hata tavanı" (ADR 0014, DemandSensing). Kesikli talepte sMAPE
    # y≈0'da patlar → wMAPE koşumunda "smape 145 > 35" yanlış/anlamsız. Tavanı **primary metriğe**
    # uygula (sMAPE-benzeri yüzde metrikler): sMAPE / wMAPE / MAPE. Diğer primary → tavan atlanır.
    # ADR 0039: panel intermittency-baskın (≥ segment_sparse_min_frac) → wMAPE doğal 50-100, tavan atlanır.
    pct_metric = (config.primary_metric or "smape").lower()
    sparse_dominant = sparse_frac >= config.dynamics.segment_sparse_min_frac
    if (
        p.smape_max is not None
        and pct_metric in {"smape", "wmape", "mape"}
        and not sparse_dominant
    ):
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
