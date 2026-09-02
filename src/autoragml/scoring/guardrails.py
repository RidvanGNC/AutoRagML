"""Model seçim guardrail'leri — quarantine bayrakları (ADR 0014, DemandSensing deseni).

Boş liste = temiz. Dolu → `ScoreRow.is_quarantined=True`, `selection_eligible=False`.
Tümü karantinaya alınırsa `selection` ham sıralamaya düşer (uyarı ile).
"""

from __future__ import annotations

import math

from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.contracts.validation import ValidationReport
from autoragml.scoring.metrics import default_primary_metric

_METRIC_CEILINGS = (("smape", "smape_mean_max"), ("rmse", "rmse_mean_max"), ("wmape", "wmape_mean_max"))


_NEG_FRAC_HARD = 0.5  # bu oranın üstünde negatif → kırpma aktif olsa bile karantina (ADR 0027)


def evaluate_guardrails(
    report: ValidationReport,
    config: RunConfig,
    task: TaskSpec,
    *,
    target_min: float | None,
    serving_clip_lower: float | None = None,
) -> list[str]:
    """Bir aday için quarantine bayraklarını döndür.

    `serving_clip_lower` (ADR 0027): serving'de uygulanacak negatif-olmayan kırpma tabanı.
    `≥ 0` ise `prediction_negative` bayrağı emit edilmez (served tahmin garanti ≥ taban) —
    ancak negatif oranı > %50 ise miskalibre kabul edilip yine karantina.
    """
    g = config.guardrails
    if not g.enabled:
        return []

    flags: list[str] = []
    primary = config.primary_metric or default_primary_metric(task.task)
    pv = report.oof_metrics.get(primary)
    if pv is None or not math.isfinite(pv):
        flags.append(f"non_finite:{primary}")

    ph = report.prediction_health
    if ph.get("n_non_finite", 0.0) > 0:
        flags.append(f"prediction_non_finite:{int(ph['n_non_finite'])}")
    if target_min is not None and target_min >= 0 and ph.get("n_negative", 0.0) > 0:
        served_floor = serving_clip_lower is not None and serving_clip_lower >= 0.0
        if not served_floor or ph.get("frac_negative", 0.0) > _NEG_FRAC_HARD:
            flags.append(f"prediction_negative:{int(ph['n_negative'])}")
    if g.prediction_hard_abs_max is not None and ph.get("pred_abs_max", 0.0) > g.prediction_hard_abs_max:
        flags.append(f"prediction_abs_max>{g.prediction_hard_abs_max:g}")
    if (
        g.prediction_scale_multiplier_max is not None
        and ph.get("pred_scale_ratio", 0.0) > g.prediction_scale_multiplier_max
    ):
        flags.append(f"prediction_scale_ratio>{g.prediction_scale_multiplier_max:g}")

    for metric_key, attr in _METRIC_CEILINGS:
        ceiling = getattr(g, attr)
        value = report.oof_metrics.get(metric_key)
        if ceiling is not None and value is not None and value > ceiling:
            flags.append(f"{metric_key}>{ceiling:g}")
    if g.abs_bias_mean_max is not None:
        bias = report.oof_metrics.get("abs_bias")
        if bias is not None and bias > g.abs_bias_mean_max:
            flags.append(f"abs_bias>{g.abs_bias_mean_max:g}")

    if report.candidate_key in set(g.model_scenario_blocklist.get(report.scenario, [])):
        flags.append(f"blocked:{report.scenario}:{report.candidate_key}")

    if report.leakage.status == "FAIL":
        flags.append("leakage_fail")

    return flags
