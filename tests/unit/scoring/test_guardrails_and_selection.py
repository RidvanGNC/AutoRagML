"""scoring.guardrails + selection (ADR 0014)."""

from __future__ import annotations

import numpy as np

from autoragml.config import resolve_run_config
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.enums import Modality, Task
from autoragml.contracts.task_spec import TaskSpec
from autoragml.scoring import score_reports
from autoragml.scoring.guardrails import evaluate_guardrails
from tests.unit.scoring._fixtures import make_profile, make_report

_TASK = TaskSpec(task=Task.FORECASTING, modality=Modality.TIMESERIES, targets=["y"], time_col="ds")


def _cfg(**over: object):
    base = {"time_col": "ds", "primary_metric": "smape"}
    return resolve_run_config(target="y", overrides={**base, **over}).config


def _cand(key: str, family: str = "gbdt") -> Candidate:
    return Candidate(
        key=key, name=key, family=family, class_path={"regression": "x.R"},
        modalities=[Modality.TIMESERIES], tasks=[Task.FORECASTING],
    )


def test_guardrail_negative_predictions_when_target_nonneg() -> None:
    rep = make_report("m", smape=20.0, n_negative=5.0)
    flags = evaluate_guardrails(rep, _cfg(), _TASK, target_min=0.0)
    assert any("prediction_negative" in f for f in flags)
    # hedef negatif olabiliyorsa bayrak yok
    assert not any("prediction_negative" in f for f in evaluate_guardrails(rep, _cfg(), _TASK, target_min=-10.0))


def test_guardrail_negative_skipped_when_serving_clips_nonneg() -> None:
    """ADR 0027: serving'de 0'a kırpılacak küçük negatifler → karantina yok."""
    rep = make_report("m", smape=20.0, n_negative=50.0, frac_negative=0.03)
    flags = evaluate_guardrails(rep, _cfg(), _TASK, target_min=0.0, serving_clip_lower=0.0)
    assert not any("prediction_negative" in f for f in flags)
    # kırpma yoksa (postprocess kapalı) yine karantina
    flags_noclip = evaluate_guardrails(rep, _cfg(), _TASK, target_min=0.0, serving_clip_lower=None)
    assert any("prediction_negative" in f for f in flags_noclip)


def test_guardrail_negative_still_flags_when_mostly_negative() -> None:
    """ADR 0027: negatif oranı > %50 → kırpma aktif olsa bile miskalibre → karantina."""
    rep = make_report("m", smape=20.0, n_negative=600.0, frac_negative=0.6)
    flags = evaluate_guardrails(rep, _cfg(), _TASK, target_min=0.0, serving_clip_lower=0.0)
    assert any("prediction_negative" in f for f in flags)


def test_guardrail_metric_ceiling_and_blocklist() -> None:
    rep = make_report("m", smape=95.0)
    cfg = _cfg(guardrails={"smape_mean_max": 50.0, "model_scenario_blocklist": {"scenario_1": ["m"]}})
    flags = evaluate_guardrails(rep, cfg, _TASK, target_min=0.0)
    assert any("smape>" in f for f in flags)
    assert any("blocked:" in f for f in flags)


def test_guardrail_leakage_and_disabled() -> None:
    rep = make_report("m", smape=10.0, leakage_fail=True)
    assert "leakage_fail" in evaluate_guardrails(rep, _cfg(), _TASK, target_min=0.0)
    assert evaluate_guardrails(rep, _cfg(guardrails={"enabled": False}), _TASK, target_min=0.0) == []


def test_one_se_rule_picks_simplest() -> None:
    reports = [
        make_report("lightgbm", smape=10.0, se=2.0),
        make_report("ridge", smape=10.8, se=2.0),  # 1 SE (2.0) içinde
        make_report("dummy_mean", smape=25.0, se=2.0),  # dışında
    ]
    cands = [_cand("lightgbm", "gbdt"), _cand("ridge", "linear"), _cand("dummy_mean", "baseline")]
    sel = score_reports(reports, cands, _cfg(), _TASK, make_profile())
    assert sel.champion.model_key == "ridge"  # gbdt ile 1-SE bandında, ama linear daha basit
    assert "lightgbm" in sel.champion.within_1se
    assert "dummy_mean" not in sel.champion.within_1se


def test_se_band_filter_excludes_unstable_candidate() -> None:
    """ADR 0038: CV'si çok kararsız (SE >> band) aday, ortalaması bantta olsa da 1-SE'ye girmez."""
    reports = [
        make_report("stat_a", smape=11.5, se=0.5),    # en iyi, kararlı
        make_report("stat_b", smape=11.8, se=0.5),    # bantta, kararlı
        make_report("lightgbm", smape=11.9, se=3.0),  # ortalama bantta AMA SE = 3.0 (2·band'ı aşar)
    ]
    cands = [
        _cand("stat_a", "statistical"), _cand("stat_b", "statistical"), _cand("lightgbm", "gbdt"),
    ]
    sel = score_reports(reports, cands, _cfg(), _TASK, make_profile())
    assert "lightgbm" not in sel.champion.within_1se  # kararsız → dışlandı
    assert sel.champion.model_key in {"stat_a", "stat_b"}


def test_best_rule_picks_lowest_metric() -> None:
    reports = [make_report("a", smape=10.0, se=2.0), make_report("b", smape=10.8, se=2.0)]
    cands = [_cand("a", "gbdt"), _cand("b", "linear")]
    sel = score_reports(reports, cands, _cfg(selection_rule="best"), _TASK, make_profile())
    assert sel.champion.model_key == "a"


def test_all_quarantined_fallback() -> None:
    reports = [make_report("a", smape=200.0, leakage_fail=True), make_report("b", smape=180.0, leakage_fail=True)]
    cands = [_cand("a"), _cand("b")]
    sel = score_reports(reports, cands, _cfg(), _TASK, make_profile())
    assert sel.champion.model_key == "b"  # ham sıralamada daha iyi
    assert "guardrail" in sel.champion.reason


def test_promotion_fails_on_smape() -> None:
    reports = [make_report("a", smape=95.0, se=1.0)]
    sel = score_reports(reports, [_cand("a")], _cfg(), _TASK, make_profile())
    assert sel.promotion.passed is False
    assert any("smape" in r for r in sel.promotion.reasons)


def test_promotion_gate_follows_primary_metric() -> None:
    """Bugfix: `smape_max` tavanı primary metriğe uygulanır (kesikli talepte sMAPE anlamsız)."""
    reports = [make_report("a", smape=200.0, se=1.0)]  # make_report: wmape = smape*0.9 = 180
    # primary=wmape → 180 > 35 → FAIL ama sebep wmape, smape değil
    sel = score_reports(reports, [_cand("a")], _cfg(primary_metric="wmape"), _TASK, make_profile())
    assert not sel.promotion.passed
    assert any("wmape" in r for r in sel.promotion.reasons)
    assert not any(r.startswith("smape") for r in sel.promotion.reasons)
    # primary=rmse (yüzde metrik değil) → yüzde-hata tavanı hiç uygulanmaz
    sel_rmse = score_reports(reports, [_cand("a")], _cfg(primary_metric="rmse"), _TASK, make_profile())
    assert not any("smape" in r or "wmape" in r for r in sel_rmse.promotion.reasons)


def test_promotion_passes() -> None:
    reports = [make_report("a", smape=12.0, se=1.0)]
    sel = score_reports(reports, [_cand("a")], _cfg(), _TASK, make_profile())
    assert sel.promotion.passed is True


def test_scoreboard_fields() -> None:
    reports = [make_report("a", smape=10.0, se=1.5), make_report("b", smape=12.0, se=2.0)]
    sel = score_reports(reports, [_cand("a"), _cand("b")], _cfg(), _TASK, make_profile())
    board = sel.scoreboard
    assert board.n_candidates == 2
    assert board.noise_floor > 0
    assert board.selection_bias_bound == board.noise_floor * np.sqrt(2 * np.log(2))
    assert board.rows[0].selection_eligible  # eligible önce
    assert board.comparison_tests is not None  # forecasting + 4 fold
