# ADR 0014 — ScoreBoard + dürüst model seçimi

**Durum:** Kabul · 2026-09-01

Kaynak: "Winning by Peeking" (arXiv 2608.07303) — test-set seçim yanlılığı, zorlanmayan
bütçe ve sonuç splicing AutoML kıyaslarını şişiriyor. Ayrıca forecasting için MCB /
Diebold-Mariano çoklu karşılaştırma testleri.

## Yedi kural

| # | Kural | Uygulama |
|---|---|---|
| 1 | **Seçim yalnız validation'da, test'te ASLA** | Model/senaryo/HPO seçimi OOF / iç-val skorları. Dış test (veya final holdout) şampiyon için **bir kez**, sadece raporlama. ADR 0011 `multi_test` yapısal engel |
| 2 | **Şampiyonu tüm train'de refit et** | sonra final skor |
| 3 | **Gerçekleşen wall-clock + aday sayısı K raporla** | `realized_seconds`, `n_candidates`, `selection_bias_bound = σ·√(2 ln K)` |
| 4 | **Marjı gürültü tabanıyla karşılaştır** | fold'lar arası metrik SE; < ~1 SE → `statistical_tie` |
| 5 | **Tie eşiği = kesinlik iddiası** | gerçek SE kullan, 1e-6 değil |
| 6 | **Bütçeyi dışarıdan zorla** | öldürülebilir süreç / per-candidate timeout (ADR 0006 runner + 0008), cooperative check değil |
| 7 | **1-SE kuralı** | en iyinin 1 SE'si içindeki **en basit/ucuz/sağlam** modeli seç (`selection_rule: one_std_err` **default**) |

## Forecasting eklentisi (opsiyonel)
`comparison_tests`: **MCB** (Multiple Comparisons with the Best) sıralaması + **Diebold-Mariano**
ikili anlamlılık, top-N aday için.

## DemandSensing'den korunanlar (aynen)
guardrail/quarantine (non-finite metrik, negatif/aşırı tahmin sayısı, metrik tavanları,
`model_scenario_blocklist`), `primary_metric_by_class` (task/demand-class ağırlıklı seçim),
`promotion_rules` (mutlak eşikler: `smape_max`, `abs_bias_max`).

## SelectionResult sözleşmesi
```
rows[]: (model, scenario) -> {
  oof_metric_mean, oof_metric_se, all_metrics_mean,
  guardrail_flags, is_quarantined, selection_eligible,
  realized_seconds, n_trials, best_iteration }
noise_floor: birincil metrik SE
selection_rule: best | one_std_err        (default one_std_err)
champion: {model, scenario, reason, within_1se: [...], statistical_ties: [...]}
selection_bias_bound: σ√(2 ln K)
comparison_tests?: {mcb_ranks, dm_pvalues}   # forecasting, opsiyonel
promotion: {passed, reasons[]}
```

## Sonuç
- `scoring/` alt: `metrics/`, `guardrails.py`, `selection.py` (1-SE + class-weighted),
  `comparison_tests.py` (opsiyonel)
- Test'e tek dokunuş `engines` orkestrasyonunda enforce edilir; `validators` `multi_test`
  ihlalini yakalar
- Bütçe enforcement `engines/runners` (kill) + `fine_tuners` (per-candidate timeout)
