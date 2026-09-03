# ADR 0039 — forecasting seçim ince ayarı (kesikli-talep promotion + basitlik-maliyet tavanı)

**Durum:** Kabul · 2026-09-03 (v1.1 forecasting benchmark v4 gözlemi)

Kaynak: m5 + tourism + m3 tam-stack benchmark'ı iki seçim davranışını ortaya çıkardı.

## Bulgu 1 — promotion `smape_max` kesikli talepte spam FAIL

`promotion.smape_max` (vars. 35.0) primary metriğe uygulanıyor (ADR 0035/bugfix 020f258).
Kesikli talepte wMAPE/sMAPE **doğal olarak 50-100** (y≈0 dinamiği). m5:
`promotion=FAIL(wmape 82.96 > 35.0)` — her segmentte, anlamsız. Kapı DemandSensing'de "absürt
hata → oto-promote etme" sağlık tabanıydı; kesikli talepte wMAPE 80 absürt değil, verinin doğası.

**Karar:** panel **intermittency-baskın** (≥ `dynamics.segment_sparse_min_frac`, vars. 0.5 —
`(intermittent + lumpy) / toplam`) ise yüzde-metrik tavanı **atlanır** (promotion reason'a not
düşülür, FAIL değil). Diğer promotion kapıları (abs_bias, rmse, fold sayısı, leakage, karantina)
aynen çalışır.

## Bulgu 2 — 1-SE basitlik bariz-daha-iyi OOF'u feda ediyor

m5 smooth+erratic leaderboard: `chronos_bolt_small` OOF wMAPE **51.71** (en iyi), ama şampiyon
`auto_theta` **53.57** (statistical=1, 1-SE bandında en basit). ~**%3.6 göreli maliyet.** 1-SE
kuralı (Breiman) bilinçli olarak biraz performansı parsimoniye takas eder — ama bunun **sınırlı**
olması gerekir.

**Karar:** 1-SE basitlik ikamesi yalnız **göreli OOF maliyeti ≤ `_SIMPLICITY_MAX_COST` (0.03)**
iken uygulanır. `within` bandından, en iyiden `> %3` uzak adaylar basitlik-ikamesine **uygun
değil** (yine bantta sayılır — bilgi amaçlı — ama şampiyon olamaz). En iyi her zaman uygun.

m5 smooth etkisi: auto_theta (53.57, +%3.6) uygun değil → `elig` = {chronos, joint(+0.5%),
classical_ensemble(+1.3%)} → en basiti seçilir. `_FAMILY_COMPLEXITY` += `foundation_ts`/`neural_ts`
(zero-shot/pretrained → operasyonel basit, 2/4).

## Sözleşme

- `scoring/__init__.score_reports` — `_sparse_fraction(profile)` → `select_champion(sparse_frac=)`.
- `scoring/selection.select_champion(..., sparse_frac: float = 0.0)`.
- `scoring/selection._evaluate_promotion(champ, report, config, sparse_frac)` — sparse ise pct tavanı atla.
- `scoring/selection.select_champion` ONE_STD_ERR — `_SIMPLICITY_MAX_COST = 0.03` göreli tavan.
- `scoring/selection._FAMILY_COMPLEXITY` — `foundation_ts: 2`, `foundation: 2`, `neural_ts: 4` (additive).
- `RunConfig` / `ScoreRow` / `PromotionResult` **değişmez** (reason listesi zenginleşir).

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `scoring/__init__._sparse_fraction(profile)` = `(intermittent + lumpy) / toplam` (TS'de);
  `score_reports` → `select_champion(sparse_frac=)`.
- `_evaluate_promotion(champ, report, config, sparse_frac)` — `sparse_frac ≥
  dynamics.segment_sparse_min_frac` iken yüzde-metrik tavanı hiç uygulanmaz.
- `select_champion` ONE_STD_ERR — `_SIMPLICITY_MAX_COST = 0.03`: `within` bandından yalnız
  `|r.mean − best.mean| ≤ 3%·|best.mean|` adaylar ( + best) basitlik-ikamesine uygun; `champ` bunlardan
  en basiti. `within_1se` yine tüm bant (bilgi amaçlı).
- `_FAMILY_COMPLEXITY` += `neural_ts: 4`, `foundation: 2`, `foundation_ts: 2`.
- Testler: `test_simplicity_substitution_capped_by_relative_cost`,
  `test_promotion_skips_pct_ceiling_on_intermittent_panel`; `test_one_se_rule_picks_simplest`
  güncellendi (ridge 10.8→10.2, %3 içine).

## Kapsam dışı

- Holdout-tabanlı band-içi yeniden sıralama → gerekirse sonra.
- `_SIMPLICITY_MAX_COST` config alanı → şimdilik sabit; benchmark ihtiyaç gösterirse config'e.
- Kesikli talebe özel promotion eşikleri (per-class) → v1.1+.
