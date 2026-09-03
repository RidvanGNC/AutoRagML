# ADR 0038 — seçim robustluğu (kararsız-CV filtresi) + heterojen panel holdout

**Durum:** Kabul · 2026-09-03 (benchmark v1.1 1. dalga bulgusu — kullanıcı onayı)

Kaynak: v1.1 tam-blok forecasting benchmark'ı (m3 + tourism).

## Bulgu

| dataset | şampiyon | test | naive | Δ% | durum |
|---|---|---|---|---|---|
| tourism_large | auto_ets | wMAPE 16.5 | 19.6 | **+15.9%** | ✅ |
| m3_monthly | **lightgbm** | sMAPE **44.7** | 14.8 | **−202%** | ❌ |

m3 leaderboard'da SE deseni:

| model | sMAPE | **OOF SE** (fold'lar arası) |
|---|---|---|
| classical_ensemble / auto_ets / auto_arima | 11.9–13.0 | **0.03–0.15** (kaya sabit) |
| lightgbm / random_forest / weighted_ensemble | 12.2–13.0 | **1.8–2.4** (savruk) |

**Reduction/ML modellerinin OOF'u fold'lar arası çok kararsız** (bir pencerede iyi, ötekinde
berbat). `lightgbm` "12.25" ± **1.76** — M3'ün kısa serilerinde trend ekstrapolasyonu →
sistematik over-predict → gerçek gelecekte sMAPE 44.

**1-SE bandı bunu yakalamıyor:** band = en iyinin ~0.59 SE'si → `[11.70, 12.29]`. `lightgbm`
(12.25) içeri, `auto_ets` (12.73) dışarı. Tie-break `(karmaşıklık, süre)` → lightgbm (gbdt=3) <
ensembleler (5) → **lightgbm şampiyon.** Oysa 12.25 ± 1.76'lık tahmin, 11.70 ± 0.035 ile
"istatistiksel eşdeğer" sayılamaz — 1-SE karşılaştırması apples-to-oranges.

**İkinci sorun:** m3 heterojen panel (seriler farklı zamanlarda bitiyor) → ADR 0020 global-cutoff
holdout yalnız geç-biten serileri yakaladı → **5652 gerçek holdout satırının 249'u** → herhangi
bir holdout-tabanlı güvenlik ağı işe yaramaz.

## Karar

### 1. Kararsız-CV filtresi (`scoring/selection._within_one_se`)

1-SE bandına yalnız **güvenilir** adaylar girer: `r.oof_metric_se > _SE_BAND_MULT · band` olan
aday (band = `max(best.se, noise_floor)`) banttan **dışlanır** (`best` hariç). `_SE_BAND_MULT = 2.0`:
"CV tahminin bandın 2×'inden geniş belirsizlikteyse, en iyiyle güvenilir biçimde eşdeğer değilsin."

- Sağlam istatistik, forecasting'e özel değil, benchmark'a ayar değil.
- Yalnız **band üyeliğini** etkiler — bir model bariz en iyiyse (`best`, bandın dışında) yine kazanır
  (M5 winner GBM örneği korunur).
- `noise_floor ≤ 0` (SE yok, <2 fold) → filtre atlanır (mevcut davranış).

m3 etkisi: lightgbm (SE 1.76 > 2·0.59=1.18) · random_forest · weighted_ensemble (SE ~2.2) → dışlanır.
Bantta kalan: `joint_ensemble` (0.035) + `classical_ensemble` (0.026) → şampiyon güvenilir klasik-ağırlıklı.

### 2. Heterojen panel holdout (`interfaces/holdout._ts_holdout`)

`group_col` varsa: global cutoff yerine **her serinin kendi son `k` dönemi** holdout
(`k = min(horizon, ...)`). Kısa seri koruması: `series_len > 2·k` olmayan seriler holdout'a girmez.
Leakage-safe: reduction `shift(h)` per-seri, `k ≤ h` → holdout satırlarının lag'leri train'e uzanır;
havuz model hiçbir seri-holdout `y`'sini görmez (mask train'den çıkarır). Tek seri → mevcut
global-cutoff mantığı.

## Sözleşme

- `scoring/selection._within_one_se` — SE-band filtresi (`_SE_BAND_MULT = 2.0`). `ScoreRow`/imza değişmez.
- `interfaces/holdout._ts_holdout` — panelde per-seri son-k. `HoldoutSplit` sözleşmesi değişmez.
- `RunConfig` değişmez.

## Kapsam dışı / sonra

- Holdout-tabanlı band-içi yeniden sıralama (top-K adayı holdout'ta skorla) → ihtiyaç görülürse v1.1+.
- Fold-eğilimli bozulma tespiti (son fold >> önceki fold'lar → karantina) → v1.1+.
- Forecasting'e özel `_FAMILY_COMPLEXITY` → gerekmedi (SE filtresi yeterli).

## Sonuç (UYGULANDI — commit 6dd7ff4, 2026-09-03)

- `scoring/selection._within_one_se` — `_SE_BAND_MULT = 2.0`; `r.oof_metric_se > 2·band` → banttan
  dışlanır (`best` hariç; `noise_floor ≤ 0` → filtre atlanır).
- `interfaces/holdout._ts_holdout` — `group_col` varsa `cumcount(ascending=False) < k` per-seri;
  `series_len > 2·k` koruması. Tek seri → global cutoff.
- Testler: `test_se_band_filter_excludes_unstable_candidate`,
  `test_ts_holdout_heterogeneous_panel_per_series`.

### m3 re-run (v4) — düzeltme doğrulandı

| | v3 (bozuk) | v4 (ADR 0038) |
|---|---|---|
| şampiyon | lightgbm (gbdt) | **patchtst** (neural_ts) |
| test sMAPE | 44.7 (**−202%** vs naive) | **16.26 (−9.9%)** |
| holdout bias | +3462 | −180 |
| internal holdout | 249 satır (%4) | **5652 satır** (tam); sMAPE 16.65 ≈ test 16.26 |

SE-band filtresi kararsız reduction'ı (lightgbm SE 1.76) eledi → şampiyon düzgün ekstrapole
eden `patchtst`. Per-seri holdout → internal holdout testi **doğru tahmin ediyor** (16.65 vs 16.26).
"no_improvement" = M3 monthly seasonal-naive'i (14.8) geçemedi — tarihsel `auto_ets` (−7.9%)
seviyesinde; **regresyon değil**.

### Yan bulgu (ayrı commit 24af8c3)

Regresyon suite: `RidgeClassifier.predict_proba` yok → ADR 0036 sınıflandırma bagging çöküyordu.
`FittedModelPipeline.supports_proba`; sınıflandırmada `PredictKind.PROBA` yoksa bagging kapalı.
