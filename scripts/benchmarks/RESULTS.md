# Benchmark sonuçları

Ham koşum çıktıları `_runs/` altında (git-ignored). Bu dosya elle özetlenir.

## 1. dalga — OpenML/sklearn tablo (2026-09-01, `--hpo none`)

Ölçü: şampiyon, **fit'in hiç görmediği %20 harici test setinde** naive baseline'ı
(ortalama / çoğunluk sınıfı) geçer mi.

| dataset | görev | şampiyon | metrik | test | naive | Δ% | holdout | süre |
|---|---|---|---|---|---|---|---|---|
| california_housing | regresyon | lightgbm | rmse | 0.446 | 1.145 | **+61%** | 0.458 | 47s |
| bike_sharing | regresyon | extra_trees | rmse | 42.9 | 178.0 | **+76%** | 41.8 | 115s |
| adult | ikili | lightgbm | f1_macro | 0.819 | 0.432 | **+90%** | 0.811 | 58s |
| credit_g | ikili | logistic | f1_macro | 0.663 | 0.412 | **+61%** | 0.714 | 11s |
| bank_marketing | ikili | lightgbm | f1_macro | 0.743 | 0.469 | **+59%** | 0.764 | 53s |
| covtype | çok-sınıf | extra_trees | f1_macro | 0.799 | 0.093 | **+755%** | 0.797 | 232s |

**6/6 SUCCESS.** Klasik + GBM/forest roster tüm tablo görevlerinde naive'i geniş farkla
geçiyor; OOF ↔ holdout tutarlı (aşırı-uyum yok). `weighted_ensemble` bu setlerde şampiyon
olmadı (tek GBM/forest zaten 1-SE bandında en iyi + en basit — beklenen davranış).

## 1b. dalga — `--hpo light` (2026-09-01)

**6/6 SUCCESS ama 4/6'da `none`'dan KÖTÜ**, 10–15× maliyetle:
credit_g +2.5% · covtype +1.9% iyi; california/bike/adult/bank ~0.5–1% kötü.
Ensemble california'da şampiyon oldu ama harici test'te genelleşmedi.
→ **ADR 0022 (k-fold bagging + `light` inner_folds=2) tetiklendi.**

## 1c. dalga — `--hpo none` + **k-fold bagging** (ADR 0022, 2026-09-02)

| dataset | metrik | 1a test | 1c test | 1a holdout | 1c holdout | not |
|---|---|---|---|---|---|---|
| california_housing | rmse↓ | 0.4459 | 0.4459 | 0.4575 | **0.4544** | bagged; test ~aynı (LightGBM düşük varyans), **holdout ↓** |
| bike_sharing | rmse↓ | 42.90 | **42.37** | 41.78 | **41.54** | bagged; **test −1.2%** (gürültülü count regresyonu — varyans azalması) |
| adult | f1↑ | 0.8194 | 0.8186 | 0.8115 | 0.8123 | sınıflandırma → bag YOK; ~aynı (run varyansı) |
| credit_g / bank / covtype | — | değişmedi | değişmedi | | | sınıflandırma → bag YOK (v1) |

**Değerlendirme:** bagging gürültülü regresyonda net kazanç (bike_sharing −1.2%),
california'da holdout iyileşmesi, gerisinde nötr — **regresyon yok**. Süre ~aynı
(`hpo=none`'da refit küçük pay). Sınıflandırma bagging (proba ortalaması) → v1.1.

## Bulunan buglar (bu dalga)

1. **`ColumnDropper.fit` yerel closure** → `save_bundle` covtype gibi kolon-düşüren
   pipeline'larda `PicklingError`. **Düzeltildi**: tüm stateless op'lar modül-düzeyi
   `__slots__` callable sınıfları (drop/date_expand/log1p/hashing).
2. `fetch_openml(version=None, data_id=...)` çakışması — harness düzeltildi.
3. `naive_prediction` majority `dtype=object` döndürüyordu — harness düzeltildi.

## Bilinen v1 sınırı

String-etiketli sınıflandırma hedefi `split_xy`'de sayısala zorlanıyor → benchmark
hedefi `pd.Categorical(...).codes` ile kodluyor (`target_encoded=true`). Otomatik
label-encoding → **v1.1 ADR**.

## 2. dalga — panel forecasting (Nixtla `datasetsforecast`)

Her seri **son `horizon` dönem** harici holdout; **seasonal-naive** baseline; sMAPE.

### 2a — reduction-only (ADR 0023 öncesi) — BAŞARISIZ
`m3_monthly` şampiyon lightgbm, sMAPE **47** vs seasonal-naive **14** (nihai holdout sMAPE **63**).
Kök neden: klasik modeller (auto_arima/ets/theta/croston) kataloğda ama reduction
pipeline'ından geçemiyor (`'DataFrame' object has no attribute 'dtype'`).

### 2b — native classical (ADR 0023) + EAT ansambl & Tweedie (ADR 0024), `--hpo none`

| dataset | primary | şampiyon | test | seasonal-naive | Δ | nihai holdout |
|---|---|---|---|---|---|---|
| m3_monthly (400 seri) | sMAPE | `auto_ets` (statistical) | 15.97 | 14.80 | −7.9% | 17.4 (ADR 0023 öncesi **63**) |
| tourism_large (555 seri) | wMAPE | `extra_trees` (forest) | **17.61** | 19.64 | **+10.3% ✅** | 18.12 |
| m5_subset (400 seri, kesikli) | wMAPE | `tsb` (intermittent) | 105.2 | 99.6 | −5.6% 🔴 | 75.9 |

**ADR 0025 (zengin reduction, ~13→42 özellik) etkisi:** tourism +5.9% → **+10.3%** (reduction'ın
zaten şampiyon olduğu panel); m3 değişmedi (klasik model M3 monthly'de reduction'ı geçiyor).
m5: recursive multi-step + per-grup olmadan rich features tek başına yetmiyor (ADR 0026).

- **m3:** ADR 0023 forecasting'i çalışır yaptı (reduction-only sMAPE 47 → auto_ets 16, holdout **63→17**).
  Seasonal-naive M3 monthly'de competition-grade → auto_ets %8 içinde, iyi sonuç. `classical_ensemble`
  (EAT, 6 üye) **kuruldu ve yarıştı** ama 1-SE kuralı simpler `auto_ets`'i tercih etti (M3 homojen,
  EAT kazancı < 1SE — **doğru davranış**; EAT heterojen M4-tarzı panelde parlar).
- **tourism:** ✅ SUCCESS (wMAPE +5.9%). İlk koşum sMAPE'de fail'di (y≈0 → sMAPE patlar) →
  `primary_metric=wmape` düzeltmesi.
- **m5 (kesikli talep): 🔴 rekabetçi DEĞİL.** wMAPE 105 (>100 = hata > toplam talep). Tweedie ipucu
  **doğru tetiklendi** ("panelin %93'ü düzensiz") ama tek başına yetmez. M5 winner'ın stack'i:
  hiyerarşik LightGBM (per store-dept, pooled değil) + Tweedie + zengin özellikler (rolling/price/
  calendar) + **recursive multi-step**. Bizde: tek pooled model + direct h-step + temel lag.
  **Sonuç: M5-rekabetçi intermittent forecasting Gap #2+#4+#5'i BİRLİKTE gerektirir** (ADR 0025+).
- **Bug (bloklamıyor):** promotion kapısı `smape_max=35` sabit → wMAPE koşumda yanlış `FAIL`. v1.1.
- `weighted_ensemble` (reduction) ve `classical_ensemble` ayrı yarışır; ortak GES v1.1.

### 2c — recursive multi-step (ADR 0026 B) + guardrail nonneg-clip (ADR 0027), `--hpo none`

| dataset | koşum | şampiyon | OOF wMAPE | nihai holdout wMAPE |
|---|---|---|---|---|
| m5_subset | direct (2b) | `tsb` | 80.4 | 75.9 |
| m5_subset | `--forecast-reduction recursive` | `tsb` | 80.4 | 75.9 |
| m5_subset | recursive **+ ADR 0027** | **`auto_ets`** | **77.8** | **74.9** |

- **Recursive tek başına M5'i hareket ettirmedi** — şampiyon yine `tsb` (reduction ailesi rekabetçi
  değil, strateji fark etmiyor). Recursive GBDT OOF ~83 (direct'e yakın); forest modeller recursive'de
  hata birikmesiyle patlıyor (wMAPE 108-127). Recursive-h CV altyapısı doğru çalışıyor.
- **ADR 0027 gerçek kazanç:** leaderboard'da en iyi tahminciler (`classical_ensemble` 77.2, `auto_ets`
  77.8, `auto_theta` 78.8) ~%2 küçük negatif tahmin (ETS/Theta additive doğası, kesikli talepte y≈0)
  yüzünden `prediction_negative` ile karantinaya alınıyordu; `auto_nonneg` kırpma bunları serving'de
  garanti 0'a çekiyor. Guardrail artık kırpmayı biliyor → şampiyon `tsb` (80.4) → `auto_ets` (77.8),
  holdout 75.9 → 74.9.
- **Klasik serving bug → ADR 0029 ile düzeltildi:** forecasting `champion_test_score` (yukarıda
  "test 105.2") klasik şampiyon için sabit/yanlıştı — `FittedClassicalForecaster.predict` yalnız
  fit-sonrası ilk `h` adımı üretiyordu; şampiyon `train−holdout`'ta fit edilince gerçek gelecek
  pencerenin ötesinde → tüm satırlar "son değer" fallback (3 koşumda byte-identik 105.2). Artık
  `predict` istenen `ds` aralığını kapsayacak kadar ileri tahmin eder. **Nihai holdout
  (orchestrator) zaten doğruydu** (OOF + holdout raporlandı).

## Sonraki dalgalar
- **ADR 0028+0029 birlikte:** m5 `--only m5_subset` (SBC segmentasyonu + düzeltilmiş klasik serving) — pending.
- **3. dalga:** yüksek boyut/seyrek, ordinal, quantile; `--hpo light/thorough` karşılaştırması.
- **v1.1:** orchestrator `champion_refit_full` (feature modellere güncellik); klasik+reduction ortak
  ensemble; segment-arası ensemble.
