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

### 2b — native classical (ADR 0023), `--hpo none`

| dataset | primary | şampiyon | test | seasonal-naive | Δ | nihai holdout |
|---|---|---|---|---|---|---|
| m3_monthly (400 seri) | sMAPE | **auto_ets** (statistical) | 15.97 | 14.80 | −7.9% | 17.4 (ADR 0023 öncesi **63**) |
| tourism_large (555 seri) | wMAPE | **extra_trees** (forest) | **18.48** | 19.64 | **+5.9% ✅** | 18.27 |
| m5_subset | wMAPE | _(koşuyor)_ | | | | |

- **m3:** M3 monthly'de seasonal-naive competition-grade (metodların çoğu ~13-15 sMAPE) — auto_ets
  %8 içinde, reduction-only'nin sMAPE 47/holdout 63'ünden **3.6× iyi**. Klasik yol devrede.
- **tourism:** **wMAPE ile SUCCESS.** İlk koşum sMAPE'de "başarısız"dı çünkü tourism serilerinde
  y≈0 → sMAPE patlıyor, croston yanlış şampiyon oluyordu. `primary_metric=wmape` (demand-planning
  standardı) → doğru seçim, seasonal-naive'i geçiyor.
- **Bug (küçük, bloklamıyor):** promotion kapısı `smape_max=35` sabit → wMAPE-optimize koşumda
  `promotion=FAIL` diyor ama seçim/başarı doğru. Metrik-duyarlı promotion → v1.1.
- `weighted_ensemble` klasiği dışlıyor (cutoff-tabanlı OOF ≠ fold-tabanlı, ADR 0023 v1).

## Sonraki dalgalar
- **3. dalga:** yüksek boyut/seyrek, ordinal, quantile; `--hpo light/thorough` karşılaştırması.
- **v1.1:** klasik+reduction ortak ensemble (ortak backtest ızgarası); recursive multi-step.
