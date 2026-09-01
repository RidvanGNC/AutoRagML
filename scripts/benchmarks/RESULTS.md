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

## Sonraki dalgalar

- **2. dalga:** TS panel (Nixtla `car_parts`, tourism) + M5 (Kaggle, elle).
- **3. dalga:** yüksek boyut/seyrek, ordinal, quantile; `--hpo light/thorough` karşılaştırması.
