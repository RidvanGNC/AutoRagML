# ADR 0010 — analyzers sözleşmesi + veri hazırlama metodolojisi

**Durum:** Kabul · 2026-09-01
**Motto:** Zamanın miktarı önemsiz, sağlıklı başarı kesin ölçüdür. Detay kaçmaz.

Güncel kaynaklarla doğrulandı — bkz. bu ADR sonundaki referanslar.

## Metodoloji — "her şeyi dönüştür sonra skew ile ele" REDDEDİLDİ

Eski yaklaşım (tüm kolonları uniform dönüştür → skew/varyans testiyle kolon at)
sızıntı üretir, sinyal atar, sırası terstir. Doğru akış:

| Adım | Katman | Kural |
|---|---|---|
| Betimle (skew, kardinalite, eksik, dtype, ADI/CV²) | `analyzers` | **karar yok, fit yok** |
| Karar ver (kolon → op) | `dynamics/planner` | yapısal eleme dönüşümden **önce**; skew → `log1p` (drop değil); yüksek kardinalite → encode (drop değil) |
| Fit et | `validators` | **yalnız fold içinde** (bkz. ADR 0011) |
| Feature selection (opsiyonel) | modelleme-zamanı | model-tabanlı + iç-fold konsensüs (cnCV); ön işleme filtresi değil |

Dönüşüm kararları **model-ailesi-farkında** (ağaç/GBM için minimal, lineer için kapsamlı).

### committed_ops vs candidate_ops
`AdaptivePlan` iki tür op taşır:
- **`committed_ops`** — yapısal, her zaman uygulanır (constant/duplicate/all-null drop,
  datetime açılımı, kategorik encode)
- **`candidate_ops`** — CV ile denenip seçilir (`log1p` / `yeo_johnson` / `quantile` /
  yok; hedef dönüşümü `TransformedTargetRegressor` sarımıyla). Skew ölçümü yalnız
  **hangi adayların önerileceğini** belirler, kararı değil. ("a priori ne işe yarar
  bilinemez; yalnız CV iyileşiyorsa uygula.")

## analyzers iç çalışma sırası (= başlama sırası)

1. `modality.detect` → `tabular | timeseries`
2. `profiling.build` → her kolon için `ColumnProfile` (ağır geçiş) ← **ilk gerçek iş**
3. `task_inference.infer` → `TaskSpec`
4. `timeseries.diagnose` → `TimeSeriesProfile` (yalnız forecasting / `time_col` varsa)
5. `quality.scan` + `leakage.scan` (yumuşak, bkz. ADR 0011)
6. `DataProfile` + `TaskSpec` birleştir

## ColumnProfile — AutoGluon FeatureMetadata deseniyle hizalı

Tek `role` yerine:
- `raw_dtype` — pandas primitifi (`int/float/category/object/datetime/bool`)
- `special_types: set[str]` — AutoGluon sözlüğü (`text`, `text_ngram`, `datetime`,
  `embedded_number`, `boolean`, ...)
- `semantic_role` — türetilmiş (`target/id/categorical/numeric_continuous/
  numeric_discrete/boolean/datetime/text/constant/unknown`)
- `flags: set[str]` — `high_cardinality · near_constant · high_missing · all_missing ·
  skewed · heavy_tailed · zero_inflated · datetime_like_string · numeric_like_string ·
  duplicate_of:<col> · monotonic · leakage_suspect`
- stats: `n_unique, missing_ratio, min/max/mean/std/skew` (numeric), `top_values`, `sample_values`
- `confidence` — düşük güvende WARNING (akış durmaz; `target` yine zorunlu — ADR 0008)

v1 kural-tabanlı çıkarım. ML-tabanlı detektör (SortingHat/Sherlock sınıfı) = opsiyonel
eklenti, gelecek. Kullanıcı `feature_metadata` override her zaman açık.

## task çıkarımı

Düz enum (v1 hepsi): `regression · binary_classification · multiclass_classification ·
multilabel_classification · quantile_regression · ordinal_regression · forecasting`.

Kurallar hedef kolona bakar (`RunConfig.task_hint` yoksa): `time_col` var + numeric →
`forecasting` (`task_hint: regression` ile ezilir, **uyarı**); numeric n_unique==2 →
binary; numeric n_unique ≤ `max_classes` (default 20) → belirsiz, default multiclass +
**uyarı**; numeric n_unique büyük → regression; kategorik n_unique ≤ eşik → multiclass;
çoklu hedef → multilabel; `RunConfig.quantiles` → quantile_regression.

Eşikler `config.analyzers.thresholds`: `max_classes_for_classification=20`,
`high_cardinality_ratio=0.5`, `high_cardinality_abs=1000`, `skew_abs=1.0`,
`near_constant_freq=0.99`, `high_missing_ratio=0.4`, `id_uniqueness_ratio=0.98`,
`text_min_avg_tokens=3`.

## TimeSeriesProfile — ölçüm; intermittency = İPUCU (router değil)

`analyzers/timeseries.py`:
- `freq` (`pandas.infer_freq` sıralı per-series; düzensizse modal-gap) + `freq_confidence`
- `span`, `regular`, `gaps[]` (grup bazında)
- `seasonality[]` — freq→periyot sözlüğü (`W→[52]`, `D→[7,365]`, `MS→[12]`, `H→[24,168]`)
  + ACF/STL ile **güç doğrulaması** (varsayma). Çoklu mevsimsellik (MSTL) mümkün.
- `trend_strength`, `stationarity` (ADF p)
- `per_series[]` — grup bazında: `n_obs, n_nonzero, zero_ratio, adi, cv2, history_weeks`,
  `intermittency_class` (SBC), `intermittency_class_recent` (son N dönem — trend/obsolescence),
  `class_changed_over_time: bool`
- `classification_scheme: "sbc" | "kh"` (default `sbc` pratik; `kh` non-lineer sınır, opsiyonel)
- `per_series_detail: full | sampled | summary_only` (default `full`; 100k grup → `sampled`)

**Intermittency sınıfı model ailesini KISITLAMAZ:**
- zero-heavy seri → aday havuzuna Croston/TSB/SBA + Tweedie/Poisson **ekle**
- sınıf → **birincil metrik** etkiler (zero-heavy → ölçekli hata / bias / CSL; MAPE değil)
- **nihai seçim yine holdout** (`scoring` + guardrail döngüsü) — dogma değil ampirik
- ADR 0004'teki DemandSensing `routing.py` deseni buna göre yumuşatıldı

## Leakage taraması (yumuşak) → ADR 0011

## Referanslar
- AutoGluon FeatureMetadata (raw + special dtypes)
- SortingHat SIGMOD 2021 — feature type inference benchmark (9 sınıf; ML +%14–38)
- Open Forecast (2024) + Kostenko & Hyndman — SBC sınıflandırma eleştirisi
- Nixtla tsfeatures — infer_freq + seasonal period sözlüğü
- scikit-learn TransformedTargetRegressor; quantile/power transform "a priori bilinemez"
- Consensus Features Nested CV (cnCV)
