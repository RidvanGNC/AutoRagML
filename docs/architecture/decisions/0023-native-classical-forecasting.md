# ADR 0023 — native classical forecasting (StatsForecast entegrasyonu)

**Durum:** Kabul · 2026-09-02

Kaynak: benchmark 2. dalga — `m3_monthly` şampiyon sMAPE **47** vs seasonal-naive **14**
(naive'den 3× kötü). Kök neden: `auto_arima`/`auto_ets`/`auto_theta`/`croston`/`tsb`
kataloğda ama **reduction (lag→tabular) pipeline'ından** geçmeye zorlanıyor →
`estimator.fit(DataFrame, Series)` → `'DataFrame' object has no attribute 'dtype'`.
Reduction-only + tek pooled model çok-serili panelde zayıf (AutoGluon-TS dersi:
klasik + reduction + ensemble birlikte).

## Karar

Klasik modeller (`family ∈ {statistical, intermittent}`) **ayrı çalıştırma yolundan**
geçer — Nixtla `StatsForecast` native API'si (panel için tasarlı):

### 1. Aday ayrımı (`TimeSeriesCoreEngine`)
`resolve_candidates` → `family` ile böl:
- **reduction adayları** (ml/gbdt/linear/forest/neural) → mevcut `run_validation_suite` (shift≥h lag).
- **klasik adaylar** → `engines/timeseries/classical.run_classical_reports`.

Raporlar birleşir → `build_weighted_ensemble` → `score_reports` → şampiyon **her iki
aileden** olabilir.

### 2. OOF: `StatsForecast.cross_validation`
`sf.cross_validation(df[unique_id,ds,y], h=horizon, n_windows=n_folds, step_size=horizon)`
→ her model + her cutoff için tahmin. Cutoff'lar = dış fold'lar. Concatenated cutoff
tahminleri = OOF (rolling-origin, leakage-safe — StatsForecast garanti eder).
Her klasik aday → sentetik `ValidationReport` (folds = cutoff'lar, `oof`, `oof_metrics`, `oof_metric_se`).

### 3. Refit + serving: `FittedClassicalForecaster`
Şampiyon klasik ise: `sf.fit(full_train_df)`. `FittedClassicalForecaster` `Predictor`
protokolünü karşılar:
- `predict(frame)` — `frame` (uid,ds sıralı) her seri için son-`h` satır = forecast ufku.
  `sf.forecast(df=history, h=h)` → `(uid,ds)` ile hizala; ufuk-dışı satırlar seri son
  train değeriyle doldurulur (skorlanmaz, NaN önler).
- pickle: fitted `StatsForecast` nesnesi joblib ile serialize edilir.

### 4. Katalog düzeltmeleri
- `season_length` mevsimsel modellere (`AutoETS`, `MSTL`, `AutoTheta`) enjekte edilir —
  `profile.timeseries.seasonality` veya freq→periyot (`_FREQ_SEASON`).
- `MSTL` → `season_length` zorunlu (yoksa aday atlanır, artık enjekte edilir).
- `freq` — `profile.timeseries.freq` (pandas offset alias) doğrudan `StatsForecast(freq=...)`.

## Sözleşme
Yeni sözleşme yok. `Candidate.family` yeterli ayraç. `BundleMetadata.params` klasik
modelde `{"season_length": s, "freq": f}`.

## Kapsam dışı / v1.1
- **GES ensemble klasik modelleri dışlar** — cutoff-tabanlı OOF ≠ fold-tabanlı reduction
  OOF (hizalama). Klasik+reduction ensemble → v1.1 (ortak backtest ızgarası).
- Klasik model **HPO yok** (zaten "Auto"). **Bagging yok** (CV zaten OOF verir).
- `per_group_champion` gerçek per-grup — klasik yol zaten per-series; reduction hâlâ pooled (v1.1).
- Neural forecasting (neuralforecast) → v1.1 ayrı ADR.

## Sonuç
- `engines/timeseries/classical.py` — `run_classical_reports` + `FittedClassicalForecaster` + `refit_classical`.
- `TimeSeriesCoreEngine` iki yolu çalıştırıp raporları birleştirir.
- `refit_champion` klasik şampiyonu `refit_classical`'a yönlendirir.
- benchmark ile ölçülür (`m3_monthly` sMAPE < seasonal-naive hedef).
