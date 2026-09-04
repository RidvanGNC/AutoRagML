# ADR 0043 — TiDE nöral forecasting modeli (ADR 0032 genişletmesi)

**Durum:** Kabul · 2026-09-04 (araştırma tetiklemeli, benchmark sonrası planlama turu)

Kaynak: [Local vs Global Models for Intermittent Time Series Forecasting](https://arxiv.org/abs/2601.14031)
(2026-01) — 5 veri seti / 40K+ gerçek panel üzerinde karşılaştırma: **TiDE** (basit MLP encoder/
decoder) global kesikli-panel forecasting'de en isabetli, aynı zamanda büyük nöral mimarilerden
belirgin ucuz. ADR 0032'de TiDE zaten değerlendirilmiş ama 6-model kilitli kapsama alınmamıştı
("`neuralforecast`: NHITS · NBEATSx · PatchTST · TFT · iTransformer · TSMixer(x) · **TiDE** · DLinear · …").

## Karar

`models/catalog/neural_ts.yaml`'a `tide` eklendi — `neuralforecast.models.TiDE`, ADR 0032'nin
aynı deseni (native panel yolu, `run_neural_ts_reports`, `FittedNeuralForecaster`). Yeni kod yok:
`_MULTIVARIATE` kümesine girmiyor (iTransformer/TSMixer gibi `n_series` gerektirmiyor — NHITS/
PatchTST/TFT/NBEATSx ile aynı tek-seri arayüzü). `[neural-ts]` extra zaten `neuralforecast`
içeriyor → ek bağımlılık yok.

`default_params: {max_steps: 500, scaler_type: robust}` — diğer sabit modellerle tutarlı.
`neural_search=True` NAS'ına dahil edilmedi (ADR 0032'de yalnız NHITS/PatchTST/TFT için `Auto*`
kilitlendi — TiDE için `AutoTiDE` eklenmesi ayrı, düşük öncelikli karar).

## Doğrulama

- `test_neural_ts_catalog_yaml_present`: `tide` katalogda + `family == "neural_ts"`.
- Mevcut `run_neural_ts_reports`/`FittedNeuralForecaster`/bundle sidecar kod yolu değişmedi →
  regresyon riski yok (aynı sınıf, farklı model referansı).

## Kapsam dışı

- `AutoTiDE` (kütüphane NAS) — istenirse ayrı küçük ADR.
- Kovaryant (futr/hist/stat_exog) desteği — TiDE bunu native destekliyor ama `Dataset.relations`
  (ADR 0009 rezerve) açılmadan framework genelinde exogenous veri akışı yok; TiDE bu ADR'de yalnız
  hedef seri üzerinde (diğer 6 model gibi) kullanılıyor.
