# ADR 0044 — split-conformal tahmin aralıkları (`predict_interval`)

**Durum:** Kabul · 2026-09-04 (v1.1 borcu: ADR 0017'de `ConformalConfig` rezerve edilmişti)

Kaynak: [MAPIE](https://mapie.readthedocs.io/) (scikit-learn-contrib, split-conformal + CQR standart
hale gelmiş) araştırması — bizim OOF mimarimiz (ADR 0011, leakage-safe) zaten split-conformal'ın
gerektirdiği "ayrılmış kalibrasyon seti"nin ta kendisi; ek fit/split gerekmiyor.

## Karar

### Yöntem: split-conformal (mutlak residual), CQR değil
`ConformalConfig` (ADR 0017, zaten kilitli: `enabled`/`coverage`/`per_group`) docstring'i "split-
conformal" diyordu — sözleşmeye sadık kalındı. Kalibrasyon = **mevcut şampiyon OOF'u**
(`calibrate` adımıyla aynı kaynak). Genişlik, finite-sample düzeltmeli mutlak-residual kantili:
`q = ceil((n+1)·coverage) / n` order statistic (MAPIE/standart split-conformal formülü).

`coverage` fit-zamanında SABİTLENMEZ — `ConformalFit` ham sıralı `|residual|` dizisini saklar
(global + yeterli örneklemli gruplar); `interval()`/`predict_interval()` her çağrıda istenen
`coverage` için kantili anında hesaplar (varsayılan: config'teki `coverage`).

### Zincirdeki yeri
`calibrate → clip → **conformal genişliği** → round`. Aralık merkezi = kalibre+clip-uygulanmış
nokta (`FittedPostprocessor._point`, `apply()` ile paylaşılan ortak adım); lower/upper ayrı ayrı
clip + round edilir (served ölçekle tutarlı, negatif alt sınır sızmaz).

### `per_group=True`
Grup (ör. seri `unique_id`) başına residual kantili. Örneklemi **< 10** (sabit, `min_group_oof`)
olan gruplar global kantile düşer — küçük örneklemde anlamsız dar/geniş aralık üretmez. `group=None`
her zaman global'i kullanır (aynı kod yolu → tutarlılık testi).

### Kapsam: yalnız `FittedModelPipeline` / `FittedEnsemblePipeline`
Bu iki sınıf **tablo regresyon + reduction-forecasting'i (+ bagged/GES ikisi de)** kapsıyor — aynı
runtime sınıfları reduction modelleri de taşıdığı için `per_group` forecasting panellerinde
(`task.group_col`) de doğrudan çalışıyor. **Kapsam dışı (ADR 0044-B takip):** native forecaster'lar
(`FittedClassicalForecaster`/`FittedNeuralForecaster`/`FittedFoundationForecaster`), `FittedJointForecaster`,
`FittedStackPipeline`, `FittedSegmentedPipeline` — bu sınıflarda `predict_interval` YOK; postprocessor
yine de conformal'ı fit ediyor (ileriye dönük, ucuz) ama pipeline onu sunmuyor. Sınıflandırma da
kapsam dışı (conformal-set farklı çıktı şekli — `[lo,hi]` değil).

### API
`predict()` değişmedi (nokta tahmin). Yeni `predict_interval(data, coverage=None) -> (lower, upper)`
— iki numpy dizisi (explain()/predict() ile tutarlı sade çıktı):
- `RunResult.predict_interval` / `AutoRagML.predict_interval` / `LoadedChampion.predict_interval`
- Şampiyon türü desteklemiyorsa (`hasattr` yok) → açık `NotImplementedError` (sessizce yanlış/point
  döndürmez); conformal fit edilmemişse (`enabled=False` veya yetersiz OOF) → `(point, point)`.

## Uygulama

- `postprocessors/conformal.py` (yeni): `fit_conformal` (saf fonksiyon, OOF residual → `ConformalFit`),
  `ConformalFit.width_for` (skaler veya satır-başı dizi). `frozen, slots` dataclass — yalnız numpy
  dizileri → joblib-picklable, sidecar/ana bundle'da sorunsuz.
- `postprocessors/pipeline.py`: `FittedPostprocessor._point` (bias/clip, round'dan önce — `apply`
  ile paylaşılır) + `.interval()` + `.has_conformal`. `Postprocessor.fit(y_true, y_pred, group=)`.
  `is_active`/`is_noop` conformal'ı da sayar.
- `contracts/postprocess_config.py`: v1 guard'ından `conformal.enabled` reddi kaldırıldı.
- `engines/champion.py`: `_maybe_postproc(..., group=)` — 5 çağrı yerinde `oof.group` / bagging
  döngüsünde biriktirilen grup dizisi geçiliyor (calibrate zaten aynı OOF'u kullanıyordu, conformal
  onu genişletiyor). `FittedModelPipeline`/`FittedEnsemblePipeline` inşasına `group_col=task.group_col`.
- `engines/model_pipeline.py` + `engines/ensemble_pipeline.py`: `_raw_point`/`_blend_point` ortak
  yardımcı (postprocessor'dan ÖNCEki nokta + grup dizisi — `predict`/`predict_interval` paylaşır).
- `contracts/run_result.py` + `interfaces/api.py`: `predict_interval` — `explain()` deseniyle aynı
  (data zorunlu, `getattr(pipeline, "predict_interval", None)` capability-check).

## Doğrulama

- `tests/unit/postprocessors/test_postprocess.py`: 8 yeni test (disabled→point, global aralık kapsama,
  per_group farklı genişlik, küçük-grup fallback, çağrı-zamanı coverage override, yetersiz-OOF sessiz
  atlama, yalnız-conformal is_noop=False).
- `tests/unit/interfaces/test_orchestrator_api_cli.py`: uçtan uca `AutoRagML().predict_interval()`
  (kapsama + coverage override), varsayılan kapalı → point==point, desteklenmeyen şampiyon türünde
  `NotImplementedError`.
- Manuel e2e smoke: sentetik regresyon, champion=linear, `predict_interval` nokta tahmini kapsayan
  makul genişlikte `(lo,hi)` üretti.
- `tests/unit/engines/` + `tests/unit/postprocessors/` tam koşum: 64 passed, 1 skipped (regresyon yok).
- ruff + mypy(146 dosya) yeşil.

## Kapsam dışı (ADR 0044-B — takip)

- Native forecaster / joint / stack / segmented şampiyonlarda `predict_interval`.
- Sınıflandırma conformal-set (farklı çıktı şekli).
- Forecasting ufuk-adımına göre stratifikasyon (h=1 hatası ≠ h=18 — `OOFArrays`'e yeni alan gerekir).
- CQR (conformalized quantile regression) — quantile-yetenekli şampiyonlarda daha dar/adaptif aralık;
  rezerve sözleşme yalnız split-conformal diyordu, CQR ayrı karar gerektirir.
