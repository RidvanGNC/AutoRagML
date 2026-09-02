# ADR 0028 — segmented champion (per-group forecasting, v1)

**Durum:** Kabul · 2026-09-02

Kaynak: SOTA gap analizi Gap #5 + m5 benchmark. M5 winner (YJ In): store (10) / store-category
(30) / store-department (70) seviyelerinde **havuzlanmış** LightGBM modelleri, her seri 6 modelin
ortalaması. Tek pooled model M5'te wMAPE ~100'de takılıyor — hızlı/yavaş hareket eden ürünler
farklı dinamik ister. Gerçek per-series (30k model) uygulanamaz + aşırı uyar.

## Karar

`AdaptivePlan.structure == "per_group_champion"` artık **pooled'a düşmez** — TS engine serileri
**segment**'lere böler ve çekirdek pipeline'ı (adaylar → CV → seçim → refit) **segment başına**
koşar. Serving her seriyi grup kimliğiyle kendi segmentinin `ModelBundle`'ına yönlendirir.

### Segmentasyon (deterministik, `dynamics/planner`)

- **v1 birincil ölçüt: SBC intermittency sınıfı** (`profile.timeseries.per_series[].intermittency_class`
  — smooth / intermittent / erratic / lumpy). Zaten hesaplanıyor; M5 problemine doğrudan oturur.
- Küçük segmentler (`< dynamics.segment_min_series`, vars. 30) en yakın komşuya birleştirilir
  (ADI'ye göre sıralı: smooth↔intermittent↔lumpy↔erratic).
- Segment sayısı ≤ 4. Tek anlamlı segment kalırsa → **pooled** (fayda yok, mesaj).
- Segmentlenmemiş seriler (per-series profili yok → kısa) → en büyük segmente.

### Engine (`TimeSeriesCoreEngine`)

- Her segment: frame'i segment serilerine filtrele → `run_core_pipeline` (reduction + classical +
  ensemble + refit) → `ModelBundle`.
- `EngineResult.champion` = tek `ModelBundle`, `pipeline` alanı **`FittedSegmentedPipeline`**
  (Predictor protokolü): `{segment_adı: alt_pipeline}` + `group → segment` haritası + fallback
  (en büyük segment, bilinmeyen kimlik için).
- `metrics_oof` = segment boyutuna göre ağırlıklı ortalama OOF metrikleri.
- `scoreboard` = birleşik — her segmentin şampiyon satırı `segment::<ad>` prefiksiyle + özet.

### Serving `FittedSegmentedPipeline.predict(frame)`

Satırları `group → segment` ile böl → her alt-kümeyi kendi alt-pipeline'ına ver → orijinal
sırada birleştir. Bilinmeyen grup → fallback segment. joblib-picklable (alt-pipeline'lar zaten).

### Sözleşme

- `AdaptivePlan.segments: list[SegmentSpec]` (`SegmentSpec {name, group_ids, source}`) — additive,
  boş liste = pooled.
- `DynamicsConfig.segment_min_series: int = 30`, `segment_max_count: int = 4` (additive).
- `FittedSegmentedPipeline` (`engines/segmented.py`), `Predictor` protokolü.
- `EngineResult` / `ModelBundle` / `BundleMetadata` / `SelectionResult` **değişmez**: champion tek
  `ModelBundle`, `pipeline` alanı `FittedSegmentedPipeline`, segment→şampiyon haritası
  `metadata.adaptive_plan_summary["segments"]` içinde (serbest dict — yeni alan gerekmez).
- `run_core_pipeline`: `structure=="per_group_champion"` + `segments==[]` artık **degradasyon değil**
  (planlayıcı "tek anlamlı segment" kararı); çok-segment `TimeSeriesCoreEngine` işi.

## Kapsam dışı / sonra

- Çok-seviyeli hiyerarşik havuzlama (M5 store/dept/cat) → v1.1 (hiyerarşi kolonları gerekir).
- Segment-arası ensemble (M5 seviyeler ortalaması) → v1.1.
- Hacim/mevsimsellik ile kümeleme (k-means) → v1.1; v1 yalnız SBC sınıfı.
- Tabular (forecasting-dışı) segmentasyon — kapsam dışı, yalnız TS.

## Sonuç (implementasyon)

- `contracts/adaptive_plan.py` `SegmentSpec` + `segments`; `contracts/dynamics_config.py` eşikler.
- `dynamics/planner.py` `_resolve_segments(profile, task, cfg)`.
- `engines/segmented.py` `FittedSegmentedPipeline` + `run_segmented(...)`.
- `engines/timeseries/core_engine.py` segment dalı.
- `engines/champion.py` `BundleMetadata.segments`.
- testler + m5 benchmark (`--only m5_subset`, segment beklenen: intermittent/lumpy ayrı).
