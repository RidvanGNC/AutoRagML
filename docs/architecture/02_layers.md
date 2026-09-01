# Katmanlar — sorumluluk sözleşmeleri (TASLAK)

Her katman: **saf dönüşüm**, girdi contract → çıktı contract. Yan etki yalnız
`persistence` / `tracking`. Aşağıdaki "girdi/çıktı" satırları kesinleşince kod başlar.

---

## config/
- **Girdi:** kullanıcı YAML/kwargs + preset adı + paket varsayılanı
- **Çıktı:** `RunConfig`
- Katmanlı merge: varsayılan ← preset ← kullanıcı dosyası ← runtime override
- Zorunlu alan yok; boş config akıllı varsayılanla çalışır (analyzers doldurur)

## io/
- **Girdi:** dosya yolu / DataFrame / klasör
- **Çıktı:** `Dataset` (lazy handle, schema, fingerprint)
- Yükleyiciler: table (csv/parquet/df), image-folder, audio-folder, text, multimodal
- Bulut: `fsspec` opsiyonel; büyük medyayı RAM'e almaz

## analyzers/  (deterministik "perception")
- **Girdi:** `Dataset` + `RunConfig`
- **Çıktı:** `DataProfile` + `TaskSpec`
- Alt: `modality`, `task_inference`, `profiling`, `quality`, `timeseries` (freq,
  seasonality, ADI/CV² intermittency, regime ipuçları)
- **Model eğitmez.**

## dynamics/  (veriye-özel strateji)
- **Girdi:** `DataProfile` + `TaskSpec` + `RunConfig`
- **Çıktı:** `AdaptivePlan`
- Kolon işlemleri, satır politikaları, yapısal seçim (pooled vs per-group champion),
  hedef dönüşümü, regime tanımı
- **Fit yok** — karar üretir; fit'i `preprocessors`/`validators` yapar

## preprocessors/
- **Girdi:** `AdaptivePlan` + train fold
- **Çıktı:** fitted `Preprocessor` (transform uygulanabilir)
- Leakage-safe: fit yalnız train'de, `validators` içinden
- Serialize edilebilir → `ModelBundle`

## models/ + registry
- **Girdi:** `TaskSpec` + `AdaptivePlan` + `RunConfig`
- **Çıktı:** `[Candidate]`
- Tablo/TS: sklearn linear/RF/ET/GBM, lightgbm, xgboost(opsiyonel), catboost(opsiyonel),
  naive/seasonal-naive/STL baseline
- registry: entry-points ile dış model eklenebilir

## fine_tuners/
- **Girdi:** `Candidate` + train/val + `budget`
- **Çıktı:** `TuningResult`
- Backend: RandomSearch (çekirdek), Optuna (opsiyonel), FLAML-CFO (opsiyonel)
- Early stopping: (a) model-içi (xgb/lgbm rounds), (b) arama-seviyesi (plateau/bütçe)

## validators/
- **Girdi:** `Candidate` (tuned) + `Dataset` + `AdaptivePlan` + split policy
- **Çıktı:** `ValidationReport`
- Split: holdout, kfold, stratified, group, TimeSeriesSplit, RollingOrigin, FixedWindow
- Fold döngüsü: preprocessors.fit(train) → model fit (early stop) → predict(test) → metrik
- Fold-güvenli regime/dynamics fit
- `leakage_checks`: tarih örtüşmesi, shift, target-türevli feature

## scoring/
- **Girdi:** `[ValidationReport]` + `RunConfig` + (opsiyonel) routing/demand-class
- **Çıktı:** `ScoreBoard` + `SelectionResult`
- `metrics` (regresyon/sınıflandırma/forecasting/CSL), `guardrails` (quarantine),
  `selection` (task/sınıf-bazlı metrik önceliği, promotion rules)

## engines/
- **Girdi:** `Dataset` + `RunConfig` + `DataProfile` + `TaskSpec`
- **Çıktı:** `EngineResult` (ScoreBoard + champion `ModelBundle`)
- `tabular/core_engine`, `timeseries/core_engine`, opsiyonel `statsforecast_engine`
- `runners/`: InProcess (varsayılan) | Subprocess | Container/Remote (v2+)

## postprocessors/
- **Girdi:** ham tahmin + `AdaptivePlan` + business kuralları
- **Çıktı:** düzeltilmiş tahmin; `ModelBundle`'a gömülür
- clip, round (eşikli), quantile calibrate, conformal interval, business-rule hook

## persistence/
- **Girdi:** champion pipeline + tüm contract nesneleri
- **Çıktı:** `ModelBundle` dosyası + `RunManifest` + `outputs/<DDMMYYYY>_<proje>_outputs/<run_id>/`
  alt yapısı (`evaluation/`, `models/`, `reports/`, `config_snapshot/`)

## reporters/
- **Girdi:** contract nesneleri + `RunManifest`
- **Çıktı:** dosyalar (MD/HTML her zaman; PDF/xlsx opsiyonel)
- EDA raporu, model card, karşılaştırma tablosu, actual-vs-pred / fold / importance grafikleri

## tracking/  (opsiyonel)
- Protokol: `on_run_start / log_params / log_metrics / log_artifact / on_run_end`
- `JsonlTracker` varsayılan (bağımlılıksız), `MlflowTracker` opsiyonel, kapalı = no-op

## llm/  (v2 — şimdilik iskele)
- `LLMProvider`: `complete`, `stream`, `embed`
- providers: openai, anthropic, bedrock, azure_openai, local, null(varsayılan)
- Çekirdek bağımsız; sırlar env'den

## interfaces/
- `api.py`: `AutoRagML().fit / predict / leaderboard / explain`
- `cli.py`: `autoragml run --data ... --target ... --preset ...`
- `agent_tools.py`: aynı fonksiyonların JSON-schema tanımı (v2 agent katmanı çağırır)
