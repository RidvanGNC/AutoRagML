# Katmanlar — sorumluluk sözleşmeleri (TASLAK)

Her katman: **saf dönüşüm**, girdi contract → çıktı contract. Yan etki yalnız
`persistence` / `tracking`. Aşağıdaki "girdi/çıktı" satırları kesinleşince kod başlar.

---

## config/
- **Girdi:** kullanıcı YAML/kwargs + preset adı + paket varsayılanı
- **Çıktı:** `RunConfig`
- Katmanlı merge: varsayılan ← preset ← kullanıcı dosyası ← runtime override
- Zorunlu alan yok; boş config akıllı varsayılanla çalışır (analyzers doldurur)

## io/  (ADR 0009)
- **Girdi:** DataFrame · `.csv` · `.parquet` · csv/parquet klasörü · (opsiyonel) DB (`[db]`)
- **Çıktı:** `Dataset`
- Sorumluluklar:
  - kaynak boyut yoklaması → eager/lazy karar (`RunConfig.io.eager_max_bytes`)
  - şema çıkarımı; `n_rows` **tam sayım**
  - wide → long `melt` (tespit + log; `layout` işaretle)
  - **strict fingerprint**: kanonik form üzerinden tek streaming geçişte SHA256
  - DB adaptörü opsiyonel (SQLAlchemy)
- v1: yalnız `modparts.tabular`. `relations` alanı `None` (rezerve).
- Bulut: `fsspec` opsiyonel.

## analyzers/  (deterministik "perception" — ADR 0010)
- **Girdi:** `Dataset` + `RunConfig`
- **Çıktı:** `DataProfile` + `TaskSpec`
- İç sıra: `modality.detect` → `profiling.build` (ColumnProfile: raw_dtype +
  special_types + semantic_role + flags) → `task_inference.infer` →
  `timeseries.diagnose` (infer_freq + freq→periyot sözlüğü + ACF/STL doğrulama;
  per-series ADI/CV² + intermittency **ölçümü**) → `quality.scan` + `leakage.scan`
- **Model eğitmez, fit etmez.** `provenance == "full"` görür.
- Düşük güven → WARNING, akış durmaz.

## dynamics/  (veriye-özel strateji — ADR 0007)
- `planner.py` — **Girdi:** `DataProfile` + `TaskSpec` + `RunConfig` · **Çıktı:** `AdaptivePlan`
  (deterministik katalog seçimi; kod üretmez; **fit yok**)
- `recipes/` — custom transform kayıt yeri; `preprocessors` arayüzüne uyan sınıflar
  (v1: elle yazılır). `AdaptivePlan` bunlara `recipe:"<ad>"` ile referans verir.
- `synthesis.py` — **v2:** LLM recipe üretir → `engines/runners` (Subprocess/Container)
  içinde doğrular → `recipes/`'e kaydeder. v1'de yok.
- Custom kod modele değil, pipeline'a girer; fit yalnız train fold'unda (`validators`).

## preprocessors/  (ADR 0011 — leakage-safe by construction)
- **Girdi:** `AdaptivePlan` (`committed_ops` + seçilmiş `candidate_ops`) + train fold
- **Çıktı:** `FittedTransform` (immutable)
- Üç ilkel ayrı: stateless `transform` / `fit(train_frame)` / `apply(X)`
- `fit`'i **yalnız `validators`** çağırır; split sınırını görmez
- `provenance_fitted_on` kaydı; `serialize()` → `ModelBundle`

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

## validators/  (ADR 0010/6 + 0011 — split sınırını yöneten TEK yer)
- **Girdi:** `Candidate` + `Dataset` + `AdaptivePlan` + split policy
- **Çıktı:** `ValidationReport`
- Split: holdout, kfold, stratified, group, TimeSeriesSplit, RollingOrigin, FixedWindow
- **Nested CV**: HPO + `candidate_ops` seçimi + (opsiyonel) feature selection **iç
  resample**'da; dış fold yalnız skorlar (multi-test leakage yok)
- Feature selection etkinse → iç-fold **konsensüs** (cnCV)
- Dış fold döngüsü: `FittedTransform.fit(train)` → model fit (early stop) →
  `apply(test)` → predict → metrik. Regime/dynamics fit de fold içinde.
- `leakage_checks` 3 kategori: `overlap` (satır/zaman örtüşmesi) · `preprocessing`
  (split öncesi fit) · `multi_test` (dış test'te seçim) → **BLOCK**

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
