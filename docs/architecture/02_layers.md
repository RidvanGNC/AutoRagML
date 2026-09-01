# Katmanlar — sorumluluk sözleşmeleri (TASLAK)

Her katman: **saf dönüşüm**, girdi contract → çıktı contract. Yan etki yalnız
`persistence` / `tracking`. Aşağıdaki "girdi/çıktı" satırları kesinleşince kod başlar.

---

## config/  (ADR 0016 — KOD YAZILDI)
- **Girdi:** `target` + preset adı + kullanıcı YAML + runtime override
- **Çıktı:** `ConfigResolution` (`config: RunConfig`, `provenance: dict[path→layer]`, `layers[]`)
- Katmanlı merge: `defaults ← preset (extends zinciri, her biri ayrı katman) ← file ← overrides`.
  Derin merge; scalar/list/None değiştirir; bilinmeyen alan → `ConfigError`
- `resolve_run_config()` — `merge.py` (provenance), `presets.py` (`extends` + döngü),
  `loaders.py` (YAML)
- Yerleşik presetler pakete gömülü: `config/_presets/*.yaml` (`list_presets()`)
- `config/settings.py` — `Settings`: `.env` + ortam; `resolve_secret(name)→SecretStr`;
  asla serialize edilmez. `RunConfig.*_env` adlarını çözer
- `target` zorunlu; forecasting preset'i `time_col` ister (RunConfig validator)

## io/  (ADR 0009 — KOD YAZILDI)
- **Girdi:** DataFrame · `.csv`/`.tsv` · `.parquet` · csv/parquet klasörü · `DbSource` (`[db]`)
- **Çıktı:** `Dataset` — `load_dataset(src, config)`
- `sources.py` kaynak çözümleme + boyut yoklama → `__init__.py` eager/lazy karar
  (`config.io.eager_max_bytes`, varsayılan 1 GiB)
- `fingerprint.py` — **strict** (sıra-bağımsız çoklu-küme hash: `sum` mod 2^64 + `xor` +
  `count` over `hash_pandas_object`; sort gerektirmez → streaming) + **fast** (structural)
- `layout.py` — wide tespiti + `melt` (eager; lazy+wide → hata) + `long/single_series/n-a`
- `lazyframe.py` — `LazyFrame` (chunk akışı, `iter_chunks`, `to_pandas`)
- `db.py` — `DbSource` + `read_sql` (sqlalchemy lazy import)
- v1: `modparts={"tabular": ...}`, `relations=None`

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

## models/ + registry  (ADR 0012 — YAML katalog)
- **Girdi:** `TaskSpec` + `AdaptivePlan` + `RunConfig` + `configs/model_catalog/*.yaml`
  (+ kullanıcı override YAML)
- **Çıktı:** `[Candidate]`
- registry: YAML okur → `class_path` importable + `requires` kurulu mu → `TaskSpec.task`
  uyumlu entry'leri `Candidate`'e çevirir; eksik dep → atla + tek WARNING
- Katalog: sklearn linear/ridge/lasso/RF/ET/GBM · lightgbm · xgboost/catboost(ops.) ·
  naive/seasonal_naive/trend_naive/stl baseline · statsforecast(ops.) ·
  Croston/TSB/SBA (intermittent — havuz genişletme)
- Entry-points = ikincil (üçüncü parti plugin); YAML kanonik

## fine_tuners/  (ADR 0013 — ensemble-öncelikli, multi-fidelity)
- **Girdi:** `Candidate` + train (iç resample) + `budget`
- **Çıktı:** `TuningResult`
- Backend: `RandomSearch` (çekirdek) + SH/Hyperband zamanlayıcı · `Optuna` (`[hpo]`,
  HyperbandPruner) · `FLAML` CFO/BlendSearch (ops.)
- `hpo_level`: `none` (sadece ensemble) · `light` (**default**, ~15 trial + pruning) · `thorough`
- Fidelity: `Candidate.fidelity` (GBM→n_estimators / büyük veri→subsample / erken rung→az fold)
- Early stopping: **fold-içi iç-val** (train'den `early_stopping_fraction`, TS'de son parça) —
  ADR 0011 uyumlu; opt-in CV-ES küçük veride
- **Yalnız iç resample'da çalışır** (dış test'e dokunmaz — ADR 0010/6)
- Per-candidate timeout enforce (ADR 0014/6)

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

## scoring/  (ADR 0014 — dürüst seçim)
- **Girdi:** `[ValidationReport]` (OOF) + `RunConfig` + (opsiyonel) demand-class
- **Çıktı:** `ScoreBoard` + `SelectionResult`
- Alt: `metrics/` · `guardrails.py` (quarantine — DemandSensing) · `selection.py`
  (**1-SE kuralı** default + class-weighted + promotion rules) · `comparison_tests.py`
  (MCB / Diebold-Mariano, forecasting, opsiyonel)
- **Seçim yalnız OOF/validation**; test'e tek dokunuş `engines`'te; `noise_floor`
  (metrik SE), `selection_bias_bound` σ√(2 ln K), `realized_seconds` + K raporlanır

## engines/  (ADR 0015 — orkestrasyon)
- **Girdi:** `Dataset` + `RunConfig` + `DataProfile` + `TaskSpec`
- **Çıktı:** `EngineResult`
- İç akış: `dynamics.planner` → `registry.resolve` → `validators` (nested CV) →
  `scoring` (OOF seçim) → şampiyon refit (tüm train) + postprocessors → `ModelBundle`
- `tabular/core_engine`, `timeseries/core_engine`, opsiyonel `statsforecast_engine`
- `runners/`: InProcess (varsayılan) | Subprocess | Container/Remote (v2+)
- Üstünde `Orchestrator` (interfaces/api): config→io→analyzers→engine seçimi→engine(ler)
  →(çok engine ise en iyi / v1.1 ensemble)→final holdout (bir kez)→persistence→reporters
  → `RunResult`

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
