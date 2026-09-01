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

## analyzers/  (deterministik "perception" — ADR 0010, KOD YAZILDI)
- **Girdi:** `Dataset` + `RunConfig` · **Çıktı:** `analyze() → (DataProfile, TaskSpec)`
- `_frame.py` (eager→handle; lazy→örneklem `analyzers.profiling_sample_rows` + düşük güven)
- İç sıra: `modality.detect` → `profiling.build_column_profiles` (raw_dtype + special_types
  + semantic_role + flags + duplicate_of; numpy skew/kurtosis) → `task_inference.infer_task`
  → `timeseries.diagnose_timeseries` (`pandas.infer_freq` + freq→periyot sözlüğü +
  numpy-ACF mevsimsellik + OLS-R² trend; per-series ADI/CV² + SBC sınıf; toplulaştırılmış
  seri üzerinde seasonality/trend) → `quality.scan` + `leakage.scan` (yumuşak, WARNING)
- Eşikler `RunConfig.analyzers` (`ThresholdConfig` + `TimeSeriesAnalyzerConfig`)
- **Model eğitmez, fit etmez.** Düşük güven → WARNING, akış durmaz.
- `statsmodels` yok: `stationarity_pvalue=None` (v1); KH şeması → SBC + uyarı (takip).

## dynamics/  (veriye-özel strateji — ADR 0007/0010/0015, KOD YAZILDI)
- `planner.build_plan(DataProfile, TaskSpec, RunConfig) -> AdaptivePlan` (deterministik; **fit yok**)
  - `committed_ops`: yapısal drop (constant/duplicate/id/text) · `date_expand` · `encode`
    (kardinaliteye göre onehot/target_encode) · `impute` · `recipe:<ad>` refleri
  - `candidate_ops`: `heavy_tailed_numeric` + `target` grupları (log1p/yeo_johnson/quantile —
    HPO uzayında; hedef negatifse log1p elenir)
  - `structure`: auto → forecasting + group_col + yeterli seri/geçmiş ise `per_group_champion`
  - `row_policies`: `intermittent_augment:<class>` (havuz genişletir), `filter_low_activity`,
    `coldstart_split` · `regimes`: scenario_2 aktifse trend/volatility/joint
  - `family_policy`: gbdt/forest→minimal, linear/neural→full
- `recipes/` — registry: `@register_recipe` · `RunConfig.dynamics.recipes` +
  `load_recipe_paths` · entry-points `autoragml.recipes`. İsim çakışması → `RecipeError`.
  `planner` recipe adlarını **plan zamanında** doğrular (fail-fast).
- `autoragml/transform.py` — `Transform` / `FittedTransform` protokolleri (ADR 0011);
  `preprocessors` ve `recipes` buna uyar.
- `synthesis.py` — v2 (LLM recipe üretimi); v1'de boş.
- `RunConfig.dynamics` (`DynamicsConfig`).

## preprocessors/  (ADR 0011 — leakage-safe by construction, KOD YAZILDI)
- **Girdi:** `AdaptivePlan` (`committed_ops` + seçilmiş `candidate_ops`) + train fold
- **Çıktı:** `FittedFeaturePipeline` (immutable, yalnız `apply`)
- `FeaturePipeline.from_plan(plan, candidate_choices)` → sıralı transform zinciri
  (drop → recipe → date_expand → impute → encode → numeric candidate)
- `catalog.py` — op → sklearn transformer sarımı (`base.SklearnColumnTransform`):
  impute · onehot/ordinal/**target_encode**(iç cross-fitting)/hashing · scale ·
  yeo_johnson · quantile · `stateless.py` (drop/date_expand/log1p/hashing)
- `target.py` — `TargetTransform` (`y` forward/inverse; engine estimator etrafında)
- `fit`/`fit_transform` **yalnız `validators`** çağırır (fold içinde). `fit_transform ≠
  fit + apply` — target_encode train'e cross-fit, test'e full-train
- `autoragml/transform.py`: `Transform.fit_transform` protokole eklendi + `BaseTransform`

## models/ + registry  (ADR 0012 — YAML katalog, KOD YAZILDI)
- **Girdi:** `RunConfig` + `TaskSpec` · **Çıktı:** `resolve_candidates(config, task) -> [Candidate]`
- `models/catalog/*.yaml` pakete gömülü (`tabular` / `baselines` / `timeseries`) +
  `RunConfig.model_catalog_override` key-bazında deep-merge
- `registry.py` — entry → `Candidate`: `requires` kurulu mu (`find_spec`) + `class_path`
  importable mı; değilse atla + **tek WARNING**. `enabled: false` → dışarıda.
  Modalite + `task in candidate.tasks` filtresi. Entry-points `autoragml.models` ikincil.
- `estimator.py` — `resolve_class_path` (forecasting → reduction ile `regression` path'i) +
  `build_estimator` (default + HPO paramları merge; `wrap` → `SimpleImputer`+model Pipeline)
- Katalog: sklearn linear/ridge/lasso/elastic_net/logistic/RF/ET/hist_gbm/mlp ·
  lightgbm (çekirdek) · xgboost/catboost (ops.) · Dummy baseline'lar ·
  statsforecast AutoARIMA/ETS/Theta/MSTL/Croston/TSB (ops. `[timeseries]`) — TS engine tüketir

## fine_tuners/  (ADR 0013 — ensemble-öncelikli, multi-fidelity, KOD YAZILDI)
- **Girdi:** `validators.Tuner` protokolü (candidate + dış-fold train frame + plan + config)
- **Çıktı:** `TunerOutcome` (`best_params`, `candidate_choices`, `nested`, `TuningResult`)
- `resolve_tuner(config)` → `hpo_level=none`→`DefaultTuner`; `light`→`RandomSearchTuner`
  (1 iç holdout); `thorough`→3-fold iç CV; `hpo_backend=optuna`→`OptunaTuner`
- `random_search.py` — random search + `Candidate.fidelity` varsa **Successive Halving**
  (`halving.py`: eta=3, her rung bütçe ×eta / hayatta kalan ÷eta). Bütçe kooperatif;
  ilk deneme sonrası projeksiyon uyarısı (ADR 0008/1)
- `optuna_backend.py` — TPE sampler (`[hpo]` extra). **v1 sınırı:** ara-adım `trial.report`
  yok → sabit fidelity (gerçek pruning callback v1.1)
- `inner_eval.py` — `build_inner_splits` + `evaluate_trial` (feature pipeline + target
  transform + estimator, **yalnız dış-fold train içinde**; higher-is-better metrik negatiflenir)
- `space.py` — `SearchDim` → değer (int/float/loguniform/categorical)
- Paylaşılan fold-frame yardımcıları `validators/frame_ops.py`
- `RunConfig.hpo_backend` (`HpoBackend`)

## validators/  (ADR 0010/6 + 0011 — split sınırını yöneten TEK yer, KOD YAZILDI)
- **Girdi:** `Candidate` + frame + `AdaptivePlan` + `DataProfile` + `TaskSpec` + `RunConfig`
- **Çıktı:** `run_validation(...) -> ValidationReport` · `run_validation_suite(...)` (aynı split)
- `splitters.py` — Holdout / KFold / StratifiedKFold / GroupKFold / RollingOrigin /
  FixedWindow + `resolve_splitter` (`split_policy` kısmi override + görev tabanlı;
  küçük veri → holdout). Frame pozisyonel indeksli.
- `runner.py` — dış fold döngüsü: `Tuner.tune` (iç resample; `DefaultTuner` = plan
  varsayılanları, `nested=False`) → `FeaturePipeline.fit_transform(train)` +
  `apply(test)` → `TargetTransform` fit(train_y) → `build_estimator` + fold-içi iç-val
  early stopping (lightgbm/sklearn-HGB) → predict → `scoring.metrics.compute_metrics`.
  OOF metrik + fold'lar arası SE.
- `leakage_checks.py` — `overlap` (satır/zaman/grup) · `preprocessing` (provenance != train)
  → `LeakageReport`. `multi_test` runner tarafından garanti (tuner test'i görmez).
- `RunConfig.validation` (`ValidationConfig`).

## scoring/  (ADR 0014 — dürüst seçim, KOD YAZILDI)
- **Girdi:** `[ValidationReport]` + `[Candidate]` + `RunConfig` + `TaskSpec` + `DataProfile`
- **Çıktı:** `score_reports(...) -> SelectionResult` (`ScoreBoard` + champion + promotion)
- `metrics/` — sMAPE/MAPE/WMAPE/RMSE/MAE/bias/CSL + accuracy/f1_macro/balanced_accuracy
- `guardrails.py` — quarantine: non-finite metrik · `prediction_health` (negatif/aşırı/
  scale-ratio; negatif yalnız `target_min≥0`) · metrik tavanları · model×scenario blocklist ·
  leakage FAIL
- `selection.py` — **1-SE kuralı** (default): birincil metrikte en iyinin `noise_floor`
  bandındaki en **basit/ucuz** aday (`_FAMILY_COMPLEXITY`). `best` = sadece en iyi metrik.
  `class_weighted_score` v1'de **bilgilendirme** (per-class SE yok → seçime girmez, v1.1).
  `promotion` = mutlak eşikler (smape_max/abs_bias_max/rmse_max/min_folds/leakage)
- `comparison_tests.py` — MCB ortalama rank + Diebold-Mariano (HLN düzeltmesi, scipy);
  forecasting + ≥3 fold, opsiyonel
- **Seçim yalnız OOF**; `noise_floor` (metrik SE medyanı), `selection_bias_bound = σ·√(2 ln K)`
- `RunConfig.promotion` (`PromotionConfig`); `GuardrailConfig` prediction eşikleri

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
