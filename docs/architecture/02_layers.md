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
  - `segments` (ADR 0028): `per_group_champion` iken SBC intermittency sınıfına göre segment
    listesi (`_resolve_segments` — küçükler ADI ekseninde komşuya birleşir, ≤ `segment_max_count`);
    < 2 anlamlı segment → boş (pooled)
  - `row_policies`: `intermittent_augment:<class>` (havuz genişletir), `filter_low_activity`,
    `coldstart_split` · `regimes`: scenario_2 aktifse trend/volatility/joint
  - `family_policy`: gbdt/forest→minimal, linear→full, neural→minimal (ADR 0030, v1 bilgi amaçlı)
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
- **nöral (ADR 0030, `[neural]`):** `neural.yaml` `real_mlp`/`tab_m`/`real_tab_r` (pytabkit sklearn).
  `torch_env.configure_torch` (seed+determinizm, idempotent) · `neural_gate.prepare_neural_candidates`
  (çalışma-zamanı kapısı: `neural_enabled` auto/on/off × GPU × satır bandı; pytabkit varsa `mlp` düşer).
  `core.run_core_pipeline` çağırır.
- **nöral mimari arama (ADR 0031, `[neural-nas]`):** `neural_search=True` → `neural_arch_search`
  adayı (`class_path: __neural_arch__` → `neural_arch.TabularModelEstimator`, pytorch_tabular).
  `fine_tuners/arch_search.ArchitectureSearchTuner` — Aşama A aile taraması + Aşama B SH (koşullu
  uzay `_spaces/neural_arch_{small,full}.yaml`, `SearchDim.condition`). `resolve_tuner` heterojen.
  Serving `persistence.bundle` `champion_neural/` sidecar. Nöral aile → şampiyon bagging yok.

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
  leakage FAIL. **ADR 0027:** serving'de `auto_nonneg` kırpma uygulanacaksa küçük negatif
  tahminler karantina yapmaz (`serving_clip_lower` — `postprocessors.steps.resolve_clip_lower`
  yeniden kullanımı); negatif oranı > %50 ise yine karantina (miskalibre).
- `selection.py` — **1-SE kuralı** (default): birincil metrikte en iyinin `noise_floor`
  bandındaki en **basit/ucuz** aday (`_FAMILY_COMPLEXITY`). `best` = sadece en iyi metrik.
  `class_weighted_score` v1'de **bilgilendirme** (per-class SE yok → seçime girmez, v1.1).
  `promotion` = mutlak eşikler (smape_max/abs_bias_max/rmse_max/min_folds/leakage)
- `comparison_tests.py` — MCB ortalama rank + Diebold-Mariano (HLN düzeltmesi, scipy);
  forecasting + ≥3 fold, opsiyonel
- **Seçim yalnız OOF**; `noise_floor` (metrik SE medyanı), `selection_bias_bound = σ·√(2 ln K)`
- `RunConfig.promotion` (`PromotionConfig`); `GuardrailConfig` prediction eşikleri

## ensembling/  (ADR 0021 — KOD YAZILDI)
- **Girdi:** `[ValidationReport]` (hizalı OOF) + `[Candidate]` + `RunConfig` + `TaskSpec` + `DataProfile`
- **Çıktı:** `build_weighted_ensemble(...) -> (ValidationReport, Candidate, EnsembleSpec) | None`
- `greedy.py` — saf numpy Caruana GES: boş ensemble → her tur validation'ı en çok iyileştiren
  modeli **tekrarlı** ekle; 6-ondalık tie yuvarlama, tie-break en düşük indeks, **use_best**
  (görülen en iyi ensemble'a dön). `bagged_greedy_selection` — model alt-kümelerinde tekrarlı GES,
  seed'li, ağırlık ortalaması (Caruana 2006)
- Sentetik `weighted_ensemble` raporu + candidate `engines/core`'da `reports`/`candidates`'e eklenir →
  `select_champion` (1-SE) onu da tartar; `_FAMILY_COMPLEXITY["ensemble"]=5` (tek model eşitse tek model)
- **v1: regresyon + forecasting** (OOF nokta tahmini → ağırlıklı ortalama). Sınıflandırma GES v1.1
- `engines/champion._refit_ensemble` — her üye **postprocess'siz** refit → `FittedEnsemblePipeline`
  (`Σ w_i·member_i.predict` + tek ensemble-düzeyi postprocess)

## engines/  (ADR 0015 — orkestrasyon, KOD YAZILDI)
- **Girdi:** `Dataset` + `RunConfig` + `DataProfile` + `TaskSpec` · **Çıktı:** `EngineResult`
- `select_engine(task, config)` → `TabularCoreEngine` | `TimeSeriesCoreEngine`
  (`RunConfig.engines={"key": ...}` override)
- `core.run_core_pipeline` (ortak): `build_plan` → `resolve_candidates` →
  `run_validation_suite(tuner=resolve_tuner(config))` → `build_weighted_ensemble` → `score_reports` →
  `refit_champion` → `ModelBundle`
- `timeseries/reduction.py` (ADR 0004/0025) — **leakage-safe** zengin özellikler (`shift ≥ horizon`):
  lag + **mevsim-hizalı lag** (`slag`) + rolling mean/std/**min/max** + **mevsimsel rolling** +
  ewm + **fark** (`diff1`, `diffs`) + **takvim** (month/dow/... + sin/cos döngüsel).
  `build_reduction_features(frame, task, *, horizon, season, add_calendar, strategy, max_lag)`;
  `season` engine'den (`_season_length`); yeni kolonlar profile'a; `pre_transform` bundle'a.
  **seasonal target differencing (ADR 0026 A):** seasonal + `strategy="direct"` → `y_sdiff_ref`
  kolonu (`shift(H≥h)`); `TargetTransform` `seasonal_difference` seçeneği `forward=y−ref`/`inverse=y+ref`;
  `sdiff_ref`/`sdiff_ref_col` (frame_ops); `FittedModelPipeline.target_ref_col` slotu; planner
  `s≥h` + trend/mevsim gücü → varsayılan.
- `timeseries/recursive.py` (ADR 0026 B) — `RunConfig.forecast_reduction="recursive"`:
  `strategy="recursive"` (`shift(1)` tabanı, lag `1..k_max`); model 1-adım eğitilir,
  **`run_recursive_reports`** rolling-origin fold'da test bloğunu **recursive-`h`** tahmin eder
  (OOF = birikimli hata). Serving `FittedRecursivePipeline` (per-seri son `h` satır recursive).
  `run_core_pipeline(recursive=, recursive_season=)`; ansambl + bagging recursive modda kapalı.
- `timeseries/classical.py` (ADR 0023/0024) — klasik adaylar (`family∈{statistical,intermittent}`)
  **Nixtla `StatsForecast` native yolu**: `cross_validation` → OOF (rolling-origin, adaptif
  pencere + kısa seri filtresi + `SeasonalNaive` fallback) → per-model `ValidationReport` **+
  `classical_ensemble`** (EAT: aynı OOF matrisinde GES — M3/M4 winner deseni). Şampiyon klasik/EAT ise
  `FittedClassicalForecaster` (çok-model + ağırlık, `sf.predict` kolonlarının ağırlıklı ortalaması).
  **ADR 0029:** `predict` istenen `ds` aralığını kapsayacak değişken ufuk kullanır
  (`_horizon_for`, `[h, h·24+366]` clamp) — şampiyon `train−holdout`'ta fit edilse bile gerçek
  gelecek serve edilir (yoksa "son değer" fallback'ine düşüyordu).
  `run_core_pipeline(run_classical=True)` iki aileyi birleştirir; `RunConfig.classical_forecasting` bayrağı.
  **v1:** reduction↔klasik ortak GES dışlanır (cutoff ≠ fold OOF); reduction pooled
- **Tweedie ipucu (ADR 0024):** `planner._model_hints` — panelin ≥%50'si düzensiz talep →
  `AdaptivePlan.model_hints` (`lightgbm:objective=tweedie`, `hist_gbm:loss=poisson`) →
  `models.apply_model_hints` reduction GBDT adaylarına merge eder
- `champion.py` — **k-fold bagged refit (ADR 0022, varsayılan)**: tek model / %100 train yerine
  `bagging.folds` (5) fold-modeli, serving = ortalama (`FittedEnsemblePipeline` eşit ağırlık);
  bagged OOF postprocess'e girer. `k<2` / `bagging.enabled=False` / sınıflandırma → tek model
  (`refit_full` benzeri). GES şampiyonda üyeler de bagged. `_fit_one` (feature pipeline + target +
  estimator+ES), `_fit_pipeline` (bag/tek karar). `model_pipeline.FittedModelPipeline` +
  `ensemble_pipeline.FittedEnsemblePipeline` (`Predictor` protokolü)
- `runners/InProcessRunner` — engine'i sarar; çökme → `EngineResult(status=FAILED)`
- `segmented.py` (ADR 0028) — `plan.segments` varsa `TimeSeriesCoreEngine` segment başına
  `_run_pooled` koşar; `run_segmented` sonuçları `FittedSegmentedPipeline` (grup→segment
  yönlendirme, bilinmeyen→en büyük segment) + birleşik `EngineResult`'a katlar. Şampiyon tek
  `ModelBundle`, `model_key="segmented"`, segment haritası `adaptive_plan_summary["segments"]`.

## postprocessors/  (ADR 0017 — KOD YAZILDI)
- **Girdi:** `PostprocessConfig` + `DataProfile` + `TaskSpec` + `champ_report.oof`
- **Çıktı:** `build_postprocessor(...) -> Postprocessor` → `.fit(y_true?, y_pred?) -> FittedPostprocessor`
  (`ModelBundle.pipeline`'a gömülür; `predict()` target-inverse'ten sonra çağırır)
- Sıra: **`calibrate → clip → round → business_rule`** (`_POST_ORDER`)
- `calibrate` (OOF): `additive_bias` (`y − mean(resid)`) · `multiplicative` (`y · clamp(Σtrue/Σpred)`);
  OOF yoksa atlanır (WARNING). `linear`/isotonic → v1.1
- `clip`: `auto_nonneg` (varsayılan; `profile.target.min ≥ 0` + regresyon → `lower=0`) ·
  opsiyonel `auto_upper` (`pXX·mult`) · açık `lower`/`upper` kazanır
- `round`: `off`/`nearest`/`threshold` (DemandSensing eşikli)/`ceil`/`floor`
- **Leakage-safe (ADR 0011):** fit yalnız `refit_champion` içinde, OOF üzerinden
- **v1 sınırı:** `conformal.enabled` + `apply_in_validation` → `ValueError` (v1.1); `business_rule`
  hook `interfaces` enjekte eder (RunConfig'e girmez)
- `BundleMetadata.postprocess_summary` — uygulanan adımların serialize kaydı

## persistence/  (ADR 0018 — KOD YAZILDI)
- **Girdi:** `RunConfig` + `Dataset` + `EngineResult` (+ opsiyonel reports/holdout/timeline)
- **Çıktı:** `persist_run(...) -> (RunPaths, RunManifest)` — tam koşum dizini
- `paths.create_run_dir` → `<output_dir>/<DDMMYYYY>_<proje>_outputs/<run_id>/`
  (`run_id = UTC %Y%m%dT%H%M%SZ`; çakışma → `-01` soneki; dolu dizin sessizce ezilmez)
  alt: `models/ evaluation/ reports/ config_snapshot/`
- `bundle.save_bundle` / `load_bundle` — **joblib tek dosya** (`champion.joblib`: canlı
  `pipeline` + metadata + metrikler + `saved_env`). `load_bundle`: `format_version` uyuşmazlığı
  → hata; sklearn/lightgbm minor sapması → WARNING. **Güvenlik:** pickle kod çalıştırır → uyarı
- `manifest.build_manifest` — fingerprint + `config.model_dump(mode="json")` + env (python/os/
  paket sürümleri/git commit best-effort) + data snapshot + timeline; reprodüksiyona yeter
- `dump.write_json` deterministik (`sort_keys`, newline); `persist_evaluation` (scoreboard/
  selection/comparison_tests + opsiyonel fold/holdout); `persist_config_snapshot` (**`.env` asla**)
- yeni: `exceptions.PersistenceError`; nihai holdout tek-seferlik skorlama → orchestrator (ADR 0020)

## reporters/  (ADR 0019 — KOD YAZILDI)
- **Girdi:** `EngineResult` + `RunManifest` + `RunPaths` (+ opsiyonel `reports`)
- **Çıktı:** `write_reports(...) -> dict[str,str]` (artifacts) — `paths.reports/` içine
- `run_report.html` (**her zaman**, tek dosya, CDN/harici asset yok, `html.escape`),
  `model_card.md` (**her zaman**, Mitchell bölümleri; oto + `TODO` placeholder),
  `leaderboard.csv` (**her zaman**, `scoreboard_to_frame`)
- `plots/*.png` yalnız `[report]` extra (matplotlib) varsa; yoksa atlanır (WARNING) — akışı kırmaz
- Deterministik: zaman yalnız `manifest.created_at`; `pipeline is None` → importance atlanır

## tracking/  (ADR 0019 — KOD YAZILDI, opsiyonel)
- Protokol `Tracker`: `start_run / log_params / log_metrics / log_artifact / end_run`
- `resolve_tracker(config, run_dir)` → `NullTracker` (none) · `JsonlTracker` (varsayılan,
  `tracking/events.jsonl` + `summary.json`, bağımlılıksız, **ağsız**) · `MlflowTracker`
  (`[tracking]` extra — yoksa `ConfigError`, sessizce jsonl'a düşmez)
- `persistence.RunPaths.tracking` alt dizini eklendi

## llm/  (v2 — şimdilik iskele)
- `LLMProvider`: `complete`, `stream`, `embed`
- providers: openai, anthropic, bedrock, azure_openai, local, null(varsayılan)
- Çekirdek bağımsız; sırlar env'den

## interfaces/  (ADR 0020 — KOD YAZILDI)
- **Girdi:** ham veri + `RunConfig` · **Çıktı:** `RunResult`
- `orchestrator.Orchestrator.run(data, config, *, resolution=None, tracker=None)` — tek akış:
  `create_run_dir` → `tracker.start_run` → **[io]** `load_dataset` → **[analyze]** `analyze` →
  **[holdout_split]** `split_holdout` → **[engine]** `select_engine` + `runner.run(train_dataset)` →
  **[holdout_score]** `score_holdout` → `champion.metrics_holdout` (**test'e TEK dokunuş**) →
  **[persist]** `persist_run` → **[report]** `write_reports` → manifest'i tam timeline ile yeniden yaz →
  `tracker.log_*` + `end_run` → `RunResult`
- io/analyze hataları **propagate**; engine `FAILED` → akış devam, minimal manifest+rapor, `RunResult`
  status=FAILED, CLI exit 1. Her stage `TimelineEntry`.
- `holdout.split_holdout` — `n_rows < ceil(min_rows_for_cv/(1−holdout_fraction))` → holdout yok (WARNING);
  tabular seed'li rastgele; TS son `min(horizon, n_periods−1)` dönem (`[group,time]` stable sıralı,
  `shift(horizon)` leakage-safe). `score_holdout` TS'de tüm sıralı frame'de predict + mask ile seçer.
- `api.AutoRagML(preset=, config_file=, **overrides).fit(data, *, target, time_col=, group_col=)` →
  `RunResult`; `.leaderboard/.predict/.explain/.champion/.manifest`. `AutoRagML.load(bundle_path)` →
  `LoadedChampion` (serving).
- `cli.main` — `autoragml run --data --target [--preset --config --time-col --group-col --output-dir
  --project-name]`; leaderboard(10)+şampiyon+promotion+çıktı dizini stdout. argparse, ek dep yok.
- `agent_tools.TOOLS` — `autoragml_run` JSON-schema tanımı (v2 executor yok).
- v1: hep `InProcessRunner`; `explain()` yapısal; SHAP → v1.1.
