# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) formatını,
sürümleme [SemVer](https://semver.org/lang/tr/)'i izler.

Her PR bu dosyaya bir satır ekler. `[Unreleased]` altında biriktirilir,
release'te tarih + sürüm ile başlığa taşınır ve git tag atılır.

## [Unreleased]

### Eklendi
- **`engines/` katmanı kodlandı** (ADR 0015): `select_engine(task, config)` + `InProcessRunner`.
  - `core.run_core_pipeline` — ortak akış: `build_plan` → `resolve_candidates` → `run_validation_suite(tuner=resolve_tuner)` → `score_reports` → `refit_champion` (tüm train; ES modelde n_estimators = validation best_iteration medyanı) → `ModelBundle`
  - `TabularCoreEngine` / `TimeSeriesCoreEngine`
  - `timeseries/reduction.py` — **leakage-safe** lag/rolling/ewm özellikleri (`shift ≥ horizon` → h-adım direkt tahminde test yalnız train `y`'sini görür); recursive multi-step v1.1
  - `model_pipeline.FittedModelPipeline` — `ModelBundle.pipeline` runtime nesnesi (`pre_transform` → feature pipeline → estimator → target inverse); TS'de reduction FE predict'te yeniden uygulanır
  - `runners/` — `EngineRunner` protokolü + `InProcessRunner` (çökme → `status=FAILED`)
- Sözleşme: `ValidationReport.prediction_health` + `.oof` (`OOFArrays`), `ScoreRow.family` (scoring turunda eklendi); `io.materialize_frame`.
- `tests/unit/engines/` — 9 test (reduction leakage-safety + grup izolasyonu, tabular/TS uçtan uca, predict, runner failure wrap).

### Değişti
- pytest: `ConvergenceWarning` + `eval_set` DeprecationWarning filtrelendi; engine e2e testleri `hpo_level=none` ile hızlandı.
- **v1 sınırı:** `per_group_champion` planlanır, pooled ile ilerlenir (per-group refit v1.1).
- **`scoring/` katmanı kodlandı** (ADR 0014): `score_reports(reports, candidates, config, task, profile) -> SelectionResult`.
  - `guardrails.py` — quarantine: non-finite metrik · prediction_health (negatif/scale-ratio; negatif yalnız hedef≥0) · metrik tavanları · model×scenario blocklist · leakage FAIL
  - `selection.py` — **1-SE kuralı** (default): birincil metrikte en iyinin noise_floor bandındaki en basit/ucuz aday; `best` alternatifi; `promotion` (mutlak eşikler); `class_weighted_score` v1'de bilgilendirme (per-class SE yok → v1.1)
  - `comparison_tests.py` — MCB ortalama rank + Diebold-Mariano (HLN düzeltmesi, scipy); forecasting + ≥3 fold
  - `__init__.build_scoreboard` — `noise_floor` (SE medyanı) + `selection_bias_bound = σ√(2lnK)`
- Sözleşme: `RunConfig.promotion` (`PromotionConfig`), `GuardrailConfig` prediction eşikleri, `ValidationReport.prediction_health` + `.oof` (excluded `OOFArrays`), `ScoreRow.family`.
- Çekirdek dep: `scipy>=1.11` (zaten sklearn/lightgbm transitif). `validators.frame_ops` — `OOFArrays` + `prediction_health`.
- `tests/unit/scoring/` — 18 test.
- **`fine_tuners/` katmanı kodlandı** (ADR 0013): `validators.Tuner` protokolünü gerçekler.
  - `random_search.py` — `RandomSearchTuner`: random search + `Candidate.fidelity` varsa **Successive Halving** (`halving.py`, eta=3, doğrulanmış formül). Kooperatif bütçe + ADR 0008/1 projeksiyon uyarısı
  - `optuna_backend.py` — `OptunaTuner` (TPE, `[hpo]` extra). v1: sabit fidelity (ara-adım pruning callback v1.1)
  - `inner_eval.py` — `build_inner_splits` + `evaluate_trial` (yalnız dış-fold train içinde)
  - `space.py` — `SearchDim` örnekleme; `resolve_tuner(config)` (level+backend)
- `RunConfig.hpo_backend` (`HpoBackend`); `TunerOutcome.tuning_result` (`contracts.TuningResult`).
- `validators/frame_ops.py` — paylaşılan fold-frame yardımcıları (runner + fine_tuners).
- Dev dep: `optuna` (backend testleri için).
- `tests/unit/fine_tuners/` — 14 test.
- **`validators/` katmanı kodlandı** (ADR 0010/6 + 0011 + 0013): split sınırını yöneten tek yer.
  - `splitters.py` — Holdout / KFold / StratifiedKFold / GroupKFold / **RollingOrigin** (genişleyen train, sabit horizon, zaman sızıntısı yok) / FixedWindow + `resolve_splitter` (`split_policy` kısmi override + görev tabanlı; küçük veri → holdout)
  - `runner.py` — nested CV: `Tuner.tune` iç resample (`DefaultTuner` = plan varsayılanları, `nested=False`) → `FeaturePipeline.fit_transform(train)` + `apply(test)` → `TargetTransform` fit(train_y) → `build_estimator` + fold-içi iç-val early stopping (lightgbm/sklearn-HGB) → `compute_metrics`. OOF metrik + fold'lar arası SE. `run_validation_suite` (aynı split, çöken aday atlanır)
  - `leakage_checks.py` — overlap (satır/zaman/grup) + preprocessing (provenance) → `LeakageReport`
- **`scoring/metrics/` kodlandı** — sMAPE/MAPE/WMAPE/RMSE/MAE/bias/CSL + accuracy/f1_macro/balanced_accuracy; `compute_metrics(y, ŷ, task)`, `default_primary_metric`, `lower_is_better`.
- `RunConfig.validation` (`ValidationConfig`).
- `tests/unit/validators/` — 19 test (rolling-origin zaman sızıntısı, resolve, leakage, nested tuner, suite skip).
- **`models/` + registry kodlandı** (ADR 0012): `resolve_candidates(config, task) -> [Candidate]`.
  - `models/catalog/*.yaml` pakete gömülü (tabular/baselines/timeseries); wheel'e dahil
  - `registry.py` — YAML deep-merge + `requires` (`find_spec`) + `class_path` importable kontrolü → eksikse atla + tek WARNING; `enabled:false`; modalite + task filtresi; entry-points `autoragml.models` ikincil
  - `estimator.py` — `resolve_class_path` (forecasting → reduction `regression` path'i) + `build_estimator` (param merge; `wrap` → imputer+model Pipeline)
  - katalog: sklearn linear/ridge/lasso/elastic_net/logistic/RF/ET/hist_gbm/mlp + lightgbm (çekirdek) + xgboost/catboost (ops.) + Dummy baseline + statsforecast (ops.)
- `RunConfig.model_catalog_override` (zaten vardı) artık `registry` tarafından tüketiliyor.
- `tests/unit/models/` — 19 test.

### Değişti
- Repo-kök `configs/model_catalog/` stub YAML'ları kaldırıldı (katalog pakete taşındı); dizin = kullanıcı override örneği.
- **`preprocessors/` katmanı kodlandı** (ADR 0011): `FeaturePipeline.from_plan(plan, candidate_choices)` → leakage-safe dönüşüm zinciri.
  - `base.py` — `SklearnColumnTransform` (kolon alt kümesine sklearn transformer; `cross_fitted` → `fit_transform` cross-fitting / `apply` full-train)
  - `catalog.py` — op → transform: impute · onehot/ordinal/**target_encode**(sklearn iç cross-fitting)/hashing · scale · yeo_johnson · quantile
  - `stateless.py` — drop / date_expand / log1p (işaretli) / hashing (CRC32 deterministik)
  - `pipeline.py` — sıralı zincir (drop→recipe→date_expand→impute→encode→numeric); `fit_transform` her adımda cross-fitting'i korur
  - `target.py` — `TargetTransform` (none/log1p/yeo_johnson/quantile; forward/inverse)
- `autoragml/transform.py`: `Transform.fit_transform` protokole eklendi + `BaseTransform`.
- `tests/unit/preprocessors/` — 16 test (target-encode sızıntı: cross-fit ≠ full-train, test target'ına dokunmaz; unseen kategori; target roundtrip).

### Değişti
- `analyzers.analyze`: modality=timeseries ama geçerli `time_col` yoksa **tabloya düşer** (forecasting zaman ekseni olmadan anlamsız).
- mypy override: `sklearn.*` (py.typed yok).
- **`dynamics/` katmanı kodlandı** (ADR 0007+0010+0015): `build_plan(DataProfile, TaskSpec, RunConfig) -> AdaptivePlan`.
  - `planner.py` — `committed_ops` (yapısal drop / date_expand / encode-by-cardinality / impute / recipe refleri) + `candidate_ops` (heavy_tailed_numeric + target grupları; hedef negatifse log1p elenir) + `structure` auto (per_group_champion eşiği) + `row_policies` (intermittent_augment/filter_low_activity/coldstart_split) + `regimes` (scenario_2) + `family_policy`
  - `recipes/` — registry: `@register_recipe` + `load_recipe_paths` + entry-points (`autoragml.recipes`); isim çakışması → `RecipeError`; `validate_recipes` plan-zamanı fail-fast
  - `autoragml/transform.py` — `Transform` / `FittedTransform` protokolleri (ADR 0011) + `StatelessFitted`
- `RunConfig.dynamics` (`DynamicsConfig`); `exceptions.RecipeError`.
- `tests/unit/dynamics/` — 16 test.

### Değişti
- `01_contracts.md`: FittedTransform protokolü `autoragml/transform.py`'a taşındı (contracts pandas'a bağlanmasın diye).
- **`analyzers/` katmanı kodlandı** (ADR 0010): `analyze(dataset, config) -> (DataProfile, TaskSpec)`.
  - `modality.py` — hint/time_col/layout/datetime-dup ile tablo↔zaman serisi
  - `profiling.py` — `ColumnProfile` (raw_dtype + special_types + semantic_role + flags + duplicate_of), numpy skew/kurtosis, `TargetSummary`
  - `task_inference.py` — 7 task; hedef kuralları; `task_hint` çelişki uyarısı
  - `timeseries.py` — `pandas.infer_freq` (+ anchored-weekly fallback) + freq→periyot sözlüğü + numpy-ACF mevsimsellik + OLS-R² trend; per-series ADI/CV² + SBC sınıf (`intermittent.py`); seasonality/trend toplulaştırılmış seri üzerinde
  - `quality.py` / `leakage.py` — dataset kalite bayrakları; yumuşak sızıntı şüphesi (WARNING) + `ColumnProfile.flags`'e `leakage_suspect`
  - `_frame.py` — lazy kaynak → örneklem + düşük güven
- `RunConfig.analyzers` (`AnalyzerConfig`: `ThresholdConfig` + `TimeSeriesAnalyzerConfig` + `profiling_sample_rows`); `TimeSeriesProfile.intermittency_summary` eklendi.
- `logging.py`; `tests/unit/analyzers/` — 23 test.

### Not
- `statsmodels` yok: `stationarity_pvalue=None`; `classification_scheme=kh` → SBC + uyarı (takip).
- **`io/` katmanı kodlandı** (ADR 0009): `load_dataset(src, config) -> Dataset`.
  - `sources.py` — DataFrame/csv/tsv/parquet/dizin/`DbSource` çözümleme + boyut yoklama
  - `fingerprint.py` — **strict** sıra-bağımsız çoklu-küme hash (sum+xor+count over `hash_pandas_object`, sort'suz, streaming) + **fast** structural
  - `layout.py` — wide tespiti + `melt` (eager) + long/single_series/n-a; lazy+wide → hata
  - `lazyframe.py` — `LazyFrame` (chunk akışı); `db.py` — SQLAlchemy adaptörü (lazy import)
  - otomatik eager/lazy (`io.eager_max_bytes`, varsayılan 1 GiB)
- `logging.py` — kütüphane logger yardımcısı (kök logger'ı yapılandırmaz).
- `tests/unit/io/` — 25 test (fingerprint sıra-bağımsızlığı/hassasiyeti, layout, eager==lazy fingerprint, DB/sqlite, boş/uzantı hataları).

### Değişti
- **`requires-python` `>=3.11` → `>=3.12`** (modern tip stub'ları — numpy 2.5 PEP 695). Ruff/mypy `py312`; CI 3.12/3.13.
- `Dataset.schema` → **`Dataset.dtypes`** (pydantic `schema()` çakışması; alias yerine temiz ad).
- Dev dep: `pandas-stubs`, `sqlalchemy`. Yeni extra: `db`. mypy override: `pyarrow.*`.
- ADR 0016: config çözümleme — katmanlı merge + alan-düzeyi provenance + preset `extends` + `Settings` (`.env`) + preset konumu (pakete gömülü).
- **`config/` katmanı kodlandı**: `resolve_run_config()` → `ConfigResolution`; `merge.py` (deep-merge + provenance), `presets.py` (`extends` zinciri + döngü tespiti), `loaders.py`, `settings.py` (dotenv parser + `SecretStr` çözümü).
- `contracts/config_resolution.py` — `ConfigResolution` (`config` + `provenance` + `layers`).
- Yerleşik presetler `src/autoragml/config/_presets/`: `tabular_fast`, `timeseries_rolling`, `demandsensing` (RunConfig alan adlarıyla birebir). Wheel'e dahil (doğrulandı).
- `tests/unit/config/` — 22 test (merge, preset extends, settings, uçtan uca çözümleme).
- `exceptions.py`: `AutoRagMLError` / `ConfigError` / `PresetError`.

### Değişti
- Çekirdek bağımlılık: `pydantic-settings>=2.2`. Dev: `types-PyYAML`.
- Repo-kök `configs/presets/` kaldırıldı (pakete taşındı); `configs/` = kullanıcı proje config'leri.
- `02_layers.md` config bölümü koda göre güncel.
- **İlk implementasyon kodu**: `contracts/` pydantic v2 modelleri (enums, _base, RunConfig, Dataset, DataProfile/ColumnProfile/TimeSeriesProfile, TaskSpec, PlanContext, AdaptivePlan, Candidate, TuningResult, ValidationReport, ScoreBoard/SelectionResult, ModelBundle, RunManifest, EngineResult, RunResult).
- `tests/contract/test_contracts_smoke.py` — 12 test (doğrulama, frozen, alias round-trip, kompozisyon).
- `pydantic.mypy` plugin; `.venv` + `pip install -e .[dev]` çalışır.

### Değişti
- **`requires-python` `>=3.10` → `>=3.11`** (enum.StrEnum, tomllib). Ruff/mypy `py311`; CI matrisi 3.11/3.12/3.13.
- Uzun stub docstring satırları reflow (E501).
- ADR 0015: `RunManifest` (genişletildi: env/timeline/warnings/realized_seconds/K) + `EngineResult` + `RunResult` + `PlanContext`; uçtan uca orkestrasyon akışı.
- Açık sorular kapandı: `PlanContext` (test/full erişim yok), `candidate_ops` → HPO arama uzayı (gruplu), recipe registry katmanlı (`dynamics/recipes/` + `recipe_paths` + entry-points).
- Stub: `contracts/{run_manifest,engine_result,plan_context,run_result}.py`.

### Değişti
- **Tüm contract'lar donduruldu** (ADR 0008-0015). `01_contracts.md` + `00_overview.md` akışı hizalandı.
- Sıradaki: `contracts/*.py` pydantic v2 + `tests/contract/` — **ilk implementasyon kodu**.
- ADR 0012: Model kataloğu **YAML** (`configs/model_catalog/*.yaml`) + registry; `class_path`, `requires`, `search_space`, `fidelity`; kullanıcı override YAML ile.
- ADR 0013: HPO ensemble-öncelikli + multi-fidelity (SH/Hyperband) + nested; `hpo_level: none|light|thorough`; fold-içi iç-val early stopping.
- ADR 0014: ScoreBoard + dürüst seçim (Winning by Peeking). Seçim yalnız validation, 1-SE kuralı default, realized wall-clock + K, σ√(2lnK), MCB/Diebold-Mariano opsiyonel.
- `models/` `fine_tuners/` `scoring/` (metrics·guardrails·selection·comparison_tests) alt iskele; `configs/model_catalog/`.

### Değişti
- `01_contracts.md`: `Candidate`, `TuningResult`, `ValidationReport`, `ScoreBoard/SelectionResult`, `ModelBundle` **donduruldu**.
- `02_layers.md`: models / fine_tuners / scoring ADR 0012-0014'e göre güncel.
- ADR 0010: `analyzers` sözleşmesi + metodoloji. Güncel kaynaklarla doğrulandı (AutoGluon FeatureMetadata, SortingHat, Nixtla tsfeatures, Open Forecast/Kostenko-Hyndman, TransformedTargetRegressor). Metodoloji: betimle→karar ver→fold'da fit; "her şeyi dönüştür sonra skew ile ele" **reddedildi**.
- ADR 0011: leakage-safe by construction (Grammar of ML Workflows + LeakageDetector). fit/transform/apply ayrımı, immutable `FittedTransform`, `Frame.provenance`, 3-kategori taksonomi (overlap/preprocessing/multi_test), nested CV zorunlu.
- `analyzers/` alt iskele: `modality/profiling/task_inference/timeseries/quality/leakage`.
- `preprocessors/base.py`, `validators/` stub güncellemeleri.

### Değişti
- `01_contracts.md`: `ColumnProfile` (raw_dtype+special_types+semantic_role+flags), `TimeSeriesProfile`, `TaskSpec` (7 task), `AdaptivePlan` (committed vs candidate ops), `FittedTransform` protokolü + `Frame.provenance` **donduruldu**.
- `02_layers.md`: analyzers / preprocessors / validators ADR 0010+0011'e göre güncel.
- ADR 0004: intermittency routing → ipucu (havuz genişletir), router değil.
- **Motto** (`00_overview.md`): zaman değil sağlıklı başarı ölçü; detay kaçmaz. Tüm adımlara uygulanır.
- ADR 0009: `Dataset` + `io` sözleşmesi — strict fingerprint (örneklem yok), long kanonik format (wide → auto-melt, sınırlı model havuzu), v1 tek analitik tablo (`relations` rezerve), otomatik eager/lazy, DB opsiyonel.
- `Dataset` alan tablosu `01_contracts.md`'de **donduruldu**; stub `contracts/dataset.py`.
- ADR 0007: `dynamics` = deterministik `planner` + custom `recipes/` plug-point + v2 `synthesis`.
- `dynamics/` alt iskele: `planner.py`, `recipes/`, `synthesis.py` (docstring stub).

### Eklendi
- ADR 0008: `RunConfig` varsayılanları + çıkarım politikası (cömert bütçe / sessiz kesme yok; katmanlı split + v2 autopilot; v1'de açık target/time/group; sırlar yalnız `.env`).
- `RunConfig` alan tablosu `01_contracts.md`'de **donduruldu**.
- Stub: `config/settings.py` (pydantic-settings, `.env`), `contracts/run_config.py`.

### Değişti
- `01_contracts.md`: `RunConfig` bölümü bullet listeden tam alan tablosuna geçti.
- `docs/architecture/01_contracts.md`: `AdaptivePlan` artık recipe referansı taşıyor; 3 açık soru ADR 0007 ile kapandı.
- `docs/architecture/02_layers.md`: `dynamics/` bölümü ADR 0007'ye göre güncellendi.

## [0.0.1] - 2026-09-01

### Eklendi
- Repo iskeleti: `src/autoragml` katman ağacı (docstring stub'ları), `pyproject.toml`
  (hatchling, extras: xgboost/catboost/timeseries/hpo/tracking/report/explain/llm-*/autogluon/dev).
- Mimari dokümanları: `docs/architecture/` genel bakış + katman taslakları.
- Karar kayıtları (ADR) 0001–0006.
- CI iskeleti: 3 OS × Python 3.10–3.12 (ruff + mypy + pytest).
- `configs/presets/` hazır reçete taslakları.

### Not
- İmplementasyon kodu yok. Sözleşmeler (`contracts/`) kesinleşmeden katman kodu yazılmayacak.
