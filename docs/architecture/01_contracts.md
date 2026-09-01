# Sözleşmeler (contracts/) — TASLAK

> Bu dosya tartışma alanı. Alanlar kesinleşince `src/autoragml/contracts/*.py`
> içinde pydantic v2 modeli olarak dondurulur ve **contract testi** yazılır.
> Kesinleşmeden hiçbir katman kodu yazılmaz.

## Neden önce bu

Her katman "girdi contract → çıktı contract" saf dönüşümü. Sözleşmeler stabil
olmadan katmanlar birbirine sızar. Sıra: contracts → config → io → analyzers → ...

## Nesneler (ilk taslak alanlar)

### RunConfig  (`config/` üretir · alanlar DONDU — ADR 0008)

Serialize edilebilir (pydantic v2). **Sır taşımaz** — yalnız `*_env` adları.

| Alan | Tip | Default | Not |
|---|---|---|---|
| `target` | `str` | — | **Zorunlu.** v1'de çıkarım yok (ADR 0008/3) |
| `time_col` | `str \| None` | `None` | forecasting task'inde **zorunlu** |
| `group_col` | `str \| None` | `None` | yoksa pooled; varsa per-group champion mümkün |
| `task_hint` | `str \| None` | `None` | `analyzers` doğrular; çelişkide hata |
| `modality_hint` | `str \| None` | `None` | v1: `tabular` \| `timeseries` |
| `project_name` | `str` | `"autoragml"` | çıktı klasör adında kullanılır |
| `output_dir` | `path` | `"outputs"` | `persistence` alt yapı kurar |
| `seed` | `int` | `42` | |
| `autopilot` | `bool` | `False` | **v1'de sabit false.** v2: analyzers her şeye karar verir |
| **`budget`** | obj | ↓ | ADR 0008/1 — sessiz kesme yok |
| `budget.total_max_seconds` | `int \| None` | `None` | global wall-clock tavan |
| `budget.per_model_max_seconds` | `int \| None` | `None` | `None` → trial sayısı yönetir |
| `budget.max_trials_per_model` | `int` | `15` | |
| `budget.min_trials_per_model` | `int` | `3` | guardrail |
| `budget.per_fold_timeout_seconds` | `int \| None` | `None` | `None` → fold iptali yok |
| `budget.runtime_projection_warn_seconds` | `int` | `7200` | ilk fold sonrası tahmini toplam bunu aşarsa **uyar, devam et** |
| **`split_policy`** | obj \| None | `None` | **kısmi** — verilen alan kazanır, gerisi `analyzers.SplitRecommendation` |
| `split_policy.kind` | enum \| None | `None` | holdout \| kfold \| stratified_kfold \| group_kfold \| time_series \| rolling_origin \| fixed_window |
| `split_policy.*` | — | — | kind'e özel: `n_folds`, `horizon`, `step`, `min_train_periods`, `test_size`... |
| `primary_metric` | `str \| None` | `None` | `None` → task'e göre varsayılan |
| `metric_by_class` | `dict \| None` | `None` | intermittent routing (Smooth→wmape, ...) |
| `scenarios` | `list[str]` | `["scenario_1"]` | TS engine; `scenario_2` opt-in |
| `hpo_level` | `HpoLevel` | `"light"` | `none` \| `light` \| `thorough` (ADR 0013) |
| `hpo_backend` | `HpoBackend` | `"random_search"` | `optuna` opsiyonel (`[hpo]`) |
| `guardrails` | `GuardrailConfig` | `{enabled: true, prediction_scale_multiplier_max: 100}` | metrik tavanları + prediction eşikleri + `model_scenario_blocklist` (ADR 0014) |
| `promotion` | `PromotionConfig` | `{smape_max: 35, min_folds: 2, require_leakage_pass: true}` | mutlak eşik kapısı (ADR 0014) — bilgilendirir, engellemez |
| `engines` | obj \| None | `None` | aktif engine + override; `None` → analyzers seçer |
| `analyzers` | `AnalyzerConfig` | varsayılan | ADR 0010 eşikleri: `thresholds` + `timeseries` + `profiling_sample_rows` |
| `dynamics` | `DynamicsConfig` | varsayılan | ADR 0007/0015: `structure`, per-group eşikleri, transform seçenekleri, kodlama, `recipes[]`, `drop_leakage_suspects` |
| `validation` | `ValidationConfig` | varsayılan | ADR 0010/6+0013: `min_rows_for_cv`, `default_kfold_splits`, `default_rolling_folds`, `holdout_fraction`, `early_stopping_fraction` |
| `tracking.backend` | enum | `"jsonl"` | none \| jsonl \| mlflow |
| `tracking.uri_env` | `str \| None` | `None` | mlflow için env-var **adı** |
| `llm` | obj \| None | `None` | v2. `{provider, model, endpoint_env, api_key_env}` — sır yok |

**Ayrı katman — `Settings`** (`config/settings.py`, pydantic-settings): `.env`'den okur,
`SecretStr` alanları, **asla serialize edilmez**. `RunConfig.*_env` adları burada çözülür.

### Dataset  (`io/` üretir · alanlar DONDU — ADR 0009)

| Alan | Tip | Not |
|---|---|---|
| `source` | obj | `{kind: dataframe\|csv\|parquet\|csv_dir\|parquet_dir\|db, ref}` |
| `dtypes` | `dict[str, str]` | kolon → ham dtype string (`schema` yerine — pydantic çakışması) |
| `shape` | `(n_rows, n_cols)` | `n_rows` **her zaman tam sayım** (lazy'de bile — tahmin yok) |
| `materialization` | `"eager" \| "lazy"` | otomatik: boyut `RunConfig.io.eager_max_bytes` eşiği |
| `handle` | ref | eager: DataFrame · lazy: pyarrow dataset / chunk iterator |
| `layout` | `"long" \| "wide_converted" \| "single_series" \| "n/a"` | TS için; wide → auto-melt (loglanır) |
| `fingerprint` | `str` (SHA256) | **STRICT** — kanonik form üzerinden tüm hücreler; tek streaming geçiş |
| `fingerprint_spec` | `str` | nasıl hesaplandığı (sıralama anahtarı, şema normalizasyonu) |
| `fingerprint_fast` | `str \| None` | structural — yalnız hızlı drift sinyali, **kimlik değil** |
| `modparts` | obj | v1: `{tabular}`. v1.1+: `image_dir`, `audio_dir`, `text_col` |
| `relations` | `None` | **REZERVE** (ADR 0009/3) — çok-tablo join spec; v1'de hep `None` |

Wide girdi → yalnız hedef geçmişi; exogenous feature yok, aday model havuzu
baseline + univariate reduction ile **sınırlı** (`analyzers` `layout`'a bakıp kısıtlar).

### ColumnProfile  (`analyzers/profiling.py` — alanlar DONDU, ADR 0010)

AutoGluon `FeatureMetadata` deseniyle hizalı: raw dtype + special types ayrımı.

| Alan | Tip | Not |
|---|---|---|
| `name` | `str` | |
| `raw_dtype` | enum | `int \| float \| category \| object \| datetime \| bool` |
| `special_types` | `set[str]` | `text · text_ngram · datetime · embedded_number · boolean` (0..n) |
| `semantic_role` | enum | `target · id · categorical · numeric_continuous · numeric_discrete · boolean · datetime · text · constant · unknown` |
| `flags` | `set[str]` | `high_cardinality · near_constant · high_missing · all_missing · skewed · heavy_tailed · zero_inflated · datetime_like_string · numeric_like_string · duplicate_of:<col> · monotonic · leakage_suspect` |
| `n_unique` / `missing_ratio` | | |
| `stats` | obj | numeric: `min/max/mean/std/skew/kurtosis`; kategorik: `top_values`; `sample_values` |
| `confidence` | `float` | düşük → WARNING (akış durmaz) |
| `inference_source` | enum | `rule \| user_override \| ml_detector(gelecek)` |

### DataProfile  (`analyzers/` üretir)
- `columns: list[ColumnProfile]`
- `n_rows`, `n_cols`
- `target_profile`: hedefin `ColumnProfile`'ı + `target_summary` (`n_classes`,
  sınıf dengesi, dağılım, `zero_ratio`)
- `timeseries: TimeSeriesProfile | None`
- `quality_flags[]`: dataset düzeyi (`duplicate_rows`, `constant_target`, `tiny_data`,
  `severe_imbalance`, ...)
- `leakage_suspects[]`: `{ column, reason, confidence }` (ADR 0011/5 — yumuşak, WARNING)
- `confidence`: genel çıkarım güveni

### TimeSeriesProfile  (`analyzers/timeseries.py` — ADR 0010)
- `freq` (`pandas.infer_freq` sıralı per-series; düzensizse modal-gap) · `freq_confidence`
- `span` · `regular: bool` · `gaps[]` (grup bazında eksik dönemler)
- `seasonality[]`: `[{period, strength}]` — freq→periyot sözlüğü + ACF/STL doğrulama
- `trend_strength` · `stationarity` (ADF p)
- `per_series[]`: grup bazında `{ n_obs, n_nonzero, zero_ratio, adi, cv2, history_weeks,
  intermittency_class, intermittency_class_recent, class_changed_over_time }`
- `classification_scheme`: `"sbc" | "kh"` (default `sbc`)
- `per_series_detail`: `full | sampled | summary_only` (default `full`)
- **Not:** intermittency sınıfı router **değil** — aday havuzunu genişletir + birincil
  metriği etkiler; nihai seçim holdout (ADR 0010).

### TaskSpec  (`analyzers/` üretir)
- `task`: `regression | binary_classification | multiclass_classification |
  multilabel_classification | quantile_regression | ordinal_regression | forecasting`
- `modality`: `tabular | timeseries` (v1.1+ `text|image|audio|mixed`)
- `targets[]`, `horizon?`, `group_col?`, `time_col?`, `quantiles?`
- `inference_confidence` + `inference_warnings[]`

### AdaptivePlan  (`dynamics/planner.py` üretir — deterministik, ADR 0007 + 0010)
Deklaratif, serialize edilebilir. Kod taşımaz; **referans** taşır.

- **`committed_ops`**: kolon → [op], **her zaman uygulanır** (yapısal):
  `drop` (constant/duplicate/all-null/monotonic-id), `date_expand`, kategorik `encode`
- **`candidate_ops`**: muamele-sınıfı → [op seçenekleri], **HPO uzayında seçilir** (ADR 0015):
  `log1p | yeo_johnson | quantile | none`; hedef dönüşümü `TransformedTargetRegressor` sarımı.
  `planner` kolonları sınıfa göre **gruplar** (patlama önleme); HPO değeri seçer;
  `hpo_level: none` → `family_policy` sabit varsayılan
- **recipe referansı**: `committed_ops`/`candidate_ops` içinde `recipe:"<ad>"` →
  `dynamics/recipes/` kayıtlı custom transform (`FittedTransform` protokolü, ADR 0011)
- `row_policies`: `filter_low_activity`, `coldstart_split`, `intermittent_augment:<class>`
  (havuz genişletir — route/kısıtla değil)
- `structure`: `pooled | per_group_champion`
- `regimes?`: tanım taşır; **fit'i `validators` yönetir** (fold-güvenli, ADR 0011/2)
- `family_policy`: model ailesine göre op yoğunluğu (ağaç → minimal, lineer → kapsamlı)
- `recipes_used[]` → RunManifest'e girer

### Candidate  (`models/` + `registry/` üretir — katalog YAML'dan, ADR 0012)
- `key`, `name`, `family`
- `factory(task, params) -> estimator` (`class_path`'ten; task→Regressor/Classifier map)
- `modalities[]`, `tasks[]`, `predict_kind[]` (point|proba|quantile)
- `search_space` (HPO uzayı — katalog YAML), `fidelity` (multi-fidelity ekseni, ADR 0013)
- `supports_early_stopping`, `early_stopping_rounds`
- `requires[]` (pip extra — yoksa entry atlanır), `wrap` (imputer/scaler)
- `source`: `builtin_catalog | user_catalog | entry_point`

### TuningResult  (`fine_tuners/` üretir — ADR 0013)
- `best_params`, `trials[]`, `spent_budget`, `realized_seconds`
- `early_stopped`, `best_iteration_per_fold[]`, `fidelity_schedule`
- `backend`: `random_search | optuna | flaml`, `hpo_level`: `none | light | thorough`

### ValidationReport  (`validators/` üretir)
- `folds[]`: `{ fold_id, train_span, test_span, predictions_ref, metrics }`
- `leakage`: `{ status: PASS|FAIL, violations[] }` — 3 kategori: `overlap | preprocessing |
  multi_test` (ADR 0011/5)
- `oof_predictions_ref`
- `nested`: HPO / `candidate_ops` seçimi iç resample'da yapıldı mı (ADR 0010/6);
  dış fold yalnız skorlar

### ScoreBoard / SelectionResult  (`scoring/` üretir — ADR 0014, dürüst seçim)
- `rows[]`: `{ model_key, scenario, oof_metric_mean, oof_metric_se, all_metrics_mean,
  guardrail_flags, is_quarantined, selection_eligible, class_weighted_score,
  realized_seconds, n_trials, best_iteration }`
- `noise_floor`: birincil metrik SE (fold'lar arası)
- `selection_rule`: `best | one_std_err` (**default `one_std_err`** — 1 SE içindeki en
  basit/ucuz/sağlam model)
- `champion`: `{ model_key, scenario, reason, within_1se[], statistical_ties[] }`
- `selection_bias_bound`: σ·√(2 ln K), `n_candidates`: K
- `comparison_tests?`: `{ mcb_ranks, dm_pvalues }` (forecasting, opsiyonel)
- `promotion`: `{ passed, reasons[] }` (mutlak eşikler — DemandSensing)
- **Seçim yalnız OOF/validation'da**; test'e tek dokunuş `engines`'te enforce, `validators`
  `multi_test` ihlalini yakalar (ADR 0014/1)

### ModelBundle  (`persistence/` üretir)
- `pipeline` (fitted: `FittedTransform`'lar + estimator + postprocessors) — tüm train'de refit
- `metadata`: feature list + hash, `task_spec`, `adaptive_plan` özeti, config snapshot,
  `best_iteration` (ES modelleri için sabit), `provenance_fitted_on`
- `champion_info`, `metrics` (OOF + final holdout — bir kez)

### RunManifest  (`persistence/` üretir — ADR 0015)
- `run_id` (zaman-tabanlı, sıralanabilir), `created_at`, `project_name`, `autoragml_version`
- `input_fingerprint` (strict), `config_snapshot` (RunConfig, sırlar maskeli)
- `env`: python, platform/OS, anahtar paket sürümleri, git commit (repo ise)
- `data_snapshot`: n_rows/n_cols, tarih aralığı, target özeti, `layout`
- `seed`
- `timeline[]`: `{ stage, start, end, status }` — hangi faz kırıldı
- `artifacts`: `{ ad: yol }`
- `champion_ref`: `ModelBundle` işaretçisi
- `realized_seconds`, `n_candidates`
- `warnings[]`: tüm WARNING'ler (düşük güven, leakage şüphesi, wide degradasyon, runtime projeksiyon)

### EngineResult  (engine → `Orchestrator` — ADR 0015)
- `engine_key`
- `scoreboard: ScoreBoard`, `selection: SelectionResult`, `champion: ModelBundle`
- `validation_reports_ref`
- `data_profile`, `task_spec`, `adaptive_plan` (engine kararları)
- `status`: `SUCCESS | PARTIAL | FAILED`, `messages[]`

### RunResult  (`interfaces/api.py` — kullanıcıya dönen — ADR 0015)
`EngineResult` + `RunManifest` sarımı + kolaylık metotları:
`.leaderboard()`, `.predict(X)`, `.explain()`, `.champion`, `.manifest`, `.reports_dir`

### PlanContext  (`FittedTransform.fit` 2. arg — ADR 0011 + 0015)
Frozen, salt-okunur:
`target · group_col · time_col · task · column_roles (semantic_role dict) · fold_id ·
train_span · seed · provenance="train"`.
Test/full veriye, split nesnesine erişim **yok**.

### FittedTransform protokolü  (`autoragml/transform.py` — ADR 0011)
`Transform` / `FittedTransform` protokolleri (`typing.Protocol`). `preprocessors` katalog
transformları ve `dynamics/recipes` custom transformlar buna uyar. Sızıntı yapısal olarak
engellenir. Üç ayrı ilkel:
- **stateless** `transform(X) -> X'` — parametre öğrenmez
- **`fit(train_frame, ctx: PlanContext) -> FittedTransform`** — yalnız `provenance == "train"`
  frame'den öğrenir; **immutable** nesne döndürür; split'ler arası yeniden kullanılamaz
- **`apply(X) -> X'`** — öğrenilmiş parametreyi uygular, saf (ek bağlam almaz)
- `provenance_fitted_on` kaydı; `serialize()` → `ModelBundle`
- `fit`'i **yalnız `validators`** çağırır (fold içinde); kullanıcı/recipe kodu split
  sınırını görmez
- registry: `dynamics/recipes/` (`@register_recipe`) · `RunConfig.recipe_paths` (proje-yerel
  dizin) · entry-points (`autoragml.recipes`). Tek registry; isim çakışması → **açık hata**
- `AdaptivePlan` içinden `recipe:"<ad>"` ile çağrılır
- v1: insan yazar · v2: `dynamics/synthesis.py` (LLM) üretir → runner'da doğrular → kaydeder

### Frame provenance  (tüm katmanlar)
Her veri frame'i `provenance: "train" | "val" | "test" | "full"` taşır.
`analyzers` / `dynamics.planner` `"full"` görür (fit etmez). `validators` fold'da
`"train"/"test"` üretir. `test`/`val`'dan fit edilmiş `FittedTransform`'un başka
partition'a `apply`'ı → hata (ADR 0011/3-4).

## Açık sorular

- ~~`dynamics` sınırı~~ → **çözüldü: ADR 0007** (planner + recipes + v2 synthesis)
- ~~`AdaptivePlan` deklaratif mi kod mu~~ → **çözüldü: deklaratif sözlük, recipe referansı taşır**
- ~~Fold-güvenli regime fit kim yönetir~~ → **çözüldü: `validators` fit eder, plan tanımı taşır**
- ~~`dynamics` metodoloji (skew ile eleme?)~~ → **çözüldü: ADR 0010** (betimle→karar→fold'da fit; committed vs candidate ops)
- ~~intermittency routing~~ → **çözüldü: ADR 0010** (ipucu — havuz genişletir + metrik; router değil)
- ~~sızıntı önleme~~ → **çözüldü: ADR 0011** (fit/transform/apply, immutable, provenance, 3-kategori)
- ~~`plan_ctx` içeriği~~ → **çözüldü: ADR 0015** (`PlanContext` frozen, test/full erişim yok)
- ~~recipe registry kapsamı~~ → **çözüldü: ADR 0015** (paket + `recipe_paths` + entry-points; çakışma→hata)
- ~~`candidate_ops` seçim algoritması~~ → **çözüldü: ADR 0015** (HPO arama uzayında, gruplu)
- Karışık modalite (v1.1+) `TaskSpec.modality = mixed` → çok engine + füzyon nasıl?
- Engine-arası ensemble (v1.1) — birden çok `EngineResult`'ı birleştirme

---

## Durum: sözleşmeler DONDU → kodlama başlıyor

`RunConfig · Dataset · ColumnProfile/DataProfile/TimeSeriesProfile · TaskSpec ·
AdaptivePlan · PlanContext · FittedTransform · Candidate · TuningResult ·
ValidationReport · ScoreBoard/SelectionResult · ModelBundle · EngineResult ·
RunManifest · RunResult` — hepsi donduruldu.

Sıradaki: `src/autoragml/contracts/*.py` pydantic v2 modelleri + `tests/contract/`.
