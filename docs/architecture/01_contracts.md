# Sözleşmeler (contracts/) — TASLAK

> Bu dosya tartışma alanı. Alanlar kesinleşince `src/autoragml/contracts/*.py`
> içinde pydantic v2 modeli olarak dondurulur ve **contract testi** yazılır.
> Kesinleşmeden hiçbir katman kodu yazılmaz.

## Neden önce bu

Her katman "girdi contract → çıktı contract" saf dönüşümü. Sözleşmeler stabil
olmadan katmanlar birbirine sızar. Sıra: contracts → config → io → analyzers → ...

## Nesneler (ilk taslak alanlar)

### RunConfig  (`config/` üretir)
- `task_hint`, `modality_hint` (opsiyonel — analyzers doğrular/doldurur)
- `target`, `group_col`, `time_col` (opsiyonel)
- `budget`: `{ mode: time|trials, value, per_fold_seconds }`
- `split_policy`: `{ kind: holdout|kfold|group_kfold|rolling_origin|fixed_window, ... }`
- `primary_metric`, `metric_by_class` (opsiyonel)
- `output_dir`, `project_name`, `seed`
- `engines`: aktif engine + opsiyonel override
- `tracking`: `{ backend: none|jsonl|mlflow, ... }`

### Dataset  (`io/` üretir)
- `handle`: lazy referans (df | dosya yolu | klasör)
- `schema`: kolon → dtype
- `n_rows`, `modparts`: `{ tabular?, image_dir?, audio_dir?, text_col? }`
- `fingerprint`: içerik hash (RunManifest'e girer)

### DataProfile  (`analyzers/` üretir)
- `columns[]`: `{ name, dtype, role (numeric|categorical|datetime|text|id|target),
  cardinality, missing_ratio, skew, n_unique }`
- `target_summary`: dağılım, dengesizlik, sınıf sayısı
- `timeseries?`: `{ freq, gaps, seasonality[], stationarity, adi, cv2, intermittent_class }`
- `leakage_suspects[]`: hedefle ~mükemmel ilişkili kolonlar
- `quality_flags[]`

### TaskSpec  (`analyzers/` üretir)
- `task`: classification | regression | forecasting | (v1.1+ ...)
- `modality`: tabular | timeseries | (v1.1+ text|image|audio|mixed)
- `targets[]`, `horizon?`, `group_col?`, `time_col?`

### AdaptivePlan  (`dynamics/planner.py` üretir — deterministik, ADR 0007)
Deklaratif ve serialize edilebilir (sözlük). Kod taşımaz; **referans** taşır.
- `column_ops`: kolon → [op]. Op iki tür:
  - **katalog op** (sabit): `target_encode`, `hashing`, `winsorize`, `log1p`,
    `date_expand`, `text_embed`, `drop`, `impute:<strategy>`
  - **recipe referansı**: `recipe:"<registry_adı>"` → `dynamics/recipes/` içinde kayıtlı,
    `preprocessors` arayüzüne uyan custom transform (v1: elle yazılır; v2: `synthesis.py` üretir)
- `row_policies`: [`filter_low_activity`, `coldstart_split`, `intermittent_route:<pipeline>`]
- `structure`: `{ pooled | per_group_champion }`, `target_transform?`
- `regimes?`: senaryo/regime tanımları — **fit'i `validators` yönetir** (fold-güvenli),
  plan yalnız tanımı taşır
- `recipes_used[]`: bu planın referansladığı recipe adları (RunManifest'e girer)

### Candidate (ModelSpec)  (`models/` + `registry/` üretir)
- `key`, `factory(params) -> estimator`, `param_space`, `family`
- `modalities[]`, `predict_kind` (point|proba|quantile)
- `supports_early_stopping`, `wrap` (imputer/scaler gerekli mi)

### TuningResult  (`fine_tuners/` üretir)
- `best_params`, `trials[]`, `spent_budget`, `early_stopped`

### ValidationReport  (`validators/` üretir)
- `folds[]`: `{ fold_id, train_span, test_span, predictions_ref, metrics }`
- `leakage`: `{ status: PASS|FAIL, violations[] }`
- `oof_predictions_ref`

### ScoreBoard / SelectionResult  (`scoring/` üretir)
- `rows[]`: `{ model_key, scenario, metrics_mean, guardrail_flags, is_quarantined,
  selection_eligible, class_weighted_score }`
- `champion`: `{ model_key, scenario, reason }`
- `promotion`: `{ passed, reasons[] }`

### ModelBundle  (`persistence/` üretir)
- `pipeline` (fitted: preprocessors + estimator + postprocessors)
- `metadata`: feature list + hash, task_spec, adaptive_plan özeti, config snapshot
- `champion_info`, `metrics`

### RunManifest  (`persistence/` üretir)
- `run_id`, `created_at`, `project_name`
- `input_fingerprint`, `config_snapshot`
- `env`: paket sürümleri, python, platform, git commit
- `artifacts`: tüm çıktı yolları
- `autoragml_version`

### Recipe (custom transform)  (`dynamics/recipes/` — ADR 0007)
`preprocessors` ile aynı arayüz; ayrı contract nesnesi değil, bir **protokol**:
- `fit(train_df, plan_ctx) -> self`
- `transform(X) -> X'`
- `get_params()` / serialize (joblib) — `ModelBundle`'a girer
- registry'ye isimle kayıtlı; `AdaptivePlan.column_ops` içinden `recipe:"<ad>"` ile çağrılır
- v1: insan yazar · v2: `dynamics/synthesis.py` (LLM) üretir, runner'da doğrular, kaydeder

## Açık sorular

- ~~`dynamics` sınırı~~ → **çözüldü: ADR 0007** (planner + recipes + v2 synthesis)
- ~~`AdaptivePlan` deklaratif mi kod mu~~ → **çözüldü: deklaratif sözlük, recipe referansı taşır**
- ~~Fold-güvenli regime fit kim yönetir~~ → **çözüldü: `validators` fit eder, plan tanımı taşır**
- `plan_ctx`: recipe'e `fit` sırasında hangi bağlam verilir (group_col, time_col, target)?
- Recipe registry: yalnız `dynamics/recipes/` klasörü mü, yoksa entry-points ile dış paket de mi?
- Karışık modalite (v1.1+) `TaskSpec.modality = mixed` → çok engine + füzyon nasıl?
