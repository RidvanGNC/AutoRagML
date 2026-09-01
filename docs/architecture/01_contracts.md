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

### AdaptivePlan  (`dynamics/` üretir)
- `column_ops`: kolon → [op] (target_encode, hashing, winsorize, log1p, date_expand,
  text_embed, drop, impute:<strategy>)
- `row_policies`: [filter_low_activity, coldstart_split, intermittent_route:<pipeline>]
- `structure`: `{ pooled | per_group_champion }`, `target_transform?`
- `regimes?`: senaryo/regime tanımları (fold-güvenli fit edilir)

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

## Açık sorular

- `dynamics` sınırı: strateji/policy katmanı mı, yoksa `models/` içine gömülü kod mu?
- `AdaptivePlan` ne kadar deklaratif olmalı (serialize edilebilir sözlük) vs. kod nesnesi?
- Fold-güvenli regime fit → `AdaptivePlan` mı taşır, `validators` mı yönetir?
- Karışık modalite (v1.1+) `TaskSpec.modality = mixed` → çok engine + füzyon nasıl?
