# ADR 0015 — RunManifest + EngineResult + engines orkestrasyon akışı

**Durum:** Kabul · 2026-09-01

Önceki açık sorular bu ADR ile kapandı: `PlanContext`, `candidate_ops` → HPO uzayı,
recipe registry katmanlı.

## Uçtan uca akış (`interfaces/api.py` → `Orchestrator.run`)

```
1  config.resolve(user_input)                    -> RunConfig
2  io.load(source)                               -> Dataset            [strict fingerprint]
3  analyzers.run(Dataset, RunConfig)             -> DataProfile, TaskSpec   [+ warnings]
4  engine seçimi: TaskSpec.modality              (v1: tabular_core | timeseries_core;
                                                  +statsforecast eğer [timeseries])
5  her engine (EngineRunner, default InProcess):
   a  dynamics.planner.build(DataProfile, TaskSpec, RunConfig)  -> AdaptivePlan
   b  registry.resolve(katalog + TaskSpec + AdaptivePlan)       -> [Candidate]
   c  validators.run  (nested CV):
        dış fold'lar:
          iç resample: fine_tuners.tune(Candidate, ...)  -> TuningResult
                       (candidate_ops seçimi arama uzayında)
          FittedTransform.fit(dış-train, PlanContext) → model fit + early stop
          → apply(dış-test) → predict → metrik
        -> Candidate başına ValidationReport   [leakage_checks: overlap/preprocessing/multi_test]
   d  scoring.build(ValidationReports, RunConfig)  -> ScoreBoard + SelectionResult
        (seçim yalnız OOF; 1-SE kuralı)
   e  şampiyon refit (tüm train) + postprocessors  -> ModelBundle
   f  -> EngineResult
6  çok engine varsa: aynı scoring kuralıyla en iyi EngineResult (engine-arası ensemble = v1.1)
7  final holdout skoru (config'te varsa): şampiyon BİR KEZ skorlanır
8  persistence.write(EngineResult, RunManifest)
     -> outputs/<DDMMYYYY>_<proje>_outputs/<run_id>/{evaluation,models,reports,config_snapshot}/
9  reporters.build(...)  -> EDA raporu, model card, karşılaştırma, grafikler
10 return RunResult   (facade: .leaderboard() .predict() .explain() .champion .manifest)
```

Yan etki yalnız 8–9 (persistence, reporters) ve `tracking`. 1–7 saf dönüşüm zinciri.

## RunManifest  (`persistence/` üretir)
- `run_id` (zaman-tabanlı, sıralanabilir), `created_at`, `project_name`, `autoragml_version`
- `input_fingerprint` (strict, ADR 0009), `config_snapshot` (RunConfig, sırlar maskeli)
- `env`: python, platform/OS, anahtar paket sürümleri, git commit (repo ise)
- `data_snapshot`: n_rows/n_cols, tarih aralığı, target özeti, `layout`
- `seed`
- `timeline[]`: aşama başına `{stage, start, end, status}` — "hangi faz kırıldı" (DemandSensing log deseni)
- `artifacts`: `{ad: yol}` tüm çıktılar
- `champion_ref`: `ModelBundle` işaretçisi
- `realized_seconds`, `n_candidates` (ScoreBoard'dan, ADR 0014)
- `warnings[]`: tüm WARNING'ler (düşük güven çıkarım, leakage şüphesi, wide-format
  degradasyon, runtime projeksiyon)

## EngineResult  (engine → orchestrator)
- `engine_key`
- `scoreboard: ScoreBoard`, `selection: SelectionResult`
- `champion: ModelBundle`
- `validation_reports_ref`
- `data_profile`, `task_spec`, `adaptive_plan` (engine'in kararları)
- `status`: `SUCCESS | PARTIAL | FAILED`, `messages[]`

## RunResult  (`interfaces/api.py` — kullanıcıya dönen)
`EngineResult` + `RunManifest` sarımı + kolaylık: `.leaderboard()`, `.predict(X)`,
`.explain()`, `.champion`, `.manifest`, `.reports_dir`.

## PlanContext  (`FittedTransform.fit` ikinci arg)
Frozen; salt-okunur. `target, group_col, time_col, task, column_roles, fold_id,
train_span, seed, provenance="train"`. Test/full veriye, split nesnesine erişim **yok**.

## candidate_ops → HPO arama uzayı
`{log1p|yeo_johnson|quantile|none}` = kategorik hiperparametre; `fine_tuners` model
paramlarıyla birlikte iç resample'da arar. `dynamics/planner` kolonları muamele
sınıfına göre **gruplar** (patlama önleme); HPO değeri seçer. `hpo_level: none` →
`family_policy` sabit varsayılan.

## Recipe registry (katmanlı)
`dynamics/recipes/` paketi (`@register_recipe`) · `RunConfig.recipe_paths` (proje-yerel
dizin) · entry-points (`autoragml.recipes`). Tek registry; isim çakışması → açık hata.

## Sonuç
- `contracts`: `RunManifest` (genişletildi), `EngineResult`, `RunResult`, `PlanContext` eklenir.
- Kodlama başlar: `contracts/` pydantic modelleri + contract testleri.
