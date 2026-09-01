# ADR 0018 — persistence: bundle serialize + RunManifest + çıktı klasör düzeni

**Durum:** Kabul · 2026-09-01

Kaynak: scikit-learn model persistence rehberi (joblib önerisi + sürüm-uyumu uyarısı);
DemandSensing `run_id = %Y%m%dT%H%M%SZ` + `outputs/<DDMMYYYY>_<proje>_outputs/` deseni.

## İlke

`persistence`, **yan etkisi olan** iki katmandan biri (diğeri `tracking`). Diğer tüm
katmanlar saf; buradan itibaren diske yazılır. Sorumluluk: (1) fitted `ModelBundle`'ı
diske/diskten taşımak, (2) `RunManifest` kurup yazmak, (3) koşum çıktı klasör düzenini
oluşturmak, (4) contract nesnelerini JSON'a dökmek.

## Kurallar

### 1. Bundle serialize formatı: **joblib**
`ModelBundle.pipeline` = `FittedModelPipeline` (fitted sklearn/lightgbm estimator +
`FittedTransform` zinciri + `FittedTargetTransform` + `FittedPostprocessor` +
opsiyonel `functools.partial` `pre_transform`). scikit-learn'in resmi önerisi joblib
(numpy dizilerini verimli pickle'lar). `compress=3` varsayılan.

- **Tek dosya:** `models/champion.joblib` — kendine yeterli:
  `{format_version, autoragml_version, saved_env, metadata(dict), metrics_oof,
  metrics_holdout, artifact_path, pipeline(canlı nesne)}`.
- **İnsan-okur ayna:** `models/champion_metadata.json` — `BundleMetadata` + metrikler
  (yüklemede GEREKMEZ; denetim/tooling için).

`FittedModelPipeline` / `FittedPostprocessor` / `FittedTargetTransform` `__slots__`
sınıflarıdır → Python varsayılan pickle protokolü yeterli (tüm slot değerleri picklable).

### 2. Güvenlik + sürüm uyumu
- **joblib.load pickle çalıştırır → kod yürütür.** `load_bundle` docstring'i +
  ilk kullanımda WARNING: "yalnız güvendiğiniz bundle'ları yükleyin".
- `saved_env` içinde `python`, `scikit-learn`, `lightgbm`, `numpy`, `scipy` sürümleri.
  `load_bundle` `format_version` uyuşmazlığında → `PersistenceError`; `scikit-learn`
  major/minor uyuşmazlığında → WARNING (sklearn pickle'ları sürümler arası garanti değil).
- `business_rule` hook (ADR 0017) importable/modül-düzeyi callable değilse pickle
  edilemez → `interfaces` enjekte ederken lambda kullanırsa persist edilemez (belgelenir).

### 3. run_id ve klasör düzeni
```
<output_dir>/<DDMMYYYY>_<project_name>_outputs/<run_id>/
  manifest.json                 RunManifest (sırlar maskeli — zaten *_env adları)
  models/
    champion.joblib
    champion_metadata.json
  evaluation/
    scoreboard.json  selection.json  comparison_tests.json
    validation_reports.json         (aday başına fold metrikleri; OOF dizileri HARİÇ)
    holdout_metrics.json            (nihai holdout — bir kez; yoksa yazılmaz)
  reports/                          (reporters katmanı doldurur; persistence yalnız oluşturur)
  config_snapshot/
    run_config.json                 config.model_dump(mode="json")
    config_resolution.json          provenance + layers (varsa)
```
- `run_id = <UTC %Y%m%dT%H%M%SZ>`; aynı saniyede çakışma → `-01`, `-02` soneki
  (`create_run_dir` sessizce üzerine YAZMAZ — dolu dizin + `exist_ok=False` → hata).
- `<DDMMYYYY>` = UTC gün-ay-yıl (kullanıcı spec'i; günlük gruplama).
- **`.env` asla kopyalanmaz.** `config_snapshot` yalnız `RunConfig` (sır taşımaz).

### 4. RunManifest kurulumu
`build_manifest(config, dataset, engine_result, *, timeline, warnings, realized_seconds,
started_at)`:
- `run_id` (verilmezse üretilir), `created_at` (ISO 8601 UTC), `project_name`,
  `autoragml_version` (`autoragml.__version__`).
- `input_fingerprint = dataset.fingerprint` (STRICT).
- `config_snapshot = config.model_dump(mode="json")`.
- `env`: `platform.python_version()`, `platform.platform()`, `sys.platform`,
  sabit paket listesinin `importlib.metadata.version`'ları; `git_commit` best-effort
  (`autoragml` kaynağı bir git repo ise) yoksa `None`.
- `data_snapshot`: satır/sütun, TS ise tarih aralığı, hedef özeti, layout.
- `timeline`: orchestrator'ın topladığı `TimelineEntry` listesi (ADR 0020).
- `champion_ref = "models/champion.joblib"`, `artifacts` = yazılan dosyaların
  run-dir'e göre yolları.

### 5. Determinizm (motto)
- Tüm JSON dump'ları `sort_keys=True`, `ensure_ascii=False`, `indent=2`, sonda newline.
- `create_run_dir` çakışma çözümü deterministik (artan sonek).
- persistence hiçbir şeyi sessizce ezmez; var olan dolu run-dir → `PersistenceError`.

### 6. Kapsam dışı (sonraki ADR'ler)
- Nihai holdout'un **bir kez** skorlanması → orchestrator akışı (ADR 0020).
- `reports/` içeriği → `reporters` (ADR 0019).
- JSONL deney takibi → `tracking` (ADR 0019).

## API (`persistence/`)

```
paths.py       RunPaths (dataclass: root/models/evaluation/reports/config_snapshot)
               create_run_dir(config, *, run_id=None, exist_ok=False) -> RunPaths
               make_run_id(now=None) -> str
bundle.py      save_bundle(bundle, path, *, compress=3) -> Path
               load_bundle(path) -> ModelBundle
manifest.py    build_manifest(...) -> RunManifest
               write_manifest(manifest, run_dir) -> Path
dump.py        write_json(obj_or_model, path) -> Path   (pydantic → model_dump(mode="json"))
               persist_evaluation(engine_result, paths, *, holdout=None) -> dict[str,str]
__init__.py    yukarıdakileri re-export + PersistenceError
```

`exceptions.py`'a `PersistenceError(AutoRagMLError)` eklenir.

## Sonuç

- `ModelBundle` diske joblib tek dosya + JSON ayna; `load_bundle` sürüm kontrollü.
- `RunManifest` reprodüksiyon için yeterli (fingerprint + config + env + seed).
- Çıktı düzeni sabit ve çakışmaya dayanıklı; `.env` asla sızmaz.
- `interfaces/Orchestrator` (ADR 0020) bu fonksiyonları sırayla çağırır.
