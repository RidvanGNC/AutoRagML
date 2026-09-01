# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) formatını,
sürümleme [SemVer](https://semver.org/lang/tr/)'i izler.

Her PR bu dosyaya bir satır ekler. `[Unreleased]` altında biriktirilir,
release'te tarih + sürüm ile başlığa taşınır ve git tag atılır.

## [Unreleased]

### Eklendi
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
