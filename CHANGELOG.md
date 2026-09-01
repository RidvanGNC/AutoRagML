# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) formatını,
sürümleme [SemVer](https://semver.org/lang/tr/)'i izler.

Her PR bu dosyaya bir satır ekler. `[Unreleased]` altında biriktirilir,
release'te tarih + sürüm ile başlığa taşınır ve git tag atılır.

## [Unreleased]

### Eklendi
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
