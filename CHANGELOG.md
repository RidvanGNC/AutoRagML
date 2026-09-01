# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) formatını,
sürümleme [SemVer](https://semver.org/lang/tr/)'i izler.

Her PR bu dosyaya bir satır ekler. `[Unreleased]` altında biriktirilir,
release'te tarih + sürüm ile başlığa taşınır ve git tag atılır.

## [Unreleased]

### Eklendi
- ADR 0007: `dynamics` = deterministik `planner` + custom `recipes/` plug-point + v2 `synthesis`.
- `dynamics/` alt iskele: `planner.py`, `recipes/`, `synthesis.py` (docstring stub).

### Değişti
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
