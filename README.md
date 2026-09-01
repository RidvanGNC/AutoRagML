# AutoRagML

Modalite-agnostik, **deterministik** AutoML çekirdeği. İlham kaynağı: bir talep-tahmini (demand sensing) MLOps hattının modelleme bel kemiği — model kataloğu → rolling-origin backtest → guardrail'li çok-metrikli skorlama → per-group champion.

> **Durum: v1 katman iskeleti tam (v0.0.1, pre-alpha).**
> `contracts/` → `config/` → `io/` → `analyzers/` → `dynamics/` → `preprocessors/` → `models/` →
> `validators/` → `scoring/` → `fine_tuners/` → `engines/` → `postprocessors/` → `persistence/` →
> `reporters/`+`tracking/` → `interfaces/` — tümü kodlandı (ADR 0008-0020). Sıradaki: Caruana ensemble,
> statsforecast engine derinleştirme, metin modalitesi.
>
> ```python
> from autoragml import AutoRagML
> result = AutoRagML(preset="tabular_fast").fit(df, target="sales")
> result.leaderboard(); result.predict(new_df)
> ```

## Hedef

- **pip paketi veya container** ile hızlı çıkarılabilir; kolay entegre; **tool olarak** kullanılabilir (Python API + CLI + ajan tool şeması).
- **Bulut zorunluluğu yok.** Tracking/registry/storage ve LLM sağlayıcı = opsiyonel eklenti; local dosya + no-op varsayılan.
- **Çok modalite yol haritası:** v1 tablo + zaman serisi; v1.1+ metin/görsel/ses; v2 RAG/agent üst katmanı.
- **Çapraz platform:** çekirdek saf-Python (numpy/pandas/pyarrow/scikit-learn/lightgbm/pydantic), Windows/macOS/Linux CI.

## Kilitlenen kararlar

Ayrıntı: [`docs/architecture/decisions/`](docs/architecture/decisions/)

| # | Karar |
|---|---|
| 0001 | Paket adı: **AutoRagML** (`autoragml`) |
| 0002 | v1 kapsamı: **tablo + zaman serisi** önce |
| 0003 | v1 tamamen deterministik; **RAG/agent ayrı üst katman (v2)** |
| 0004 | Engine stratejisi: **hibrit** — primitif çekirdek + opsiyonel eklenti motorlar; AutoGluon'dan Caruana ensemble selection **vendor**'lanır (atıf ile) |
| 0005 | **LLM sağlayıcı soyutlaması** — kullanıcı seçer (OpenAI/Anthropic/Bedrock/Azure/local); çekirdek bağımsız |
| 0006 | **EngineRunner tier'ları** — InProcess (varsayılan) → Subprocess (venv izolasyonu) → Container/Remote (v2+); container-mesh dayatma değil, eskalasyon |
| — | `statsforecast` **v1'e dahil** (`[timeseries]` extra) |
| — | Sürüm disiplini: **SemVer** + `CHANGELOG.md` (Keep a Changelog), tek sürüm kaynağı `src/autoragml/__init__.py` |

## Yapı

```
src/autoragml/
  contracts/     katmanlar arası tipli omurga (ÖNCE bu dondurulur)
  config/        şema + YAML + katmanlı merge + preset
  io/            yükleyiciler + lazy Dataset
  analyzers/     veriyi anlama -> DataProfile + TaskSpec
  dynamics/      veriye-özel strateji -> AdaptivePlan
  preprocessors/ fit/transform, leakage-safe
  models/        base modeller + registry (Candidate üreticileri)
  fine_tuners/   HPO + early stopping + bütçe
  validators/    split stratejileri + CV koşucu + leakage testi
  scoring/       metrikler + guardrail + seçim
  engines/       orkestrasyon (tabular, timeseries) + runners/
  postprocessors/ clip/round/calibrate/business-rule
  persistence/   ModelBundle + RunManifest + çıktı klasör düzeni
  reporters/     EDA / model card / grafik / karşılaştırma
  tracking/      opsiyonel (JSONL varsayılan, MLflow opsiyonel)
  registry/      eklenti kaydı (entry-points)
  llm/           LLMProvider soyutlaması (v2)
  interfaces/    api.py · cli.py · agent_tools.py
  _vendor/       üçüncü parti izole kod (atıf: NOTICE)
```

## Geliştirme

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -e ".[dev,timeseries]"
ruff check . && mypy && pytest
```

**macOS:** LightGBM'in çalışması için OpenMP runtime gerekir — `brew install libomp`
(kurulu değilse `lightgbm` sessizce aday havuzundan düşer, log'da uyarı çıkar).
