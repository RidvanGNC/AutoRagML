# AutoRagML — Mimari Genel Bakış

## Amaç

Bir talep-tahmini MLOps hattının modelleme bel kemiğini (model kataloğu →
rolling-origin backtest → guardrail'li çok-metrikli skorlama → per-group champion)
**tek dikeyden çıkarıp** modalite-agnostik, config-driven, tool-usable bir pakete
dönüştürmek.

## İlkeler

1. **Deterministik çekirdek.** v1'de LLM/agent yok. Bileşen orkestrasyonu →
   tekrarüretilebilir, sandbox derdi minimal. RAG/agent v2'de ayrı üst katman.
2. **Tipli sözleşmeler.** Her katman `contracts/` nesnesi alır, `contracts/`
   nesnesi üretir. Yan etki yalnız `persistence` ve `tracking`'te.
3. **Bulut opsiyonel.** Çekirdek bağımlılıklarında sıfır bulut. Tracking, storage,
   LLM sağlayıcı = eklenti; local dosya + no-op varsayılan.
4. **Eklenti mimarisi.** engine / model / metric / preprocessor / llm-provider
   Python entry-points ile dışarıdan genişletilebilir; çekirdeğe dokunmadan.
5. **Çapraz platform.** Çekirdek: numpy, pandas, pyarrow, scikit-learn, lightgbm,
   pydantic, joblib, pyyaml. Ağır DL/serimonik bağımlılıklar opsiyonel extra.

## Uçtan uca akış (v1)

```
Dataset (io)
  └─ analyzers ──────────► DataProfile + TaskSpec
       └─ dynamics ──────► AdaptivePlan
            └─ engine (tabular | timeseries)
                 ├─ models + registry ──► [Candidate...]
                 ├─ her Candidate: fine_tuners ──► validators ──► ValidationReport
                 ├─ scoring ──► ScoreBoard (+ guardrail) ──► SelectionResult
                 ├─ champion refit (tüm veri) ──► ModelBundle
                 └─ postprocessors eklenir
  └─ persistence ──► RunManifest + outputs/<DDMMYYYY>_<proje>_outputs/<run_id>/
  └─ reporters ────► EDA / model card / karşılaştırma / grafik
```

## Katman sorumlulukları

Ayrıntı: [`02_layers.md`](02_layers.md). Sözleşmeler: [`01_contracts.md`](01_contracts.md).

## Karar kayıtları

[`decisions/`](decisions/) — 0001…0006. Yeni her mimari karar bir ADR dosyası.

## Yol haritası

| Sürüm | Kapsam |
|---|---|
| v0.x | `contracts` + `config` + `io` + `analyzers` + tablo `core_engine` iskeleti |
| v1.0 | Tablo + zaman serisi tam akış; `statsforecast` engine; Caruana ensemble; raporlar |
| v1.1 | Subprocess runner gerçek kullanım; AutoGluon opsiyonel engine; metin modalitesi |
| v1.2 | Görsel + ses modaliteleri |
| v2.0 | RAG bilgi tabanı + agent üst katmanı (`llm/` + `interfaces/agent_tools.py` üstünde) |
