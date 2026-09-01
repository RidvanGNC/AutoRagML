# AutoRagML — Mimari Genel Bakış

## Amaç

Bir talep-tahmini MLOps hattının modelleme bel kemiğini (model kataloğu →
rolling-origin backtest → guardrail'li çok-metrikli skorlama → per-group champion)
**tek dikeyden çıkarıp** modalite-agnostik, config-driven, tool-usable bir pakete
dönüştürmek.

## Motto

> Zamanın miktarı önemsiz, **sağlıklı başarı** kesin ölçüdür. Detaylar önemlidir ve
> kaçırılması söz konusu değildir.

Hız/kısa yol uğruna doğruluktan taviz yok. Kesinlik gereken yerde yaklaşık yöntem yok
(ör. `fingerprint` strict, örneklem değil). Kenar durumlar ve genişleme dikişleri
tasarımda şimdi düşünülür; v1 kapsamı gereksiz şişirilmez.

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

## Uçtan uca akış (v1) — ADR 0015

```
config.resolve -> RunConfig
io.load        -> Dataset            [strict fingerprint]
analyzers      -> DataProfile + TaskSpec   [+ warnings]
engine seçimi  (modalite)
  her engine (EngineRunner):
    dynamics.planner  -> AdaptivePlan
    registry.resolve  -> [Candidate]           (YAML katalog)
    validators (nested CV):
       iç resample: fine_tuners.tune  -> TuningResult   (candidate_ops HPO uzayında)
       FittedTransform.fit(train, PlanContext) -> model fit + early stop -> apply(test)
       -> ValidationReport            [leakage: overlap/preprocessing/multi_test]
    scoring  -> ScoreBoard + SelectionResult   (seçim yalnız OOF; 1-SE kuralı)
    şampiyon refit (tüm train) + postprocessors -> ModelBundle
    -> EngineResult
final holdout (varsa): şampiyon BİR KEZ skorlanır
persistence -> RunManifest + outputs/<DDMMYYYY>_<proje>_outputs/<run_id>/
reporters   -> EDA / model card / karşılaştırma / grafik
-> RunResult  (.leaderboard / .predict / .explain / .champion / .manifest)
```

## Katman sorumlulukları

Ayrıntı: [`02_layers.md`](02_layers.md). Sözleşmeler: [`01_contracts.md`](01_contracts.md).

## Karar kayıtları

[`decisions/`](decisions/) — 0001…0020. Yeni her mimari karar bir ADR dosyası.
Katman kodu: `contracts` → `config` → `io` → `analyzers` → `dynamics` → `preprocessors` →
`models` → `validators` → `scoring` → `fine_tuners` → `engines` → `postprocessors` →
`persistence` → `reporters`+`tracking` → `interfaces` **(tümü yazıldı — v1 iskelet tam).**

## Yol haritası

| Sürüm | Kapsam |
|---|---|
| v0.x | `contracts` + `config` + `io` + `analyzers` + tablo `core_engine` iskeleti |
| v1.0 | Tablo + zaman serisi tam akış; `statsforecast` engine; Caruana ensemble; raporlar |
| v1.1 | Subprocess runner gerçek kullanım; AutoGluon opsiyonel engine; metin modalitesi |
| v1.2 | Görsel + ses modaliteleri |
| v2.0 | RAG bilgi tabanı + agent üst katmanı (`llm/` + `interfaces/agent_tools.py` üstünde) |
