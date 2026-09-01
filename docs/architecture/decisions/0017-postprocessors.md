# ADR 0017 — postprocessors: fit-ayrımlı tahmin düzeltme zinciri

**Durum:** Kabul · 2026-09-01

Kaynak: DemandSensing `_clean_faz2_amounts` / `_round_faz2_amounts` (clip lower=0 +
p99·mult üst sınır + eşikli yuvarlama); split-conformal (Vovk/Angelopoulos) — v1.1.

## İlke

Ham model tahmini (target-inverse **sonrası**, kullanıcıya dönmeden **önce**), iş
kuralı + istatistiğe uygun hale getiren **deterministik, sıralı, fit-ayrımlı** zincir.
`ModelBundle.pipeline` içine gömülür → `FittedModelPipeline.predict()` sonunda çalışır;
serving'de ve (v1.1'de opsiyonel) validation'da **aynı kod**.

## Kurallar

### 1. Leakage-safe (ADR 0011 uyumu)
Postprocessor'lar da 3-ilkel ayrımını izler:
- **stateless** — `clip`, `round`: parametre öğrenmez.
- **fit(y_true, y_pred) → FittedPostprocessor** — `calibrate` (bias/oran): **yalnız
  `champ_report.oof`** (`OOFArrays`) üzerinden. OOF = validators'ın ürettiği
  partition-temiz tahmin; train label'ı doğrudan görülmez → sızıntı yok.
- **apply(y_pred) → y_pred'** — saf, immutable.

Fit **yalnızca** `engines/champion.refit_champion` içinde. `analyzers`/`planner`
postproc parametresi fit etmez — yalnız `DataProfile.target_profile.stats.min`
üzerinden **öneri** (auto_nonneg).

### 2. Deterministik sıra (`_POST_ORDER`)
`calibrate` → `clip` → `round` → `business_rule` (hook)

Gerekçe: kalibrasyon ham tahmin dağılımında anlamlıdır (negatifleri kırpmadan önce
bias düzelt) → sonra fiziksel/iş sınırı (clip) → sonra tamsayı talep (round) →
mutlak son söz kullanıcı hook'u.

### 3. Kalibrasyon yöntemleri (v1)
- `additive_bias`: `y' = y − mean(y_pred_oof − y_true_oof)` (1 parametre, robust).
- `multiplicative`: `y' = y · clamp(Σy_true / Σy_pred, ratio_bounds)` (WMAPE-uyumlu
  ölçek düzeltme; `ratio_bounds` varsayılan (0.2, 5.0) güvenlik bandı).
- `linear` (a + b·y, OOF OLS) / isotonic → **v1.1** (küçük OOF'ta aşırı-öğrenme riski).

OOF yoksa (ör. `champ_report.oof is None`): kalibrasyon **atlanır** (WARNING), clip/round
çalışmaya devam eder.

### 4. auto_nonneg
`clip.auto_nonneg=True` (varsayılan) ve `clip.lower is None` ve görev regresyon/forecasting
ve `profile.target_profile.stats.min ≥ 0` → etkin `clip.lower = 0.0` (mesaj loglanır).
Kullanıcı `clip.lower`'ı açıkça verirse onunki kazanır. `auto_nonneg=False` ile kapatılır.

### 5. auto_upper (opsiyonel)
`clip.auto_upper_multiplier` verilirse: fit-zamanı `y_true_oof`'un
`auto_upper_percentile` (varsayılan 99) kuantili · çarpan → etkin `clip.upper`
(kullanıcı `clip.upper` verdiyse onunki kazanır). Çarpan `None` → auto_upper kapalı.

### 6. v1 sınırları (sözleşme rezerve, guard raise)
- `conformal` alt-sözleşmesi eklenir ama `conformal.enabled=True` → `ValueError`
  (v1.1: split-conformal mutlak-kalıntı kuantili + `FittedModelPipeline.interval()` +
  `RunResult.predict_interval()`).
- `apply_in_validation=True` → `ValueError` (v1: serving-only; scoreboard ham modeli
  yansıtır → adaylar karşılaştırılabilir, postproc kötü modeli maskeleyemez).
- `business_rule` hook: `RunConfig`'e girmez (serialize edilemez); `interfaces` katmanı
  `FittedPostprocessor`'a enjekte eder. v1 engine yolu `None` bırakır.

### 7. is_active / no-op
`PostprocessConfig` **varsayılanı tam no-op** (clip çözülmez, `round.mode="off"`,
`calibrate.method="off"`). Hiçbir adım etkin değilse `FittedModelPipeline.postprocessor
= None` → `predict()` ekstra maliyet almaz.

## Sözleşme

`contracts/postprocess_config.py`: `ClipConfig`, `RoundConfig`, `CalibrateConfig`,
`ConformalConfig`, `PostprocessConfig`. `RunConfig.postprocess: PostprocessConfig`.
`BundleMetadata.postprocess_summary: dict[str, Any]` — uygulanan adımların
serialize edilebilir kaydı (persistence/reporters okur).

## API

```
postprocessors/
  __init__.py   build_postprocessor(cfg, profile, task) -> Postprocessor
  pipeline.py   Postprocessor.fit(y_true?, y_pred?, groups?) -> FittedPostprocessor
                FittedPostprocessor.apply(y_pred) -> y_pred'   (immutable, __slots__)
  steps.py      _resolve_clip / _calibrate_params / _apply_round  (saf yardımcılar)
```

## Sonuç

- `FittedModelPipeline` `postprocessor` slotu alır; `predict()` target-inverse'ten
  sonra `postprocessor.apply` çağırır (None ise atlar).
- `refit_champion` `build_postprocessor` → `champ_report.oof` ile `fit` → bundle'a gömer;
  `BundleMetadata.postprocess_summary` doldurur.
- Determinist: tüm işlemler numpy, rastgelelik yok.
