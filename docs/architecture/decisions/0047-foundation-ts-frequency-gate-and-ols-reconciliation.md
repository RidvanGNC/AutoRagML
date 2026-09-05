# ADR 0047 — foundation-TS/nöral-TS frekans kapısı + OLS reconciliation + tourism hiyerarşi

**Durum:** Kabul · 2026-09-05 (full benchmark tourism gerilemesi + geniş literatür taraması)

Kaynak: full-profil benchmark (`_runs/20260905T095551Z/PARTIAL_RESULTS.md`) — tourism_large tam
ölçekte **-23.1% NO_IMPROVEMENT** (v4 auto_ets **+15.9%**'du). ADR 0042 guard'ı dev'de (120 seri)
çalıştı ama full'de (555 seri) timesfm SE'si 1.70→0.85'e düştüğü için `margin > max(SE)` eşiği
kolaylaştı → timesfm tutuldu → OOF 15.5 vs gerçek holdout 24.2 (+56% sapma).

## Literatür bulguları (geniş tarama — forum/GitHub/Kaggle/akademik)

| kaynak | bulgu |
|---|---|
| [TS FM Benchmarking Challenges](https://arxiv.org/html/2510.13654v1) (Ekim 2025) | Tourism Monthly/Quarterly/Yearly + M3 + M4 **Chronos/TimesFM/Moirai ön-eğitim korpuslarında** (Tablo 5). Standart benchmark'ların yalnız %7'si temiz. Sızıntı: "32 puan MAPE" / "%47-184 MSE avantajı" (temiz veride %0.3-14). |
| [Monash Archive](https://forecastingdata.org/) — Tourism Monthly (medyan MASE) | ETS **1.276** ≈ DeepAR **1.247** (en iyi) ≈ ARIMA 1.337; generic ML CatBoost 1.461 / FFNN 1.450 / PR 1.484 belirgin daha kötü. |
| [Tourism Forecasting Competition](https://robjhyndman.com/papers/forecompijf.pdf) (Athanasopoulos-Hyndman 2011) | "Pure time series approaches forecast tourism demand MORE accurately than methods with explanatory variables. ARIMA/ETS consistently beat seasonal-naive for seasonal data." |
| [Cherry-Picking](https://arxiv.org/abs/2412.14435) (AAAI 2025) | "Deep learning shows high sensitivity to dataset selection; classical methods are more robust." 4 dataset → %46 metot "best in class" gösterilebilir. |
| [FPP3 §11](https://otexts.com/fpp3/tourism.html) | Tourism reconciliation: **ETS taban + OLS veya MinT**. OLS, MinT'i geçti (agg RMSE 1803 vs 2157). Bottom-up çok daha kötü. |

**Teşhis:** v4 (auto_ets +15.9%) doğruydu ve literatürle rekabetçiydi. Gerileme, tam da bu
benchmark'larda kontamine olan foundation_ts modellerini havuza eklemekten. Guard yığını
(ADR 0038→0039→0042→0042-B) temelde çözülemez bir şeyle savaşıyor.

## Karar

### (A) Yapısal frekans kapısı — post-hoc guard DEĞİL

`engines/timeseries/classical.is_low_frequency_panel(profile)` — freq ∈ {M, Q, Y, A}. Bu panel
tipinde:
- `run_foundation_ts_reports` ve `run_neural_ts_reports` **erken dönüş** yapar (aday havuza hiç
  girmez) — `config.foundation_enabled`/`neural_enabled` `"on"` değilse.
- `"on"` (açık opt-in) → kapı bypass edilir; kullanıcı özel (kontamine-olmayan) aylık verisinde
  foundation/nöral istiyorsa açar.
- `"auto"` = GPU kapısı + frekans kapısı; `"off"` = hiç.

**Neden guard değil kapı:** (1) kontaminasyon benchmark-özgü, post-hoc tespit edilemez;
(2) "seçim yalnız OOF'tan" ilkesi (ADR 0014) bozulmaz — aday hiç yarışmaz; (3) guard-yığını durur;
(4) literatür: aylıkta klasik ≈ en iyi DL, generic ML daha kötü → gerçek veride bile kayıp az.

### (B) OLS reconciliation varsayılan

`RunConfig.hierarchy_reconcile_method: Literal["ols", "wls_struct"] = "ols"`. `reconcile()` ve
`FittedHierarchicalForecaster` `method` parametresi alır. İkisi de residual/OOF gerektirmez
(`mint_shrink` → ADR 0045-B). FPP3'te OLS tourism'de MinT'i geçmişti.

### (C) tourism hiyerarşi benchmark'ı

`BenchmarkDataset.hierarchy_cols` alanı + `run_forecasting`'de `overrides["hierarchy_cols"]`
aktarımı. Yeni dataset `tourism_hier` — TourismLarge'ın **coğrafya ağacı** (76 bölge → 27 zone →
7 eyalet → 1 toplam), tüm-amaç serileri. `group_col="region"`, `hierarchy_cols=["state","zone"]`.
Grouped (coğrafya × seyahat amacı) yapının **amaç boyutu atlandı** — `hierarchy_cols` lineer bir
ağaç; crossed/grouped hiyerarşi desteği ayrı bir iş (ADR 0048?). Mevcut `tourism_large` (düz
555-seri panel) korundu — `tourism_hier` additive.

## Beklenen etki (ölçülmedi — kullanıcı benchmark/test istemedi)

- tourism_large: foundation_ts/neural_ts havuzdan çıkar → `joint_ensemble` (16.87) veya
  `dynamic_theta`/`auto_ets` (~17.3) şampiyon → holdout ~17-18 → vs naive 19.64 → **+8-13% SUCCESS**.
- m3_monthly: aynı mantık → klasik ansambl şampiyon; M3 seasonal-naive near-SOTA olduğundan
  yine "par" civarı (v4 patchtst -9.9%'du) ama en azından kontamine-timesfm seçilmez.
- m5/m4_hourly/ett (daily/hourly) → **etkilenmez**, foundation_ts genuinely yardım ediyor.
- tourism_hier: reconciliation'ın gerçek testi (OLS + coğrafya ağacı).

## Doğrulama

- ruff + mypy(147 dosya) yeşil.
- `is_low_frequency_panel` smoke: {MS,M,Q,QS,Y,A}→True · {W,D,H,B,None}→False.
- `tourism_hier` loader smoke: 76 bölge / 27 zone / 7 eyalet, `[state,zone,region,ds,y]`.
- `reconcile(S, y_hat, method="ols")` smoke: coherent (çocuklar toplamı = ebeveyn).
- `tests/unit/engines/test_hierarchical.py` 6/6 yeşil (OLS varsayılanıyla).
- **Kullanıcı kararı: formal pytest testleri + benchmark bu turda YAPILMADI.** Testler + full
  benchmark, ADR 0047 gözden geçirildikten sonra bir sonraki turda.

## Kapsam dışı

- ADR 0042 guard'ı KALDIRILMADI — daha yüksek frekanslarda (W/D/H) foundation_ts hâlâ yarışıyor,
  guard orada koruma sağlamaya devam ediyor. Frekans kapısı yalnız M/Q/Y'yi kesiyor.
- Crossed/grouped hiyerarşi (tourism'in tam yapısı: coğrafya × amaç) — lineer `hierarchy_cols`
  bunu ifade edemez; "bring-your-own-S" yolu veya grouped-spec desteği ayrı ADR.
- `mint_shrink` reconciliation (OOFArrays.ds gerektirir) — ADR 0045-B.
- Frekans kapısının `TabPFN`/`TabICL` (tablo foundation) üzerinde etkisi YOK — onlar sentetik-veri
  ön-eğitimli, kontaminasyon riski taşımıyor (ADR 0042'de de kümede değillerdi).
