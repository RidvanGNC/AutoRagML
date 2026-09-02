# ADR 0024 — klasik model ansamblı (EAT) + intermittent için Tweedie objective

**Durum:** Kabul · 2026-09-02

Kaynak: SOTA gap analizi (`research/2026-09-sota-gap-analysis.md`).
- **M3 yarışması: Theta** kazandı; en iyi ansambl **EAT = ETS + ARIMA + Theta**.
- **M4 yarışması:** forecast-combination + meta-learning (HOC2: AutoETS+AutoARIMA+Theta+
  TBATS+SeasonalNaive konveks kombinasyonu).
- **M5 yarışması:** hiyerarşik LightGBM + **Tweedie loss** (talebin %78.6'sı düzensiz).

## Bölüm A — klasik model ansamblı

### Sorun
`build_weighted_ensemble` (GES) reduction adaylarının fold-hizalı OOF'unda çalışır;
klasik adaylar cutoff-tabanlı OOF (StatsForecast `cross_validation`) ile dışlanır. Ama
**tüm klasik adaylar aynı `sf.cross_validation` çağrısından geçer** → `cv` DataFrame'i
aynı → OOF **kendi içinde tam hizalı**. Yalnız reduction raporları önce geldiği için
`_aligned_reports[0]` reduction oluyor.

### Karar
`run_classical_reports` per-model raporlara ek olarak bir **`classical_ensemble`** raporu
üretir:
- Klasik OOF matrisi `[n_satır × n_klasik_model]` üzerinde GES / bagged-GES (`ensembling.greedy`).
- Ağırlığı > 0 üyeler; `< 2` üye → ansambl yok.
- Sentetik `ValidationReport(candidate_key="classical_ensemble")` + `Candidate(family="ensemble",
  ensemble_members={aday_key: ağırlık})`.
- Scoreboard'da her aday gibi yarışır (`_FAMILY_COMPLEXITY["ensemble"]=5`).

### Refit
`champion.key == "classical_ensemble"` → `_classical_ensemble_bundle`:
- `ensemble_members`'taki her klasik model **tek `StatsForecast(models=[...])`** içinde fit.
- `FittedClassicalForecaster` çok-model + ağırlık vektörü tutar; `predict` = `sf.predict(h)`
  kolon-başı-model çıktılarının ağırlıklı ortalaması.

### v1 sınırı korunur
Klasik + reduction **ortak** ensemble (cutoff ↔ fold OOF hizalama) hâlâ v1.1.
İki ayrı ensemble (`weighted_ensemble` reduction, `classical_ensemble` klasik) scoreboard'da yarışır.

## Bölüm B — Tweedie objective (intermittent talep)

### Karar
`dynamics/planner` intermittency ipucundan **model param ipucu** üretir:
`AdaptivePlan.model_hints: dict[str, dict[str, Any]]` (aday_key veya family → param sözlüğü).

`_model_hints(profile, task)`:
- forecasting + `intermittency_summary`'de `{intermittent, lumpy, erratic}` payı ≥ %50 →
  - `lightgbm`: `{"objective": "tweedie", "tweedie_variance_power": 1.3}`
  - `hist_gbm`: `{"loss": "poisson"}` (sklearn HGB Tweedie yok; Poisson en yakın, y≥0 şart)
  - `xgboost`: `{"objective": "reg:tweedie", "tweedie_variance_power": 1.3}`
- yalnız **reduction** GBDT adaylarına; klasik/lineer/forest'a değil.

`engines/core.run_core_pipeline`: `resolve_candidates` sonrası
`models.apply_model_hints(candidates, plan.model_hints)` → eşleşen adayın
`default_params` ile merge (immutable `model_copy`).

"Magic multipliers" (M5 2. sıra) ≈ `postprocess.calibrate="multiplicative"` — **zaten var**.

## Sözleşme
- `AdaptivePlan.model_hints` (additive; DONDU'ya uyumlu).
- Yeni key sabiti: `ensembling`/`classical` içinde `CLASSICAL_ENSEMBLE_KEY = "classical_ensemble"`.
- `run_classical_reports` → `tuple[list[ValidationReport], list[Candidate]]` (raporlar + sentetik cand).

## Kapsam dışı / sonra
- HOC2 tarzı **horizon-optimize** kombinasyon (her horizon adımı için ayrı ağırlık) → v1.1.
- TBATS eklenmesi (statsforecast'te yok; `tbats` paketi) → v1.1.
- Recursive multi-step + zengin reduction özellikleri → ADR 0025.

## Sonuç
- `classical.py`: `_classical_ensemble_report` + `FittedClassicalForecaster` çok-model + refit.
- `planner._model_hints` + `models.apply_model_hints`; `AdaptivePlan.model_hints`.
- benchmark: m3 (EAT ansamblı) + m5 (Tweedie) ile ölçülür.
