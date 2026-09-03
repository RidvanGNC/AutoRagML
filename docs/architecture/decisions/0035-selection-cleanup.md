# ADR 0035 — seçim temizliği (champion refit-on-full · ortak forecasting ensemble · aile-arası tie-break)

**Durum:** Kabul · 2026-09-03 (kullanıcı 3 kararı kilitledi — hepsi önerilen seçenek)

**Kilitli kararlar:**
- **K1 = (a)** — `EngineResult.finalize` closure; orchestrator holdout skorundan sonra çağırır.
- **K2 = (a)** — ortak cutoff ızgarası: reduction modelleri de klasik CV cutoff'larında
  değerlendirilir → tek GES tüm forecasting aileleri üzerinde.
- **K3 = (c)** — `_within_one_se` aday-başı SE; bandda eşitlikte robustluk (fold sayısı + SE) →
  family complexity → süre.

Kaynak: v1.1 sırası. ADR 0014/0020/0021/0023 boyunca biriken 3 "seçim borcu":
1. **champion refit-on-full** — ADR 0029'dan ertelendi (orchestrator refit closure).
2. **klasik + reduction ortak forecasting ensemble** — ADR 0023/0024'ten ertelendi (ortak backtest).
3. **aile-arası tie-break** — 1-SE bandında heterojen OOF ızgaralarında adil kıyas.

(Metrik-duyarlı promotion → **✅ zaten yapıldı**, commit 020f258.)

---

## Parça 1 — champion refit-on-full

### Sorun
ADR 0020: orchestrator holdout carve → engine yalnız `train − holdout` görür → şampiyon o veride
fit → **serving modeli holdout'u hiç görmez**. Forecasting'de holdout = *en yakın* dönem →
servis edilen tahminci "şimdi"den 1 ufuk önce eğitilmiş kalır (ciddi kusur). Tabular'da ~%10-20
daha az veri (küçük veride önemli).

ADR 0029 klasik forecaster için `_horizon_for` ile *serving* tarafını yamadı ama model hâlâ
`train − holdout`'ta fit. Reduction + nöral forecaster'lar erken bitiyor.

### Doğru sıralama (değişmez)
1. engine: CV(train) → seç → **şampiyonu `train`'de refit** → holdout-şampiyonu
2. orchestrator: holdout-şampiyonunu holdout'ta skorla → **dürüst holdout metriği** (tek dokunuş)
3. engine: **şampiyonu `train + holdout` (full) refit** → serving/persist edilen model
4. full-şampiyon, 2. adımın holdout metriğini taşır

### Tasarım seçenekleri (K1)
- **(a) `EngineResult.finalize` closure** — engine, `(selection, candidates, reports, plan,
  profile, task, config, tuner, pre_transform)`'ı kapatan bir callable set eder
  (`arbitrary_types_allowed`, serialize edilmez). Orchestrator: `if er.finalize: er.champion =
  er.finalize(frame_full)`. `run_core_pipeline` closure'ı kurar; TS/tabular engine sadece
  `EngineResult`'ı geçirir. **Segmented (ADR 0028):** her segment closure'ı → birleşik finalize.
  ~120 satır, ADR 0015 `EngineResult`'a 1 excluded alan.
- **(b) reports'u persist et + `validation_reports_ref`'ten reload** — `EngineResult` zaten ref
  alanına sahip. Ağır (numpy diske), orchestrator↔persistence sıralaması karışır.
- **(c) Ertelemeyi sürdür** — yalnız forecasting'de küçük ek: reduction/nöral refit'e holdout'u
  da kat (serving-only, scoreboard değişmez). Tabular'da hiç. En düşük risk, yarım çözüm.
- **Öneri: (a)** — tam çözüm, kapsanabilir risk. (Not: bir kez denenip geri alınmıştı —
  `CoreRun` sınıf refaktörü çok genişti; bu sürüm tek closure alanı.)

### Guard
Full-refit şampiyonu holdout metriğini **kötüleştiremez** kontrolü YOK (holdout artık train'in
parçası — tekrar skorlanamaz). Bunun yerine: full-refit **aynı** aday + aynı `candidate_choices`
+ aynı `params` + `best_iteration` (ES modelleri holdout büyümesiyle orantılı ölçeklenir). Model
değişmez, yalnız veri büyür → regresyon riski minimal (ADR 0022 bagging zaten benzer).

---

## Parça 2 — klasik + reduction ortak forecasting ensemble

### Sorun
GES (ADR 0021) yalnız reduction OOF'unu (fold-hizalı) blend'ler. Klasik (statsforecast) OOF'u
`cross_validation` cutoff-tabanlı → hizasız → ayrı `classical_ensemble` (EAT, ADR 0024). İkisi
**birlikte** blend'lenemiyor. M4 winner = forecast combination (klasik + ML birlikte).

### Tasarım seçenekleri (K2)
- **(a) Ortak cutoff ızgarası** — reduction modellerini de klasik `cross_validation` cutoff'larında
  değerlendir (aynı `h`, `n_windows`, `step_size`). Tüm OOF aynı ızgarada → tek GES. Reduction
  için ek CV maliyeti (zaten fold CV yapıyor → ~2×).
- **(b) Klasiği reduction fold ızgarasına taşı** — `StatsForecast.cross_validation` yerine elle
  rolling-origin, reduction fold sınırlarıyla. statsforecast API'sinden uzaklaşır.
- **(c) Post-hoc hizalama** — her iki ızgaradan ortak (unique_id, ds) kesişimi → o alt kümede GES.
  Basit ama OOF küçülür, bazı seriler düşer.
- **Öneri: (a)** — en temiz; `run_classical_reports` zaten cutoff ızgarası kuruyor, reduction'ı
  oraya sokmak `FittedModelPipeline`'ı cutoff başına predict etmek. Orta karmaşıklık.
- **Alternatif: ertele** — bu M4/heterojen-panel kazancı; M5 segmented + M3 auto_ets zaten iyi.

---

## Parça 3 — aile-arası tie-break

### Sorun
1-SE bandı (ADR 0014): `noise_floor` = fold-SE medyanı. Klasik OOF (cutoff, 3 pencere) ile
reduction OOF (fold, 4-5) farklı örneklem → SE ölçekleri farklı. `_within_one_se` tek `noise_floor`
kullanıyor → heterojen bandda yanlış üye. Tie-break yalnız `_FAMILY_COMPLEXITY` + `realized_seconds`.

### Tasarım seçenekleri (K3)
- **(a) Aday-başı SE** — `_within_one_se` her adayın kendi `oof_metric_se`'siyle karşılaştırır
  (medyan yerine): `r.mean ≤ best.mean + max(best.se, r.se)`. Daha savunulabilir.
- **(b) Robustluk tie-break** — bandda eşitlikte: daha çok fold + daha düşük SE (daha güvenilir
  tahmin) tercih; sonra family complexity; sonra süre.
- **(c) İkisi de** (a) + (b).
- **Öneri: (c)** — küçük, düşük risk, seçimi dürüstleştirir.

---

## Sözleşme (kilitlenince)

- `contracts/engine_result.py` — `finalize: FinalizeFn | None = Field(default=None, exclude=True)`
  (`arbitrary_types_allowed=True`). ADR 0015 "dondu" notu → additive-excluded istisna.
- `engines/core.run_core_pipeline` — closure kurar; `_finalize_on_full(frame_full) -> ModelBundle`.
- `engines/segmented.run_segmented` — segment closure'larını birleşik finalize'a sarar.
- `interfaces/orchestrator` — `holdout_score` sonrası `finalize` stage; `champion` değişir,
  `metrics_holdout` korunur; manifest/persist full-şampiyonu yazar.
- `engines/timeseries/classical.py` + `core.py` — reduction'ı cutoff ızgarasında değerlendiren
  ortak yol (Parça 2 seçilirse).
- `scoring/selection._within_one_se` — aday-başı SE + robustluk tie-break (Parça 3).
- `RunConfig` (additive): `champion_refit_full: bool = True` (Parça 1 kapatılabilir).

## Kapsam dışı / sonra

- Nested holdout / repeated holdout → v1.1+.
- Konformal aralıklar (ADR 0017 ertelemesi) → ayrı ADR.
- HPO koşumunda finalize re-tune donması → şimdilik yeniden tune (ADR 0022 bagging deseni;
  `hpo_level=none` deterministik). İstikrarsızlık görülürse param-donması v1.1+.

## Sonuç (Parça 1 + 3 UYGULANDI — commit [pending], 2026-09-03)

### Parça 3 — aile-arası tie-break ✅
- `contracts/scoreboard.ScoreRow.n_folds` (additive).
- `scoring/selection._within_one_se` — aday-başı SE: band = `max(best.se, r.se, noise_floor)`.
- `select_champion` tie-break: `(_FAMILY_COMPLEXITY, -n_folds, oof_metric_se, realized_seconds)`.

### Parça 1 — champion refit-on-full ✅
- `contracts/engine_result.EngineResult.finalize: Any` (`exclude=True` — serialize edilmez).
- `engines/core.run_core_pipeline` — `_finalize_on_full(full_frame)` closure: `pre_transform`
  (reduction partial) full ham frame'e uygulanır → `refit_champion`. Tabular passthrough,
  recursive passthrough (kendi FE'si).
- `engines/segmented.run_segmented` — `_seg_finalize`: her segment kendi `finalize`'ıyla
  segment-filtreli full frame'de refit → birleşik `FittedSegmentedPipeline`.
- `interfaces/orchestrator` — `holdout_score` sonrası **`finalize` stage**: `champion_refit_full`
  açık + holdout var + `engine_result.finalize` varsa → `champion = finalize(frame_full)`,
  `metrics_holdout` (train-şampiyonundan ölçülen dürüst metrik) yeni bundle'a aktarılır.
  Refit çökerse train-şampiyonu korunur (fail-safe).
- `RunConfig.champion_refit_full: bool = True`.
- Testler: `test_champion_refit_full_stage`, `test_champion_refit_full_can_be_disabled`.

### Parça 2 — klasik + reduction ortak forecasting ensemble → **[pending]**
Ayrı commit (ortak cutoff ızgarası — CV hizalama makinesi).
