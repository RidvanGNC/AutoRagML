# ADR 0036 — sınıflandırma GES + bagging (olasılık OOF)

**Durum:** Kabul · 2026-09-03 (kullanıcı 2 kararı kilitledi — hepsi önerilen)

**Kilitli kararlar:**
- **K1 = (c)** — GES arama metriği: `primary_metric` proba-aware (`roc_auc`/`log_loss`) ise onda
  ara; nokta metrik (`accuracy`/`f1_macro`) ise GES **log-loss**'ta arar, seçim/promotion/rapor
  yine `primary_metric`.
- **K2 = evet** — bagging = fold `predict_proba` ortalaması → argmax.
- **K3 = binary + multiclass.** multilabel → v1.1+.
- **K4 = kalibrasyon bu ADR'de YOK** — ayrı ADR (ADR 0017 konformal ertelemesiyle).

Kaynak: v1.1 sırası. ADR 0021 (GES) + ADR 0022 (bagging) **"v1: regresyon + forecasting"** dedi;
sınıflandırma tek-model refit'te kalıyor (`_BAGGABLE_TASKS`/`_ELIGIBLE_TASKS` sınıflandırmayı
dışlıyor). Regresyon GES+bagging'den fayda görüyor (benchmark: ADR 0022 net pozitif) — sınıflandırma
aynı kazançtan mahrum. Tamamlık gap'i.

## Araştırma özeti (2026-09)

- **AutoGluon:** Caruana GES sınıflandırmada da **olasılık OOF** üstünde: her modelin `predict_proba`
  OOF matrisi → ağırlıklı ortalama olasılık → metrik (varsayılan **log-loss**, ya da eval metriği).
  Ağırlık = seçilme sayısı / S. k-fold bagging her satır için OOF olasılık üretir → leakage yok.
- Aynı Caruana algoritması; tek fark: 1-boyutlu tahmin yerine `(n, C)` olasılık matrisi blend'i.

## İlke

- **Yeni engine yok.** Mevcut GES (`ensembling/`) + bagging (`champion._fit_pipeline`) sınıflandırmaya
  genişler. Nokta OOF yerine **olasılık OOF** (`predict_proba`).
- **Leakage-safe by construction (ADR 0011):** olasılık OOF de nested CV'den; `fit` yalnız validators.
- `predict_proba` sunmayan modeller (nadir) → sınıflandırma GES/bagging'e girmez; tek-model şampiyon olabilir.

## Açık kararlar (kullanıcıya sorulacak)

### K1 — GES arama metriği (sınıflandırma)
- **(a) Her zaman log-loss** — proper scoring rule, pürüzsüz yüzey (AutoGluon varsayılanı).
  Nihai sıralama/rapor yine `primary_metric` (accuracy/f1).
- **(b) primary_metric** (accuracy/f1) — kullanıcının hedefiyle birebir ama argmax → basamaklı/pürüzlü
  GES yüzeyi, yerel minimuma takılır.
- **(c) primary proba-aware ise onu, değilse log-loss** — `roc_auc`/`log_loss` → doğrudan;
  `accuracy`/`f1_macro` → GES log-loss'ta arar, primary'de raporlar.
- **Öneri: (c)** — kullanıcı proba metriği seçtiyse ona uy; nokta metriği seçtiyse arama için
  log-loss (daha iyi GES), seçim/promotion primary'de.

### K2 — bagging olasılık yolu
- Sınıflandırma bagging = fold-modellerinin **`predict_proba` ortalaması** → `argmax` (serving).
  OOF olasılık ortalaması postprocess'e/GES'e girer. `_BAGGABLE_TASKS` += binary/multiclass.
- **Öneri: evet, standart.** (onay sorusu)

### K3 — kapsam
- **(a) binary + multiclass** — multilabel ve ordinal hariç (ordinal zaten regresyon yolunda;
  multilabel ayrı — sigmoid başına, GES yine olasılık ama şekil farklı).
- **(b) + multilabel** — ek şekil işleme.
- **Öneri: (a)** — multilabel → v1.1+ (kullanım az).

### K4 — olasılık kalibrasyonu
- Ensemble olasılıkları Platt/isotonic ile kalibre edilsin mi?
- **Öneri: hayır (bu ADR'de)** — kalibrasyon ayrı ADR (konformal aralıklarla birlikte, ADR 0017
  ertelemesi). Bu ADR yalnız GES+bagging.

## Sözleşme (kilitlendi)

- `contracts/validation.OOFArrays` — `y_proba: np.ndarray | None = None` (n×C; additive).
- `scoring/metrics` — `compute_proba_metrics(y_true, y_proba, task) -> {log_loss, roc_auc, +argmax
  accuracy/f1}`; `PROBA_METRICS` kümesi.
- `validators/runner.run_validation` — sınıflandırmada `est.predict_proba` OOF'u topla →
  `OOFArrays.y_proba`; `est` proba sunmuyorsa `y_proba=None`.
- `ensembling/greedy.greedy_selection_proba(proba_stack: (m,n,C), y_true, metric_fn, ...) -> (m,)` —
  Caruana döngüsü, `ens_sum` şekli `(n, C)`.
- `ensembling/__init__.build_weighted_ensemble` — `_ELIGIBLE_TASKS` += binary/multiclass;
  sınıflandırma dalı: `y_proba` dolu raporlar → `greedy_selection_proba` → `weighted_ensemble`.
- `engines/model_pipeline.FittedModelPipeline.predict_proba` (+ sınıflandırmada `predict` = argmax).
- `engines/ensemble_pipeline.FittedEnsemblePipeline` — proba modu: üye `predict_proba` ağırlıklı
  ortalama → `predict` argmax, `predict_proba` matris.
- `engines/champion` — `_BAGGABLE_TASKS` += binary/multiclass; bag OOF olasılık; `_refit_ensemble`
  sınıflandırmada proba üye blend'i.
- `RunConfig` — değişiklik yok (ensemble/bagging config'leri zaten var); belki `ges_classification_metric`
  (K1'e göre).
- `Candidate`/`EngineResult`/`ModelBundle` **değişmez**.

## Kapsam dışı / sonra

- Olasılık kalibrasyonu (Platt/isotonic/conformal) → ayrı ADR.
- Multilabel GES → v1.1+.
- L2 stacking sınıflandırma (proba OOF) → ADR 0034 genişletmesi, v1.1+.
- Sınıf-dengesiz özel muamele (SMOTE vb.) → kapsam dışı (v1 felsefesi: dönüştürme değil metrik seçimi).
- Public API `predict_proba` (`RunResult`/`LoadedChampion`) → ayrı küçük iş (ADR 0037 ile birlikte olabilir).

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `scoring/metrics`: `compute_proba_metrics(y_true, y_proba, *, classes)` → `log_loss` + `roc_auc`
  (ikili/OvR-macro) + argmax nokta metrikleri. `PROBA_METRICS` + `is_proba_metric`.
- `validators/frame_ops.OOFArrays` += `y_proba` (n×C) + `classes` (additive dataclass alanı).
- `validators/runner.run_validation` — sınıflandırmada fold-başı `est.predict_proba` toplanır
  (`_predict_proba` / `_estimator_classes`); tüm fold'lar proba verirse `OOFArrays.y_proba` =
  vstack, `oof_metrics` proba metrikleriyle zenginleşir. `predict_proba` yoksa → `None` (aday
  yine nokta OOF ile yarışır).
- `ensembling/greedy.greedy_selection_proba(proba_stack (m,n,C), ...)` — Caruana döngüsü olasılık
  matrisi üstünde.
- `ensembling/__init__._build_classification_ensemble` — `_proba_aligned` raporlar → GES;
  arama metriği: primary proba-aware ise primary, değilse `log_loss` (K1c). Blend proba →
  `argmax` → `weighted_ensemble` (`y_pred`=etiket, `y_proba`=blend).
- `engines/model_pipeline.FittedModelPipeline` — `predict_proba` + `classes` property (`_design_matrix`).
- `engines/ensemble_pipeline.FittedEnsemblePipeline` — `classes` verildiğinde **olasılık modu**:
  `_blend_proba` ağırlıklı → `predict` argmax, `predict_proba` matris.
- `engines/champion` — `_BAGGABLE_TASKS` += binary/multiclass; bag `FittedEnsemblePipeline(classes=)`;
  `_refit_ensemble` sınıflandırmada `classes` geçirir.
- `Candidate`/`EngineResult`/`ModelBundle`/`RunConfig` **değişmedi**.
- Testler: `test_classification_ges.py` (5) + `test_bagging.py::test_classification_bagged_or_ensembled_discrete`.

**Benchmark → tüm v1.1 eklemeleri bitince.**
