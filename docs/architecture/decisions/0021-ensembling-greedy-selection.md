# ADR 0021 — ensembling: Caruana greedy selection (+ bagged GES)

**Durum:** Kabul · 2026-09-01

Kaynak: Caruana et al. 2004 "Ensemble Selection from Libraries of Models"; Caruana et al.
2006 "Getting the Most Out of Ensemble Selection" (bagging); TabArena (NeurIPS 2025) —
post-hoc GES ensemble tek tek her modelden **ve AutoGluon'dan** anlamlı daha iyi;
referans implementasyonlar: AutoGluon (Apache-2.0), auto-sklearn (BSD-3).

## İlke

En yüksek getirili eksik. `validators`'ın ürettiği **hizalı OOF tahminlerinden** post-hoc
ağırlıklı ensemble kurulur; tek-model şampiyonuyla **aynı seçim mekanizmasında yarışır**
(sentetik `ValidationReport` + `Candidate`).

### ADR 0004 revizyonu
ADR 0004 "AutoGluon kodunu `_vendor/`'a kopyala" diyordu. GES algoritması **Caruana 2004
akademik makalesinden**; AutoGluon-özel değil. Bu yüzden `_vendor/` yerine **temiz
implementasyon** `autoragml/ensembling/` — kod idiomlarımıza uygun, tam tipli, deterministik,
Türkçe yorumlu. Makale + AutoGluon/auto-sklearn referans olarak docstring + `NOTICE`'ta anılır.
Caruana ensemble için `_vendor/` planı iptal.

## Kapsam

- **v1: regresyon + forecasting.** OOF = nokta tahminleri → ağırlıklı ortalama → doğrudan
  metrik. Tam da GES'in çalıştığı yer.
- **Sınıflandırma GES → v1.1** (olasılık OOF gerekir; `OOFArrays` şu an yalnız `y_pred`).

## Algoritma (`ensembling/greedy.py`, saf numpy)

### Greedy Ensemble Selection
1. Girdi: `P` (n_örnek × n_model) OOF nokta tahmin matrisi, `y_true`, metrik (`lower_is_better`).
2. **Sorted init:** ensemble'ı tek-model metriğine göre en iyi `sorted_init_k` model ile başlat.
3. `max_models` tur: her aday `m` için varsayımsal ensemble ortalaması
   `(Σ_current + P[:,m]) / (k+1)` → metrik. En iyiyi seç (**with replacement** — bir model
   çok kez seçilebilir).
   - Tie: metrik 6 ondalığa yuvarlanır → önce ensemble'da olan model → sonra en düşük indeks
     (seed'li deterministik).
   - Non-finite metrik → `+inf`.
4. **use_best:** greedy boyunca **görülen en iyi** ensemble durumuna geri dön (son değil —
   auto-sklearn'ün meşhur bug'ı buydu).
5. Ağırlık: `w[m] = seçilme_sayısı(m) / toplam_seçim`.

### Bagged GES (varsayılan açık)
`n_bags` kez: model kütüphanesinin rastgele `bag_fraction` alt-kümesinde (seed'li) GES →
ağırlık vektörü. Bag'ler arası ağırlıkları **ortala**. Val setine aşırı-uyumu azaltır
(Caruana 2006).

## Sözleşme

`contracts/ensemble_spec.py`:
```python
class EnsembleSpec(FrozenContract):
    member_keys: list[str]          # ağırlığı > 0 olan aday key'leri
    weights: list[float]            # member_keys ile hizalı, toplamı ~1
    method: Literal["ges", "bagged_ges"]
    n_bags: int = 0
    oof_metric: float               # ensemble'ın OOF birincil metriği
    base_model_count: int
```
`RunConfig.ensemble: EnsembleConfig`:
```python
class EnsembleConfig(Contract):
    enabled: bool = True
    max_models: int = 50            # GES tur sayısı
    sorted_init_k: int = 1
    bagging: bool = True
    n_bags: int = 20
    bag_fraction: float = 0.5
    min_base_models: int = 2        # < ise ensemble yok
```
`BundleMetadata.ensemble: dict[str, Any] = {}` — üye key + ağırlık kaydı.

## Akış entegrasyonu

`ensembling.build_weighted_ensemble(reports, candidates, config, task, profile)
-> tuple[ValidationReport, Candidate, EnsembleSpec] | None`:
- eligible = quarantine olmayan, OOF'u hizalı (aynı `y_true`, aynı fold `n_test` dizisi) reportlar.
- `< min_base_models` → `None`.
- GES/bagged-GES → ağırlıklar. `w == 0` üyeler düşer. Tek üye kalırsa → `None`.
- Ensemble OOF = `Σ w_i · P_i`. **Fold başına** (OOF'u kümülatif `n_test` ile dilimle) metrik →
  `oof_metric_se`. Sentetik `ValidationReport(candidate_key="weighted_ensemble", folds=..., oof=...)`
  + sentetik `Candidate(key="weighted_ensemble", family="ensemble", class_path="__ensemble__")`.

`engines/core.run_core_pipeline`: `score_reports`'tan **önce** ensemble raporu `reports`'a,
sentetik candidate `candidates`'e eklenir → `select_champion` (1-SE dahil) **onu da tartar**.
Ensemble sadece gerçekten daha iyiyse (1-SE bandı + `_FAMILY_COMPLEXITY["ensemble"]` en yüksek
karmaşıklık) şampiyon olur.

`engines/champion.refit_champion`: `candidate.key == "weighted_ensemble"` →
- her üye adayı **postprocess'siz** tüm train'de refit (`_refit_member` → `FittedModelPipeline`
  `postprocessor=None`).
- `FittedEnsemblePipeline(members, weights, pre_transform)` — `predict` = `Σ w_i · member_i.predict`.
- ensemble-düzeyi postprocess: harmanlanmış OOF üzerinde `build_postprocessor(...).fit(...)` → tek
  postprocessor tüm ensemble çıktısına.
- `BundleMetadata.ensemble = {"members": {k: w}}`, `model_key="weighted_ensemble"`.

`scoring/selection._FAMILY_COMPLEXITY`: `"ensemble"` en yüksek (tek model eşitse tek model kazanır).

## Determinizm (motto)
Tüm rastgelelik `np.random.default_rng(config.seed)` — bag örneklemesi, tie-break. Aynı
OOF + seed → aynı ağırlıklar.

## Kapsam dışı / sonra
- L2 stacking (meta-learner) → v1.1.
- Sınıflandırma GES (olasılık OOF) → v1.1.
- Pareto-frontier / çoklu ensemble (AutoGluon `expand_pareto_frontier`) → v1.1.

## Sonuç
- `autoragml/ensembling/` temiz GES + bagged-GES; `_vendor/` Caruana planı iptal.
- Ensemble tek-model şampiyonuyla aynı 1-SE seçiminde yarışır; kazanırsa `FittedEnsemblePipeline`.
- Sözleşme: `EnsembleSpec`, `RunConfig.ensemble`, `BundleMetadata.ensemble`.
