# ADR 0022 — k-fold bagging: şampiyon refit + HPO iç-fold düzeltmesi

**Durum:** Kabul · 2026-09-02

Kaynak: AutoGluon (*"you may not need any hyperparameter tuning... best accuracy by
num_bagging_folds + stacking/ensemble"*; bagged model = k child, tahminler ortalanır,
tek modelden güçlü); TabArena (post-hoc ensembling kritik); nested-CV literatürü
(*"a fixed tuning set risks overfitting to specific instances"*).

## Gözlem (1. dalga benchmark, `none` vs `light`)

`--hpo light` 6 datasetin **4'ünde `none`'dan kötü**, 2'sinde iyi — 10–15× maliyetle.
Ensemble california'da OOF/1-SE'de kazandı ama harici test'te genelleşmedi (val'a
hafif aşırı-uyum). Kök neden: (1) `light` tek iç holdout → yüksek varyanslı config
seçimi; (2) **bagging yok** → şampiyon tek model, %100 train — varyansı yüksek.

## Kararlar

### 1. Şampiyon refit **bagged** (varsayılan açık)
`refit_champion` artık tek model / %100 train yerine:
- `validators` ile **aynı splitter** (`resolve_splitter`) → k fold.
- HPO **bir kez** (`tuner.tune(work)`) → seçilen config tüm bag'lerde sabit.
- Her fold'un train'inde bir `FittedModelPipeline` (kendi feature-pipeline fit'i +
  kendi fold-içi early stopping). Postprocess **üye düzeyinde YOK**.
- Serving modeli = `FittedEnsemblePipeline(k üye, eşit ağırlık)` — `predict` = k üye
  tahmininin ortalaması.
- **Bagged OOF** = her fold-modelinin kendi held-out fold'undaki tahmini (concat) →
  ensemble-düzeyi postprocess bunun üzerinde fit edilir.

`k < 2` (küçük veri → splitter Holdout'a düşer) veya `bagging.enabled=False` veya
`n_rows > bagging.max_rows` → **tek model refit** (eski davranış, `refit_full` benzeri).

**v1 kapsam: yalnız regresyon + forecasting.** Sınıflandırmada `.predict()` hard-label
döndürür → k modelin ortalaması sürekli değer (ör. 0.6) verir. Olasılık ortalaması +
argmax → **v1.1** (sınıflandırma GES ile birlikte, olasılık OOF eklendiğinde).

### 2. GES ensemble üyeleri de bagged
`_refit_ensemble` her GES üyesini `bag_folds=k` ile refit eder → her üye kendi
`FittedEnsemblePipeline` (bag); GES `FittedEnsemblePipeline` bunları sarar (iç içe,
`predict` özyinelemeli). Ensemble-düzeyi postprocess GES harmanlanmış OOF'unda.

### 3. HPO iç-fold ≥ 2
`resolve_tuner`: `light` artık `inner_folds=2` (eskiden 1), `thorough` 3. Tek fold
config seçimi çok gürültülü (literatür).

### 4. `hpo_level` varsayılanı değişmedi
`light` kalır — ama düzeltilmiş (inner_folds=2) + bagging ile taban güçlü.
AutoGluon dersi: iyi defaults + bagging + ensemble ≈ HPO; kullanıcı `none`/`thorough` seçebilir.

## Sözleşme

`contracts/run_config.py`:
```python
class BaggingConfig(Contract):
    enabled: bool = True
    folds: int = 5          # AutoGluon 5-10 önerir
    max_rows: int | None = None   # üstünde tek-model refit (süre kaçış kapısı)
```
`RunConfig.bagging: BaggingConfig`. `BundleMetadata.ensemble` bagged şampiyonda
`{"bagged": true, "folds": k, "model": <key>}`.

## Determinizm
Splitter zaten seed'li; bag üye sırası fold sırası. Aynı veri+seed → aynı bagged model.

## Kapsam dışı / sonra
- **Repeated bagging** (`num_bag_sets` — AutoGluon çoklu tekrar) → v1.1.
- **L2 stacking** (bagged OOF → meta-learner) → v1.1 (artık altyapı hazır: temiz OOF var).
- Bagged model → `refit_full` collapse (çıkarım hızı) → opsiyonel, v1.1.

## Sonuç
- `refit_champion` / `_refit_ensemble` `bag_folds` alır; `_fit_pipeline` k-fold bagging yapar.
- Serving modeli `FittedEnsemblePipeline` (bag = eşit ağırlık, aynı model).
- `light` HPO 2 iç fold. Benchmark ile ölçülecek (`--hpo none` bagging etkisini izole eder).
