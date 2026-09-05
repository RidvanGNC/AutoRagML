# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) formatını,
sürümleme [SemVer](https://semver.org/lang/tr/)'i izler.

Her PR bu dosyaya bir satır ekler. `[Unreleased]` altında biriktirilir,
release'te tarih + sürüm ile başlığa taşınır ve git tag atılır.

## [Unreleased]

### Eklendi (v1.1 devamı — ADR 0033-0046)
- **Foundation modeller** (ADR 0033) — `[foundation]`/`[foundation-ts]` extra: **TabPFN** (tablo,
  in-context; lisans/token gerektirdiği için **varsayılan katalogda KAPALI**, kullanıcı kararı:
  "ücretli kısımlar listede olmayacak") + **Chronos-Bolt** (zero-shot forecasting, auth'suz).
  `models/foundation_tab.py` (`FoundationTabEstimator`) + `engines/timeseries/foundation_ts.py`
  (rolling-origin zero-shot OOF, fit yok) + `models/foundation_gate.py` (ayrı kapı, nöralden bağımsız).
- **L2 stacking** (ADR 0034) — saf stacking (yalnız L1 OOF matrisi Z, orijinal öznitelik yok);
  muhafazakâr `auto` (n_rows≥2000 + ≥4 çeşitli L1 ailesi); L2 OOF en iyi L1'i geçmiyorsa önerilmez.
  `ensembling/stacking.py` + `engines/stack_pipeline.py`.
- **Seçim temizliği** (ADR 0035) — `champion_refit_full` (holdout skorundan sonra train+holdout'ta
  yeniden fit, `EngineResult.finalize` closure) + klasik+reduction ortak forecasting GES
  (`engines/timeseries/joint_ensemble.py`) + aile-arası tie-break robustluğu.
- **Sınıflandırma GES + bagging** (ADR 0036) — olasılık-OOF tabanlı Caruana ensemble
  (`greedy_selection_proba`), bagged olasılık ortalaması + argmax; `predict_proba` sunmayan
  estimator'larda (RidgeClassifier vb.) bagging otomatik kapanır.
- **Açıklanabilirlik** (ADR 0037) — `RunResult.explain()`/`AutoRagML.explain()`: SHAP (kuruluysa,
  ağaç/linear) + model-agnostik permutation fallback (`explain/attribution.py`).
- **Panel holdout + seçim robustluğu** (ADR 0038, m3/m5 benchmark bulgusundan) — kararsız-CV
  filtresi (`_within_one_se`: `SE > 2·band` olan aday banttan dışlanır) + per-seri son-h holdout
  (heterojen panelde global cutoff yanlış hizalanıyordu). m3 champion sMAPE 44.7→16.3 düzeltmesi.
- **Forecasting seçim ince ayarı** (ADR 0039) — kesikli-baskın panelde yüzde-metrik promotion
  tavanı atlanır; 1-SE basitlik ikamesi en fazla **%3** göreli maliyete izin verir (bariz-daha-iyi
  foundation/nöral OOF feda edilmez).
- **Akademik model taraması** (ADR 0040, TabArena/AMLB/GIFT-Eval araştırması) — klasik forecasting:
  `auto_ces`/`auto_tbats`(opt-in)/`dynamic_theta`/`imapa`/`adida`. Tablo: **EBM** (glassbox GAM),
  **KNN**, SVR/SVC (opt-in), **NGBoost** (olasılıksal boosting). **TabICL** — TabPFN'e auth'suz
  in-context alternatif (clf+reg+forecasting-reduction).
- **TimesFM foundation-TS backend** (ADR 0041) — `foundation_ts.py` backend-agnostik hale getirildi
  (`_ForecastBackend` protokolü); Chronos'a ek olarak Google TimesFM 2.5 (zero-shot, auth'suz).
- **Foundation-TS OOF güven guard'ı** (ADR 0042, dev-benchmark bulgusundan) — Chronos/TimesFM
  ön-eğitim korpusu M3/M4/tourism benchmark'larını içerdiğinden rolling-origin OOF "ezberden"
  iyimser olabiliyordu (m3: OOF 8 → gerçek holdout 17). `_MAX_CV_WINDOWS` 2→4 + ince-kanıt/
  kontamine-OOF marj guard'ı (`foundation_ts`/`neural_ts` OOF-en-iyisi, güvenilir (≥3 fold) en iyi
  rakibi `max(SE)` marjıyla geçemezse düşürülür). tourism: -18.3% NO_IMPROVEMENT → +19.5% SUCCESS.
- **TiDE forecasting modeli** (ADR 0043) — `neuralforecast.models.TiDE`; kesikli panellerde büyük
  mimarilerden daha isabetli + ucuz olduğunu gösteren güncel araştırmadan (zaten kurulu bağımlılık).
- **Split-conformal `predict_interval()`** (ADR 0044) — `postprocessors/conformal.py`: kalibrasyon
  = mevcut şampiyon OOF'u (ek split/fit gerekmez), finite-sample düzeltmeli mutlak-residual kantili;
  `coverage` çağrı-zamanında override edilebilir. `per_group=True` (küçük örneklem → global'e düşer).
  Kapsam: `FittedModelPipeline`/`FittedEnsemblePipeline` (tablo + reduction-forecasting + bagged/GES).
  `RunResult`/`AutoRagML`/`LoadedChampion.predict_interval(data, coverage=None) -> (lower, upper)`.
- **Hiyerarşik reconciliation** (ADR 0045) — `RunConfig.hierarchy_cols`; `hierarchicalforecast.
  aggregate()` ile bottom panel üst düğümlerle genişletilir → **mevcut** `TimeSeriesCoreEngine`
  akışından geçer (yeni motor yok) → `MinTrace(wls_struct)` served tahminleri tutarlı hale getirir
  (çocuklar toplamı = ebeveyn). `engines/timeseries/hierarchical.py`.
- **Per-series şampiyon deneyi** (ADR 0046, deneysel — VARSAYILAN DEĞİL) — `structure=
  "per_series_champion"`: her seri kendi tek-üyeli segmenti (mevcut `run_segmented` mekanizması
  yeniden kullanılır). **Ampirik sonuç:** tourism alt-kümesinde `auto`/pooled +7.3% (41s) vs
  per-series **-4.1%** (227s, bir seri şampiyonu baseline'a düştü) — global/kümeleme yaklaşımının
  (ADR 0028) izole per-seri refit'ten üstün olduğu doğrulandı; özellik var ama önerilmiyor.
- **Foundation-TS/nöral-TS frekans kapısı** (ADR 0047, full-benchmark tourism gerilemesi + geniş
  literatür taraması) — Chronos/TimesFM/Moirai ön-eğitim korpusu M3/M4/tourism Monthly/Quarterly/
  Yearly benchmark'larını içeriyor ([2510.13654](https://arxiv.org/html/2510.13654v1): standart
  benchmark'ların yalnız %7'si temiz; sızıntı "32 puan MAPE"). ADR 0042 post-hoc guard'ı tam ölçekte
  yetersiz (SE veri arttıkça küçülüyor → eşik kolaylaşıyor). **Yapısal kapı:** `is_low_frequency_panel`
  (freq ∈ {M,Q,Y,A}) iken `run_foundation_ts_reports`/`run_neural_ts_reports` erken dönüş — aday
  havuza hiç girmez; `*_enabled="on"` ile açık opt-in bypass. Literatür: aylıkta klasik (ETS/ARIMA/
  Theta) ≈ en iyi DL, generic ML daha kötü (Monash Archive Tourism Monthly ETS MASE 1.28 ≈ DeepAR 1.25).
  Guard KALDIRILMADI — W/D/H frekanslarda foundation_ts hâlâ yarışıyor. TabPFN/TabICL (sentetik
  ön-eğitim) etkilenmez.
- **OLS reconciliation varsayılan** (ADR 0047) — `RunConfig.hierarchy_reconcile_method`
  (`ols` | `wls_struct`, ikisi de residual gerektirmez). FPP3 §11: OLS tourism'de MinT'i geçmişti.
- **`tourism_hier` benchmark dataset'i** (ADR 0047) — TourismLarge coğrafya ağacı (76 bölge → 27
  zone → 7 eyalet), `hierarchy_cols=["state","zone"]` + MinTrace(ols). `BenchmarkDataset.hierarchy_cols`
  + `run_forecasting` aktarımı. Grouped (coğrafya × amaç) yapının amaç boyutu atlandı (lineer).
- **Model kataloğu genişlemesi:** akademik tarama (üstte) + TiDE dahil toplam model sayısı ~40+.
  `tests/conftest.py::_lean_catalog` (autouse) — ağır/yavaş modeller (EBM/NGBoost/SVR/AutoTBATS/
  TabICL) test suite'inde varsayılan kapalı, `@pytest.mark.full_catalog` ile açılır (üretimde açık).
- **Benchmark harness genişlemesi** — 6→23 dataset (tablo 16, forecasting 7: M3/M4×3/M5/Tourism/ETT);
  `--profile dev` (12 dataset + sıkı veri capleri, ~90-120dk hızlı geliştirme sinyali) vs tam profil
  (23 dataset, ~16sa release doğrulaması).
- **Lisans:** proje **Apache License 2.0** altında yayımlanıyor (ticari kullanım serbest, telif/
  lisans bildirimleri korunmalı) — bkz. `LICENSE` + `NOTICE`.

### Eklendi (v1.1)
- **Nöral forecasting katmanı** (ADR 0032) — `[neural-ts]` extra: `neuralforecast`. Native panel
  yolu (ADR 0023 klasik deseni): `NeuralForecast.cross_validation` → OOF, `.fit`/`.predict` → serving.
  Modeller: NHITS · PatchTST · TFT · NBEATSx · iTransformer · TSMixer (+ `neural_search=True` →
  `AutoNHITS`/`AutoPatchTST`/`AutoTFT`, kütüphane HPO). `neural_enabled=auto` iken **yalnız GPU**
  (CPU'da CV çok pahalı); `neural_ts_min_series` (20) kapısı.
  - `engines/timeseries/neural_ts.py`: `run_neural_ts_reports` + `FittedNeuralForecaster` +
    `refit_neural_ts`. `run_core_pipeline(run_neural_ts=)`. `champion._neural_ts_bundle`.
  - Serving: `persistence.bundle` `champion_neural_ts/` sidecar (`NeuralForecast.save`/`load`).
  - **Bağımlılık:** neuralforecast → `torch≥2.9.1` → `torch 2.14.0+cu126` (`[neural]`+`[neural-nas]`+
    `[neural-ts]` tek venv'de çalışır ama kırılgan — üretimde ayrı venv önerilir).
- **Nöral mimari arama** (ADR 0031) — `[neural-nas]` extra: `pytorch-tabular`. `neural_search=True`
  ile iki aşama: **A)** aile taraması (MLP · GANDALF · FT-Transformer, kısa epoch) → en iyi ≤2 aile ·
  **B)** kazanan ailede koşullu mimari+HP arama (`n_layers` → `layer_width`/`residual`/`activation`/
  `dropout`/`lr`...), Successive Halving (epoch = fidelity). `small`/`full` uzay.
  - `models/neural_arch.py::TabularModelEstimator` (sklearn-uyumlu `TabularModel` sarımı).
  - `SearchDim.condition` (additive) — aile-özel HP'ler (`{"param":"family","eq":"gandalf"}`).
  - `fine_tuners/arch_search.py::ArchitectureSearchTuner` (Tuner protokolü, nested CV korunur;
    `neural_arch_search` dışı adayları fallback tuner'a devreder). `neural_search_budget_seconds`.
  - Serving: `persistence.bundle` `champion_neural/` sidecar (`TabularModel` joblib-picklable değil).
  - `neural_arch_search` yalnız `neural_search=True` + GPU/satır kapısı; RealMLP-default + GBDT
    havuzda kalır → arama bulamazsa GES güçlü default'a döner (regresyon garantisi).
  - Nöral aileler artık şampiyon refit'te **bagging yok** (tek-model — bagging pahalı).
  - `ArchitectureSearchTuner._cache`: arama dış CV fold başına DEĞİL bir kez koşar (sonraki
    fold'lar aynı mimariyi değerlendirir — Auto-PyTorch deseni, sızıntı yok).
  - **Doğrulandı (RTX 4060):** 3 aile fit/predict · bundle sidecar round-trip · e2e'de
    `neural_arch_search` leaderboard'da (küçük veride default'ları geçmedi — GES doğru modeli seçti).
- **Nöral tablo katmanı** (ADR 0030) — `[neural]` extra: `pytabkit` + `torch`. Katalog:
  `real_mlp` (RealMLP-TD, meta-tune edilmiş güçlü default), `tab_m` (TabM, MLP ansamblı),
  `real_tab_r` (RealTabR). Yeni engine yok — sklearn arayüzlü katalog girişleri.
  - `models/torch_env.py`: seed + determinizm + cudnn (`configure_torch`, idempotent; torch yoksa no-op).
  - `models/neural_gate.py`: çalışma-zamanı kapısı — `neural_enabled` (`auto`=GPU varsa / `on` / `off`)
    × satır bandı (`neural_min_rows`/`neural_max_rows`); pytabkit havuza girince sklearn `mlp` düşer.
  - `RunConfig` +`neural_enabled`/`neural_device`/`neural_determinism`/`neural_min_rows`/`neural_max_rows`;
    `tabular_fast` preset → `neural_enabled="off"`.
  - `EnvInfo.accelerator` (additive) — manifest'e torch/cuda sürüm + device + determinizm modu.
  - Determinizm varsayılanı `best_effort` (CPU tam deterministik; GPU manifest'e sürüm yazar);
    `configure_torch` RTX 40xx için `float32_matmul_precision("high")`.
  - `real_tab_r` opt-in-opt-in (`requires: [pytabkit, skorch, faiss]`) — `[neural]` extra yalnız pytabkit+torch.
  - **Doğrulandı (RTX 4060):** torch 2.6.0+cu124 GPU tespiti · e2e'de RealMLP GBDT'yi geçti
    (nonlinear veri RMSE 0.39 vs 0.79) · GES nöralleri aldı · serving + `accelerator` manifest OK.
  - **v1 sınırı:** `FeaturePipeline` nöral adaylara da tam uygulanır (pytabkit çift-transform) → v1.1.
  - FT-Transformer / mimari arama → ADR 0031.

### Düzeltildi
- **Klasik forecaster serving — arbitrary gelecek penceresi** (ADR 0029, m5 benchmark bulgusundan):
  `FittedClassicalForecaster.predict` sabit `sf.predict(h=self._h)` çağırıyordu → yalnız fit sonrası
  ilk `h` adım. Şampiyon `train − holdout`'ta fit edilince (ADR 0020 holdout carve) gerçek gelecek
  bu pencerenin ötesinde kalıp tarih-merge tutmuyor, tüm satırlar "son değer" fallback'ine düşüyordu
  (m5: 3 farklı klasik şampiyonda byte-identik wMAPE 105.2). Artık `predict` istenen `ds` aralığını
  kapsayacak kadar ileri tahmin eder (`_horizon_for`, `[h, h·24+366]` clamp). `__init__` `freq` alır.

### Değişti
- **Guardrail `prediction_negative` serving-clip'e duyarlı** (ADR 0027, m5 benchmark bulgusundan):
  `postprocess` negatif-olmayan kırpma tabanı (`clip.lower ≥ 0` veya `auto_nonneg` + forecasting/
  regresyon + `target_min ≥ 0`) uygulayacaksa, küçük negatif OOF tahminleri **artık karantinaya
  almıyor** (served tahmin garanti `≥ 0`). İstisna: negatif oranı > %50 → miskalibre → yine karantina.
  OOF metrikleri ham tahminde kalır (ADR 0017 serving-only kilidi korunur). m5'te bu, en iyi
  tahmincilerin (`classical_ensemble` wMAPE 77.2, `auto_ets` 77.8) yanlış elenmesini düzeltir.
  `prediction_health` → `n_pred` + `frac_negative` (additive); `evaluate_guardrails(serving_clip_lower=)`.

### Eklendi
- **Segmented champion** (ADR 0028, gap analizi Gap #5 — M5 winner hiyerarşik havuzlama deseni):
  `plan.structure == "per_group_champion"` artık pooled'a düşmez — TS engine serileri **SBC
  intermittency sınıfına** göre segmentlere böler (`_resolve_segments`; küçük segmentler ADI
  ekseninde komşuya birleşir, ≤ `segment_max_count`), her segment için tam çekirdek pipeline
  (`run_core_pipeline`) koşar. Serving `FittedSegmentedPipeline` her seriyi grup kimliğiyle kendi
  segmentinin şampiyonuna yönlendirir (bilinmeyen kimlik → en büyük segment). Şampiyon tek
  `ModelBundle` (`pipeline` = `FittedSegmentedPipeline`, segment→model haritası
  `adaptive_plan_summary`). Sözleşme: `AdaptivePlan.segments` + `SegmentSpec`,
  `DynamicsConfig.segment_min_series`/`segment_max_count`/`segment_sparse_min_frac` (additive).
  `engines/segmented.py`. Birleşik scoreboard: sentetik `segmented` satırı (boyut-ağırlıklı
  metrikler) + `segment::model` satırları.
  **Kesikli-baskınlık kapısı:** yalnız `(intermittent + lumpy) / total ≥ 0.5` ise segmentle —
  düzgün-baskın panelde pooled cross-series öğrenme hard serilere yardım ediyor (tourism +10.3%
  segmentleyince +8.9%'a düşüyordu → kapı ile geri alındı).
  **v1 sınırı:** çok-seviyeli hiyerarşi (store/dept) + segment-arası ensemble + k-means kümeleme → v1.1.
  **Benchmark (m5_subset, `--hpo none`):** ilk kez seasonal-naive'i geçti — test wMAPE 105.2→83.3
  (**+16.4%**), 3 segment (smooth+erratic/intermittent/lumpy), süre 5679s→3055s.
- **Recursive multi-step reduction** (ADR 0026 Bölüm B, gap analizi Gap #4 — M5/AutoGluon-TS deseni):
  `RunConfig.forecast_reduction: Literal["direct", "recursive"] = "direct"`.
  - `build_reduction_features(strategy="recursive", max_lag=None)`: `shift(1)` tabanı, lag `1..k_max`
    (`k_max = max(h, 3s, 12)`), rolling/ewm/min-max `shift(1)` üzerinde; `y_sdiff_ref` üretilmez.
  - Model **1-adım-ileri** eğitilir; **CV recursive-`h`**: `run_recursive_reports` rolling-origin
    fold'da test bloğunu adım-adım tahmin eder (özellikler yeniden kurulur, tahmin geri beslenir) →
    OOF birikimli-hata skoru serving davranışını ölçer.
  - **Serving:** `FittedRecursivePipeline` (`Predictor` protokolü, `__slots__`, joblib-picklable) —
    her seri son `horizon` satır recursive, kalan `NaN`. `RecursiveRecipe` saf param (`TaskSpec` +
    `season` + `add_calendar` + `horizon`).
  - `weighted_ensemble` recursive modda devre dışı (ansambl refit direct özellik kurar); bagging yok
    (tek 1-adım model refit).
  - Sözleşme: `run_core_pipeline(recursive=, recursive_season=)`, `refit_champion(recursive_season=)`.
  - `tests/unit/engines/test_reduction.py` +1 (recursive lag leakage-safe), `test_engines_e2e.py` +1.
- **Seasonal target differencing** (ADR 0026 Bölüm A): `candidate_ops` `target` grubuna
  `seasonal_difference` seçeneği — forecasting + mevsim ≥ horizon + trend/mevsim gücü → **varsayılan**.
  Eğitim hedefi `y_t − y_{t−s}` (leakage-safe: `s ≥ h` iken `y_{t−s}` train aktüeli); tersine çevirme
  `pred + {target}_sdiff_ref`. `TargetTransform.forward/inverse` opsiyonel `ref` arg;
  `FittedModelPipeline.target_ref_col` slotu; `runner`/`champion` warmup (ref=NaN) satırlarını atar.
- **Zengin reduction özellikleri** (ADR 0025 — MLForecast paritesi, gap analizi Gap #2):
  `build_reduction_features` ~15 → ~40 özellik, tümü leakage-safe (`shift ≥ horizon` veya takvim):
  - **takvim/tarih:** `month`/`quarter`/`dayofweek`/`dayofyear`/`weekofyear` + `is_month_start/end` +
    döngüsel `sin/cos(month)`, `sin/cos(dayofweek)` (gelecek tarihler için bilinir → sızıntı yok)
  - **mevsim-hizalı lag** `y_slag_{H+k·s}` (`H = ceil(h/s)·s`) + **mevsimsel rolling** (aynı-mevsim ort/std)
  - rolling'e **min/max** + pencereye `season` eklendi
  - **fark özellikleri:** `y_diff1_lag_h` (`shift(h)−shift(h+1)`), `y_diffs_lag_h` (`shift(h)−shift(h+s)`)
  - `season` `TimeSeriesCoreEngine`'den `_season_length(profile)` ile geçer; `pre_transform` taşır
  - gerçek (seasonal) target differencing transform + recursive multi-step → v1.1/ADR 0026
  - `tests/unit/engines/test_reduction.py` — 6 test (tüm özellikler leakage-safe, takvim warmup'sız).
- **Klasik model ansamblı (EAT) + Tweedie objective** (ADR 0024, SOTA gap analizinden):
  - `run_classical_reports` per-model raporlara ek olarak **`classical_ensemble`** üretir —
    aynı `cross_validation` OOF matrisinde GES/bagged-GES (M3 winner Theta, M4 winner forecast-
    combination; ETS+ARIMA+Theta = EAT). Cutoff-hizalı → GES sorunsuz.
  - `FittedClassicalForecaster` çok-model + ağırlık: `predict` = model-başı forecast kolonlarının
    ağırlıklı ortalaması. `refit_classical_ensemble` üyeleri tek `StatsForecast`'ta fit eder.
  - **Tweedie:** `planner._model_hints` — `intermittency_summary`'de düzensiz pay ≥ %50 →
    `AdaptivePlan.model_hints` (`lightgbm:tweedie`, `hist_gbm:poisson`, `xgboost:reg:tweedie`).
    `models.apply_model_hints` reduction GBDT `default_params`'a merge eder. "Magic multipliers"
    ≈ `postprocess.calibrate="multiplicative"` (zaten var).
  - `_season_length` düzeltmesi: freq'in doğal periyodu önce (günlük→7); yıllık (365) AutoARIMA'yı
    patlatıyordu.
  - Sözleşme: `AdaptivePlan.model_hints`. `run_classical_reports` → `(reports, extra_candidates)`.
  - `tests/unit/dynamics/test_planner.py` +2 (Tweedie ipucu), `test_classical_forecasting.py` +1.
- **Native classical forecasting** (ADR 0023): benchmark 2. dalga bulgusu — `m3_monthly` şampiyon
  sMAPE 47 vs seasonal-naive 14 (klasik modeller kataloğda ama reduction pipeline'ından geçemiyordu).
  - `engines/timeseries/classical.py` — `family∈{statistical,intermittent}` adaylar Nixtla
    `StatsForecast` native yolundan: `cross_validation` → OOF (rolling-origin), `fit`/`predict` → serving.
  - CV pencere sayısı seri uzunluğuna uyarlanır (ilk eğitim penceresi ≥ 2·season); kısa seriler
    filtrelenir; `SeasonalNaive` fallback; `season_length` mevsimsel modellere oto-enjekte.
  - `TimeSeriesCoreEngine` iki yolu birleştirir (`run_core_pipeline(run_classical=…)`); şampiyon
    her iki aileden olabilir. `refit_champion` klasik şampiyonu `_classical_bundle`'a yönlendirir.
  - `RunConfig.classical_forecasting: bool = True` (büyük panelde yavaş — kapatılabilir).
  - **v1 sınırı:** GES ensemble klasik modelleri dışlar (cutoff-tabanlı OOF ≠ fold-tabanlı); v1.1.
  - `datasetsforecast` dev bağımlılığı (benchmark 2. dalga: M3/TourismLarge/M5).
  - `tests/unit/engines/test_classical_forecasting.py` (3).
- **k-fold bagged şampiyon refit** (ADR 0022): benchmark bulgusu — `--hpo light` 6 datasetin 4'ünde
  `none`'dan **kötü** (tek iç fold → val'a aşırı-uyum) + bagging yoktu.
  - `refit_champion`: tek model / %100 train yerine `bagging.folds` (5) fold-modeli; serving = ortalama
    (`FittedEnsemblePipeline` eşit ağırlık). Bagged OOF postprocess'e girer. `k<2` / `bagging.enabled=False` /
    `n_rows>bagging.max_rows` / **sınıflandırma** → tek model refit (`refit_full` benzeri).
  - GES `weighted_ensemble` şampiyonda üyeler de bagged (iç içe `FittedEnsemblePipeline`).
  - `resolve_tuner`: `light` HPO artık **2 iç fold** (eskiden 1); `thorough` 3.
  - Sözleşme: `RunConfig.bagging` (`BaggingConfig`), `BundleMetadata.ensemble={"bagged":true,...}`,
    `engines.model_pipeline.Predictor` protokolü.
  - **v1 sınırı:** sınıflandırma bagging (olasılık ortalaması + argmax) → v1.1.
  - `tests/unit/engines/test_bagging.py` — 5 test.
- **`ensembling/` katmanı** (ADR 0021): Caruana greedy ensemble selection (+ bagged-GES).
  - `greedy.py` — saf numpy GES: her tur validation'ı en çok iyileştiren modeli **tekrarlı** ekle;
    6-ondalık tie yuvarlama + en düşük indeks tie-break + **use_best**. `bagged_greedy_selection`
    (model alt-kümelerinde tekrarlı GES, seed'li, ağırlık ortalaması — Caruana 2006). Temiz implementasyon
    (akademik makaleden; `_vendor/` kopyalama planı iptal — NOTICE güncellendi).
  - `build_weighted_ensemble(reports, candidates, config, task, profile)` → sentetik `weighted_ensemble`
    `ValidationReport` + `Candidate` + `EnsembleSpec`; `engines/core`'da tek-model şampiyonuyla aynı
    **1-SE seçiminde** yarışır (`_FAMILY_COMPLEXITY["ensemble"]=5`).
  - `engines/champion._refit_ensemble` — her üye **postprocess'siz** refit → `FittedEnsemblePipeline`
    (`Σ wᵢ·memberᵢ.predict` + tek ensemble-düzeyi postprocess).
  - **v1: regresyon + forecasting** (OOF nokta tahmini). Sınıflandırma GES (olasılık OOF) → v1.1.
- **Model kataloğu:** `mlp` (sklearn `MLPRegressor/Classifier`) **etkinleştirildi** — ilk "neural" adım;
  `scale: true` → `build_estimator` wrap'e `StandardScaler` eklendi. `Candidate.scale` alanı.
- Sözleşme: `EnsembleSpec`, `RunConfig.ensemble` (`EnsembleConfig`), `BundleMetadata.ensemble`,
  `Candidate.ensemble_members`/`scale`. `EngineStatus`: ensemble/reduction mesajları artık PARTIAL yapmıyor
  (yalnız gerçek sorunlar).
- `tests/unit/ensembling/` (9) + `tests/unit/engines/test_ensemble_integration.py` (3) + `test_stateless_pickle.py` (6) — 251 test toplam.
- **`scripts/benchmarks/`** — gerçek verisetlerinde uçtan uca koşum + harici test setinde naive baseline
  karşılaştırması (`python -m scripts.benchmarks.run`). 1. dalga: 6 OpenML/sklearn tablo verisi
  (regresyon/ikili/çok-sınıf). Sonuç: **6/6 SUCCESS** (naive'i +59…+755% geçti, OOF↔holdout tutarlı);
  covtype `ColumnDropper` pickle bug'ını ortaya çıkardı (yukarıda düzeltildi). Detay: `scripts/benchmarks/RESULTS.md`.

### Düzeltildi
- **`preprocessors/stateless`: fitted op'lar joblib/pickle ile serialize edilemiyordu** — `ColumnDropper.fit`
  (ve date_expand/log1p/hashing) **yerel closure** (`_fn`) döndürüyordu → `save_bundle` kolon-düşüren
  pipeline'larda `PicklingError` (benchmark covtype ortaya çıkardı). Tüm op'lar modül-düzeyi `__slots__`
  callable sınıflarına çevrildi.
- **CI macOS:** `brew install libomp` adımı eklendi — LightGBM'in OpenMP runtime'ı olmadan
  import edilemiyordu → registry `lightgbm`'i düşürüyor → 7 test (`test_tuners`, `test_estimator`,
  `test_registry`, `test_runner`) macos-latest'te patlıyordu.
- `models/registry`: "paket kurulu ama import patlıyor" durumu artık gerçek hatayı log'a yazıyor
  (`_import_hint`), sadece "importable değil" demiyor.
- CI: `actions/checkout@v4→v5`, `actions/setup-python@v5→v6` (Node 20 deprecation uyarısı; yeni sürümler Node 24 native).
- `test_timeseries_and_leakage.py`: `DatetimeIndex + pd.Timedelta` aritmetiği ayrı `date_range` ile değiştirildi (NumPy 2.5 "generic timedelta unit" DeprecationWarning).

### Eklendi
- **`interfaces/` katmanı kodlandı** (ADR 0020 — son katman): `Orchestrator` + `AutoRagML` facade + `autoragml run` CLI.
  - `Orchestrator.run(data, config, *, resolution=None, tracker=None) -> RunResult` — tek akış:
    `create_run_dir → tracker.start_run → [io] load_dataset → [analyze] analyze → [holdout_split] →
    [engine] select_engine + runner → [holdout_score] → champion.metrics_holdout (test'e TEK dokunuş) →
    [persist] persist_run → [report] write_reports → manifest'i tam timeline ile yeniden yaz → tracker.log_* + end_run`.
    io/analyze fail-fast; engine `FAILED` → akış devam (minimal manifest+rapor, CLI exit 1). Her stage `TimelineEntry`.
  - `interfaces/holdout.py` — `split_holdout` (yetersiz veri → yok+WARNING; tabular seed'li rastgele;
    TS son `min(horizon, n_periods−1)` dönem, `[group,time]` stable sıralı, `shift(horizon)` leakage-safe) +
    `score_holdout` (TS'de tüm sıralı frame'de predict + mask).
  - `AutoRagML(preset=, config_file=, **overrides).fit(data, *, target, time_col=, group_col=) -> RunResult`;
    `.leaderboard/.predict/.explain/.champion/.manifest`; `AutoRagML.load(bundle_path) -> LoadedChampion`.
  - `cli.main` — `autoragml run --data --target [...]`; leaderboard(10) + şampiyon + promotion + çıktı dizini stdout.
  - `agent_tools.TOOLS` — `autoragml_run` JSON-schema (v2 executor yok).
- `tests/unit/interfaces/` — 13 test (holdout carve tabular/TS/skip, uçtan uca akış + artifact ağacı +
  timeline, facade fit/predict/explain/load, CLI smoke).
- **v1 katman iskeleti tamamlandı** — `contracts`…`interfaces` (ADR 0008-0020).
- **`reporters/` + `tracking/` katmanları kodlandı** (ADR 0019 — grup: bağımlılıksız çıktı/gözlem sink'leri).
  - `reporters.write_reports(engine_result, manifest, paths, ...)` → `paths.reports/`: `run_report.html`
    (**her zaman**, tek dosya, CDN yok, `html.escape`), `model_card.md` (**her zaman**, Mitchell bölümleri
    + `TODO` placeholder), `leaderboard.csv` (**her zaman**). `plots/*.png` yalnız `[report]` extra varsa
    (yoksa WARNING, akış kırılmaz).
  - `tracking`: `Tracker` protokolü (`start_run/log_params/log_metrics/log_artifact/end_run`);
    `resolve_tracker(config, run_dir)` → `NullTracker` (none) · `JsonlTracker` (varsayılan, `events.jsonl`
    + `summary.json`, bağımlılıksız/ağsız) · `MlflowTracker` (`[tracking]` extra — yoksa `ConfigError`).
- Sözleşme: `persistence.RunPaths.tracking` alt dizini; `FittedModelPipeline.estimator` salt-okunur property.
- `report` extra'dan `jinja2` çıkarıldı (HTML bağımlılıksız); `dev` + mypy override'a `matplotlib`/`mlflow`.
- `tests/unit/{tracking,reporters}/` — 9 test (JSONL tam döngü + determinizm, resolver, HTML self-contained,
  model card bölümleri, plot üretimi).
- **`persistence/` katmanı kodlandı** (ADR 0018): `persist_run(config, dataset, engine_result, ...) ->
  (RunPaths, RunManifest)` — yan etkisi olan katman.
  - `paths.create_run_dir` → `<output_dir>/<DDMMYYYY>_<proje>_outputs/<run_id>/` (`run_id = UTC
    %Y%m%dT%H%M%SZ`; çakışma → `-01`; dolu dizin sessizce ezilmez) alt: `models/ evaluation/ reports/ config_snapshot/`
  - `bundle.save_bundle`/`load_bundle` — **joblib tek dosya** (canlı `pipeline` + metadata + `saved_env`);
    yüklemede `format_version` + sklearn/lightgbm sürüm sapması kontrolü; pickle güvenlik uyarısı
  - `manifest.build_manifest` — fingerprint + config snapshot + env (python/os/paket sürümleri/git commit) +
    data snapshot + timeline → reprodüksiyona yeter
  - `dump.write_json` deterministik (`sort_keys`, sonda newline); `persist_evaluation` / `persist_config_snapshot`
    (**`.env` asla kopyalanmaz**)
- Sözleşme: `exceptions.PersistenceError`. mypy override'a `joblib.*` eklendi.
- `tests/unit/persistence/` — 9 test (run_id formatı, klasör düzeni + çakışma soneki, bundle round-trip
  aynı tahmin, format guard'ları, deterministik JSON, `persist_run` tam ağaç + sırsızlık).
- **`postprocessors/` katmanı kodlandı** (ADR 0017): `build_postprocessor(cfg, profile, task) -> Postprocessor`
  → `.fit(y_true?, y_pred?) -> FittedPostprocessor` (`ModelBundle.pipeline`'a gömülür).
  - Sıra: `calibrate → clip → round → business_rule` (`_POST_ORDER`)
  - `calibrate` (yalnız `champ_report.oof`): `additive_bias` · `multiplicative` (clamp'li); OOF yoksa atlanır. `linear`/isotonic → v1.1
  - `clip`: `auto_nonneg` (hedef min ≥ 0 + regresyon → `lower=0`) · opsiyonel `auto_upper` (`pXX·mult`) · açık sınır kazanır
  - `round`: `off`/`nearest`/`threshold` (DemandSensing eşikli)/`ceil`/`floor`
  - **leakage-safe** (ADR 0011): fit yalnız `refit_champion` içinde, OOF üzerinden
  - `FittedModelPipeline` `postprocessor` slotu alır; `predict()` target-inverse sonrası uygular (None ise atlar)
- Sözleşme: `RunConfig.postprocess` (`PostprocessConfig` + `ClipConfig`/`RoundConfig`/`CalibrateConfig`/`ConformalConfig`);
  `BundleMetadata.postprocess_summary`.
- `tests/unit/postprocessors/` — 18 test (auto_nonneg, clip/round/calibrate, sıra, v1 guard'lar, business_rule immutability, engine e2e gömme).
- **v1 sınırı:** `conformal.enabled` + `apply_in_validation` → `ValueError` (v1.1: split-conformal + `predict_interval`).
- **`engines/` katmanı kodlandı** (ADR 0015): `select_engine(task, config)` + `InProcessRunner`.
  - `core.run_core_pipeline` — ortak akış: `build_plan` → `resolve_candidates` → `run_validation_suite(tuner=resolve_tuner)` → `score_reports` → `refit_champion` (tüm train; ES modelde n_estimators = validation best_iteration medyanı) → `ModelBundle`
  - `TabularCoreEngine` / `TimeSeriesCoreEngine`
  - `timeseries/reduction.py` — **leakage-safe** lag/rolling/ewm özellikleri (`shift ≥ horizon` → h-adım direkt tahminde test yalnız train `y`'sini görür); recursive multi-step v1.1
  - `model_pipeline.FittedModelPipeline` — `ModelBundle.pipeline` runtime nesnesi (`pre_transform` → feature pipeline → estimator → target inverse); TS'de reduction FE predict'te yeniden uygulanır
  - `runners/` — `EngineRunner` protokolü + `InProcessRunner` (çökme → `status=FAILED`)
- Sözleşme: `ValidationReport.prediction_health` + `.oof` (`OOFArrays`), `ScoreRow.family` (scoring turunda eklendi); `io.materialize_frame`.
- `tests/unit/engines/` — 9 test (reduction leakage-safety + grup izolasyonu, tabular/TS uçtan uca, predict, runner failure wrap).

### Değişti
- pytest: `ConvergenceWarning` + `eval_set` DeprecationWarning filtrelendi; engine e2e testleri `hpo_level=none` ile hızlandı.
- **v1 sınırı:** `per_group_champion` planlanır, pooled ile ilerlenir (per-group refit v1.1).
- **`scoring/` katmanı kodlandı** (ADR 0014): `score_reports(reports, candidates, config, task, profile) -> SelectionResult`.
  - `guardrails.py` — quarantine: non-finite metrik · prediction_health (negatif/scale-ratio; negatif yalnız hedef≥0) · metrik tavanları · model×scenario blocklist · leakage FAIL
  - `selection.py` — **1-SE kuralı** (default): birincil metrikte en iyinin noise_floor bandındaki en basit/ucuz aday; `best` alternatifi; `promotion` (mutlak eşikler); `class_weighted_score` v1'de bilgilendirme (per-class SE yok → v1.1)
  - `comparison_tests.py` — MCB ortalama rank + Diebold-Mariano (HLN düzeltmesi, scipy); forecasting + ≥3 fold
  - `__init__.build_scoreboard` — `noise_floor` (SE medyanı) + `selection_bias_bound = σ√(2lnK)`
- Sözleşme: `RunConfig.promotion` (`PromotionConfig`), `GuardrailConfig` prediction eşikleri, `ValidationReport.prediction_health` + `.oof` (excluded `OOFArrays`), `ScoreRow.family`.
- Çekirdek dep: `scipy>=1.11` (zaten sklearn/lightgbm transitif). `validators.frame_ops` — `OOFArrays` + `prediction_health`.
- `tests/unit/scoring/` — 18 test.
- **`fine_tuners/` katmanı kodlandı** (ADR 0013): `validators.Tuner` protokolünü gerçekler.
  - `random_search.py` — `RandomSearchTuner`: random search + `Candidate.fidelity` varsa **Successive Halving** (`halving.py`, eta=3, doğrulanmış formül). Kooperatif bütçe + ADR 0008/1 projeksiyon uyarısı
  - `optuna_backend.py` — `OptunaTuner` (TPE, `[hpo]` extra). v1: sabit fidelity (ara-adım pruning callback v1.1)
  - `inner_eval.py` — `build_inner_splits` + `evaluate_trial` (yalnız dış-fold train içinde)
  - `space.py` — `SearchDim` örnekleme; `resolve_tuner(config)` (level+backend)
- `RunConfig.hpo_backend` (`HpoBackend`); `TunerOutcome.tuning_result` (`contracts.TuningResult`).
- `validators/frame_ops.py` — paylaşılan fold-frame yardımcıları (runner + fine_tuners).
- Dev dep: `optuna` (backend testleri için).
- `tests/unit/fine_tuners/` — 14 test.
- **`validators/` katmanı kodlandı** (ADR 0010/6 + 0011 + 0013): split sınırını yöneten tek yer.
  - `splitters.py` — Holdout / KFold / StratifiedKFold / GroupKFold / **RollingOrigin** (genişleyen train, sabit horizon, zaman sızıntısı yok) / FixedWindow + `resolve_splitter` (`split_policy` kısmi override + görev tabanlı; küçük veri → holdout)
  - `runner.py` — nested CV: `Tuner.tune` iç resample (`DefaultTuner` = plan varsayılanları, `nested=False`) → `FeaturePipeline.fit_transform(train)` + `apply(test)` → `TargetTransform` fit(train_y) → `build_estimator` + fold-içi iç-val early stopping (lightgbm/sklearn-HGB) → `compute_metrics`. OOF metrik + fold'lar arası SE. `run_validation_suite` (aynı split, çöken aday atlanır)
  - `leakage_checks.py` — overlap (satır/zaman/grup) + preprocessing (provenance) → `LeakageReport`
- **`scoring/metrics/` kodlandı** — sMAPE/MAPE/WMAPE/RMSE/MAE/bias/CSL + accuracy/f1_macro/balanced_accuracy; `compute_metrics(y, ŷ, task)`, `default_primary_metric`, `lower_is_better`.
- `RunConfig.validation` (`ValidationConfig`).
- `tests/unit/validators/` — 19 test (rolling-origin zaman sızıntısı, resolve, leakage, nested tuner, suite skip).
- **`models/` + registry kodlandı** (ADR 0012): `resolve_candidates(config, task) -> [Candidate]`.
  - `models/catalog/*.yaml` pakete gömülü (tabular/baselines/timeseries); wheel'e dahil
  - `registry.py` — YAML deep-merge + `requires` (`find_spec`) + `class_path` importable kontrolü → eksikse atla + tek WARNING; `enabled:false`; modalite + task filtresi; entry-points `autoragml.models` ikincil
  - `estimator.py` — `resolve_class_path` (forecasting → reduction `regression` path'i) + `build_estimator` (param merge; `wrap` → imputer+model Pipeline)
  - katalog: sklearn linear/ridge/lasso/elastic_net/logistic/RF/ET/hist_gbm/mlp + lightgbm (çekirdek) + xgboost/catboost (ops.) + Dummy baseline + statsforecast (ops.)
- `RunConfig.model_catalog_override` (zaten vardı) artık `registry` tarafından tüketiliyor.
- `tests/unit/models/` — 19 test.

### Değişti
- Repo-kök `configs/model_catalog/` stub YAML'ları kaldırıldı (katalog pakete taşındı); dizin = kullanıcı override örneği.
- **`preprocessors/` katmanı kodlandı** (ADR 0011): `FeaturePipeline.from_plan(plan, candidate_choices)` → leakage-safe dönüşüm zinciri.
  - `base.py` — `SklearnColumnTransform` (kolon alt kümesine sklearn transformer; `cross_fitted` → `fit_transform` cross-fitting / `apply` full-train)
  - `catalog.py` — op → transform: impute · onehot/ordinal/**target_encode**(sklearn iç cross-fitting)/hashing · scale · yeo_johnson · quantile
  - `stateless.py` — drop / date_expand / log1p (işaretli) / hashing (CRC32 deterministik)
  - `pipeline.py` — sıralı zincir (drop→recipe→date_expand→impute→encode→numeric); `fit_transform` her adımda cross-fitting'i korur
  - `target.py` — `TargetTransform` (none/log1p/yeo_johnson/quantile; forward/inverse)
- `autoragml/transform.py`: `Transform.fit_transform` protokole eklendi + `BaseTransform`.
- `tests/unit/preprocessors/` — 16 test (target-encode sızıntı: cross-fit ≠ full-train, test target'ına dokunmaz; unseen kategori; target roundtrip).

### Değişti
- `analyzers.analyze`: modality=timeseries ama geçerli `time_col` yoksa **tabloya düşer** (forecasting zaman ekseni olmadan anlamsız).
- mypy override: `sklearn.*` (py.typed yok).
- **`dynamics/` katmanı kodlandı** (ADR 0007+0010+0015): `build_plan(DataProfile, TaskSpec, RunConfig) -> AdaptivePlan`.
  - `planner.py` — `committed_ops` (yapısal drop / date_expand / encode-by-cardinality / impute / recipe refleri) + `candidate_ops` (heavy_tailed_numeric + target grupları; hedef negatifse log1p elenir) + `structure` auto (per_group_champion eşiği) + `row_policies` (intermittent_augment/filter_low_activity/coldstart_split) + `regimes` (scenario_2) + `family_policy`
  - `recipes/` — registry: `@register_recipe` + `load_recipe_paths` + entry-points (`autoragml.recipes`); isim çakışması → `RecipeError`; `validate_recipes` plan-zamanı fail-fast
  - `autoragml/transform.py` — `Transform` / `FittedTransform` protokolleri (ADR 0011) + `StatelessFitted`
- `RunConfig.dynamics` (`DynamicsConfig`); `exceptions.RecipeError`.
- `tests/unit/dynamics/` — 16 test.

### Değişti
- `01_contracts.md`: FittedTransform protokolü `autoragml/transform.py`'a taşındı (contracts pandas'a bağlanmasın diye).
- **`analyzers/` katmanı kodlandı** (ADR 0010): `analyze(dataset, config) -> (DataProfile, TaskSpec)`.
  - `modality.py` — hint/time_col/layout/datetime-dup ile tablo↔zaman serisi
  - `profiling.py` — `ColumnProfile` (raw_dtype + special_types + semantic_role + flags + duplicate_of), numpy skew/kurtosis, `TargetSummary`
  - `task_inference.py` — 7 task; hedef kuralları; `task_hint` çelişki uyarısı
  - `timeseries.py` — `pandas.infer_freq` (+ anchored-weekly fallback) + freq→periyot sözlüğü + numpy-ACF mevsimsellik + OLS-R² trend; per-series ADI/CV² + SBC sınıf (`intermittent.py`); seasonality/trend toplulaştırılmış seri üzerinde
  - `quality.py` / `leakage.py` — dataset kalite bayrakları; yumuşak sızıntı şüphesi (WARNING) + `ColumnProfile.flags`'e `leakage_suspect`
  - `_frame.py` — lazy kaynak → örneklem + düşük güven
- `RunConfig.analyzers` (`AnalyzerConfig`: `ThresholdConfig` + `TimeSeriesAnalyzerConfig` + `profiling_sample_rows`); `TimeSeriesProfile.intermittency_summary` eklendi.
- `logging.py`; `tests/unit/analyzers/` — 23 test.

### Not
- `statsmodels` yok: `stationarity_pvalue=None`; `classification_scheme=kh` → SBC + uyarı (takip).
- **`io/` katmanı kodlandı** (ADR 0009): `load_dataset(src, config) -> Dataset`.
  - `sources.py` — DataFrame/csv/tsv/parquet/dizin/`DbSource` çözümleme + boyut yoklama
  - `fingerprint.py` — **strict** sıra-bağımsız çoklu-küme hash (sum+xor+count over `hash_pandas_object`, sort'suz, streaming) + **fast** structural
  - `layout.py` — wide tespiti + `melt` (eager) + long/single_series/n-a; lazy+wide → hata
  - `lazyframe.py` — `LazyFrame` (chunk akışı); `db.py` — SQLAlchemy adaptörü (lazy import)
  - otomatik eager/lazy (`io.eager_max_bytes`, varsayılan 1 GiB)
- `logging.py` — kütüphane logger yardımcısı (kök logger'ı yapılandırmaz).
- `tests/unit/io/` — 25 test (fingerprint sıra-bağımsızlığı/hassasiyeti, layout, eager==lazy fingerprint, DB/sqlite, boş/uzantı hataları).

### Değişti
- **`requires-python` `>=3.11` → `>=3.12`** (modern tip stub'ları — numpy 2.5 PEP 695). Ruff/mypy `py312`; CI 3.12/3.13.
- `Dataset.schema` → **`Dataset.dtypes`** (pydantic `schema()` çakışması; alias yerine temiz ad).
- Dev dep: `pandas-stubs`, `sqlalchemy`. Yeni extra: `db`. mypy override: `pyarrow.*`.
- ADR 0016: config çözümleme — katmanlı merge + alan-düzeyi provenance + preset `extends` + `Settings` (`.env`) + preset konumu (pakete gömülü).
- **`config/` katmanı kodlandı**: `resolve_run_config()` → `ConfigResolution`; `merge.py` (deep-merge + provenance), `presets.py` (`extends` zinciri + döngü tespiti), `loaders.py`, `settings.py` (dotenv parser + `SecretStr` çözümü).
- `contracts/config_resolution.py` — `ConfigResolution` (`config` + `provenance` + `layers`).
- Yerleşik presetler `src/autoragml/config/_presets/`: `tabular_fast`, `timeseries_rolling`, `demandsensing` (RunConfig alan adlarıyla birebir). Wheel'e dahil (doğrulandı).
- `tests/unit/config/` — 22 test (merge, preset extends, settings, uçtan uca çözümleme).
- `exceptions.py`: `AutoRagMLError` / `ConfigError` / `PresetError`.

### Değişti
- Çekirdek bağımlılık: `pydantic-settings>=2.2`. Dev: `types-PyYAML`.
- Repo-kök `configs/presets/` kaldırıldı (pakete taşındı); `configs/` = kullanıcı proje config'leri.
- `02_layers.md` config bölümü koda göre güncel.
- **İlk implementasyon kodu**: `contracts/` pydantic v2 modelleri (enums, _base, RunConfig, Dataset, DataProfile/ColumnProfile/TimeSeriesProfile, TaskSpec, PlanContext, AdaptivePlan, Candidate, TuningResult, ValidationReport, ScoreBoard/SelectionResult, ModelBundle, RunManifest, EngineResult, RunResult).
- `tests/contract/test_contracts_smoke.py` — 12 test (doğrulama, frozen, alias round-trip, kompozisyon).
- `pydantic.mypy` plugin; `.venv` + `pip install -e .[dev]` çalışır.

### Değişti
- **`requires-python` `>=3.10` → `>=3.11`** (enum.StrEnum, tomllib). Ruff/mypy `py311`; CI matrisi 3.11/3.12/3.13.
- Uzun stub docstring satırları reflow (E501).
- ADR 0015: `RunManifest` (genişletildi: env/timeline/warnings/realized_seconds/K) + `EngineResult` + `RunResult` + `PlanContext`; uçtan uca orkestrasyon akışı.
- Açık sorular kapandı: `PlanContext` (test/full erişim yok), `candidate_ops` → HPO arama uzayı (gruplu), recipe registry katmanlı (`dynamics/recipes/` + `recipe_paths` + entry-points).
- Stub: `contracts/{run_manifest,engine_result,plan_context,run_result}.py`.

### Değişti
- **Tüm contract'lar donduruldu** (ADR 0008-0015). `01_contracts.md` + `00_overview.md` akışı hizalandı.
- Sıradaki: `contracts/*.py` pydantic v2 + `tests/contract/` — **ilk implementasyon kodu**.
- ADR 0012: Model kataloğu **YAML** (`configs/model_catalog/*.yaml`) + registry; `class_path`, `requires`, `search_space`, `fidelity`; kullanıcı override YAML ile.
- ADR 0013: HPO ensemble-öncelikli + multi-fidelity (SH/Hyperband) + nested; `hpo_level: none|light|thorough`; fold-içi iç-val early stopping.
- ADR 0014: ScoreBoard + dürüst seçim (Winning by Peeking). Seçim yalnız validation, 1-SE kuralı default, realized wall-clock + K, σ√(2lnK), MCB/Diebold-Mariano opsiyonel.
- `models/` `fine_tuners/` `scoring/` (metrics·guardrails·selection·comparison_tests) alt iskele; `configs/model_catalog/`.

### Değişti
- `01_contracts.md`: `Candidate`, `TuningResult`, `ValidationReport`, `ScoreBoard/SelectionResult`, `ModelBundle` **donduruldu**.
- `02_layers.md`: models / fine_tuners / scoring ADR 0012-0014'e göre güncel.
- ADR 0010: `analyzers` sözleşmesi + metodoloji. Güncel kaynaklarla doğrulandı (AutoGluon FeatureMetadata, SortingHat, Nixtla tsfeatures, Open Forecast/Kostenko-Hyndman, TransformedTargetRegressor). Metodoloji: betimle→karar ver→fold'da fit; "her şeyi dönüştür sonra skew ile ele" **reddedildi**.
- ADR 0011: leakage-safe by construction (Grammar of ML Workflows + LeakageDetector). fit/transform/apply ayrımı, immutable `FittedTransform`, `Frame.provenance`, 3-kategori taksonomi (overlap/preprocessing/multi_test), nested CV zorunlu.
- `analyzers/` alt iskele: `modality/profiling/task_inference/timeseries/quality/leakage`.
- `preprocessors/base.py`, `validators/` stub güncellemeleri.

### Değişti
- `01_contracts.md`: `ColumnProfile` (raw_dtype+special_types+semantic_role+flags), `TimeSeriesProfile`, `TaskSpec` (7 task), `AdaptivePlan` (committed vs candidate ops), `FittedTransform` protokolü + `Frame.provenance` **donduruldu**.
- `02_layers.md`: analyzers / preprocessors / validators ADR 0010+0011'e göre güncel.
- ADR 0004: intermittency routing → ipucu (havuz genişletir), router değil.
- **Motto** (`00_overview.md`): zaman değil sağlıklı başarı ölçü; detay kaçmaz. Tüm adımlara uygulanır.
- ADR 0009: `Dataset` + `io` sözleşmesi — strict fingerprint (örneklem yok), long kanonik format (wide → auto-melt, sınırlı model havuzu), v1 tek analitik tablo (`relations` rezerve), otomatik eager/lazy, DB opsiyonel.
- `Dataset` alan tablosu `01_contracts.md`'de **donduruldu**; stub `contracts/dataset.py`.
- ADR 0007: `dynamics` = deterministik `planner` + custom `recipes/` plug-point + v2 `synthesis`.
- `dynamics/` alt iskele: `planner.py`, `recipes/`, `synthesis.py` (docstring stub).

### Eklendi
- ADR 0008: `RunConfig` varsayılanları + çıkarım politikası (cömert bütçe / sessiz kesme yok; katmanlı split + v2 autopilot; v1'de açık target/time/group; sırlar yalnız `.env`).
- `RunConfig` alan tablosu `01_contracts.md`'de **donduruldu**.
- Stub: `config/settings.py` (pydantic-settings, `.env`), `contracts/run_config.py`.

### Değişti
- `01_contracts.md`: `RunConfig` bölümü bullet listeden tam alan tablosuna geçti.
- `docs/architecture/01_contracts.md`: `AdaptivePlan` artık recipe referansı taşıyor; 3 açık soru ADR 0007 ile kapandı.
- `docs/architecture/02_layers.md`: `dynamics/` bölümü ADR 0007'ye göre güncellendi.

## [0.0.1] - 2026-09-01

### Eklendi
- Repo iskeleti: `src/autoragml` katman ağacı (docstring stub'ları), `pyproject.toml`
  (hatchling, extras: xgboost/catboost/timeseries/hpo/tracking/report/explain/llm-*/autogluon/dev).
- Mimari dokümanları: `docs/architecture/` genel bakış + katman taslakları.
- Karar kayıtları (ADR) 0001–0006.
- CI iskeleti: 3 OS × Python 3.10–3.12 (ruff + mypy + pytest).
- `configs/presets/` hazır reçete taslakları.

### Not
- İmplementasyon kodu yok. Sözleşmeler (`contracts/`) kesinleşmeden katman kodu yazılmayacak.
