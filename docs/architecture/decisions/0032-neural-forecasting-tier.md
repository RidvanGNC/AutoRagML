# ADR 0032 — nöral forecasting katmanı (NHITS · PatchTST · TFT · NBEATSx · iTransformer · TSMixer)

**Durum:** Kabul · 2026-09-03 (kullanıcı 3 kararı kilitledi)

**Kilitli kararlar:**
1. **6 model:** NHITS · PatchTST · TFT · NBEATSx · iTransformer · TSMixer.
2. **Mimari arama:** `neural_search=True` → **kütüphane `Auto*` modelleri** (`AutoNHITS`/…). ADR 0031
   tuner'ı uyarlanmıyor — `neuralforecast` mimari aramayı model-içinde (bizden iyi ayarlı uzaylarla)
   yapıyor; aile-arası seçim zaten bizim dış CV + GES/1-SE işi (6 `Auto*` havuzda yarışır). Tek
   "eksik" (ADR 0031 Aşama A hızlı taraması) nöral-TS'de gereksiz — tam CV karşılaştırması daha dürüst.
3. **CPU:** `neural_enabled=auto` iken CPU'da nöral-TS **tamamen dışlanır** (CV çok pahalı); yalnız
   GPU veya açık `neural_enabled=on`.

Kaynak: v1.1 sırası (ADR 0030 sonrası). Forecasting'de nöral yok — reduction (GBDT/RealMLP) +
klasik (StatsForecast). M5 2. sıra NBEATS/seq2seq kullandı; PatchTST/NHITS güncel SOTA panel
modelleri. `neuralforecast` (Nixtla) `statsforecast`'la **aynı ekosistem/API** —
`.cross_validation` / `.fit` / `.predict` (ADR 0023 deseni birebir uygulanır).

## Araştırma özeti (2026-09)

- **`neuralforecast`**: NHITS · NBEATSx · PatchTST · TFT · iTransformer · TSMixer(x) · TiDE · DLinear ·
  DeepAR · TimesNet ... `NeuralForecast` sınıfı, `StatsForecast` ile aynı arayüz.
- Benchmark: **PatchTST genel şampiyon**, TSMixer ≈ iTransformer ardından; NHITS hızlı + güçlü baz.
- **`Auto*` modelleri** (`AutoNHITS`/`AutoPatchTST`/`AutoTFT`): kütüphane-içi HPO (Ray Tune / Optuna
  backend), ön-tanımlı veya kullanıcı arama uzayı. **Nöral-TS mimari araması kütüphanede hazır** —
  ADR 0031'deki gibi kendimiz yazmıyoruz.

## İlke

**Yeni engine yok** — `TimeSeriesCoreEngine` native panel yolu (ADR 0023 klasik gibi).
`family: neural_ts` adayları `NeuralForecast` üzerinden: `.cross_validation` → OOF (rolling-origin,
leakage-safe), `.fit`/`.predict` → serving. Reduction pipeline'ından geçmez (panel API). Çekirdek
torch'suz (ADR 0003) — yalnız `[neural-ts]` extra.

## Karar

### 1. Modeller (`models/catalog/neural_ts.yaml`, `[neural-ts]` extra) — KİLİTLİ: 6 model

| key | sınıf | not |
|---|---|---|
| `nhits` | `neuralforecast.models.NHITS` | hızlı, güçlü baz — hiyerarşik interpolasyon |
| `patchtst` | `neuralforecast.models.PatchTST` | transformer, benchmark şampiyonu |
| `tft` | `neuralforecast.models.TFT` | yorumlanabilir + exogenous (v1.1'de exog yok — ADR 0009) |
| `nbeatsx` | `neuralforecast.models.NBEATSx` | M5 2. sıra ailesi — trend/mevsim blokları |
| `itransformer` | `neuralforecast.models.iTransformer` | ters-token transformer |
| `tsmixer` | `neuralforecast.models.TSMixer` | MLP-mixer, PatchTST'ye yakın |

- `neural_search=True` (ADR 0031 bayrağı yeniden kullanılır) → `Auto{NHITS,PatchTST,TFT,NBEATSx,
  iTransformer,TSMixer}` (kütüphane HPO; `neural_search_budget_seconds` → `Auto*` `num_samples`/
  `time_budget_s`; backend `optuna` varsa TPE, yoksa random).
- `neural_search=False` → sabit meta-tune default'lar.
- `requires: [neuralforecast]`. `Auto*` için `optuna` opsiyonel (`[hpo]` — zaten var).

### 2. Entegrasyon (`engines/timeseries/neural_ts.py` — ADR 0023 paralı)

- `run_neural_ts_reports(frame, profile, task, config, candidates) -> (reports, extra_candidates)`
  — `run_classical_reports` ikizi. `NeuralForecast(models=[...], freq=..).cross_validation(df, h,
  n_windows, step_size)` → per-model `ValidationReport` (pencere-indeksli fold gruplama).
- Adaptif pencere sayısı + kısa-seri filtresi (klasikteki mantık).
- `run_core_pipeline(..., run_neural_ts=config...)` — reduction/klasik/nöral-TS üç aile birleşir;
  şampiyon herhangi birinden. GES: nöral-TS OOF cutoff-tabanlı → klasikle aynı sınıf, `classical_ensemble`
  benzeri `neural_ts` GES v1.1 (v1: ayrı yarışır).
- `FittedNeuralForecaster` (`Predictor` protokolü) — `NeuralForecast.predict` + tarih hizalama
  (ADR 0029 `_horizon_for` mantığı: istenen pencereyi kapsayacak `h`).
- `champion._neural_ts_bundle` — `refit_neural_ts` / `refit_neural_ts_auto`.

### 3. Kapı + kaynak (ADR 0030 ile aynı)

- `neural_enabled` (`auto`=GPU / `on` / `off`) — `neural_gate` genişler (`neuralforecast` de gözetilir).
- `neural_ts_min_series` (vars. 20) · `neural_ts_min_history_mult` (vars. 3× season) — altında atlanır.
- `configure_torch` (ADR 0030) determinizm.
- CPU'da nöral-TS **çok pahalı** → `auto` modda CPU'da havuza girmez (GPU şart).

### 4. Serving / persistence

`NeuralForecast.save(dir)` / `NeuralForecast.load(dir)` → `persistence.bundle` `_NEURAL_TS_DIR`
sidecar (ADR 0031 `_NEURAL_DIR` deseni; `FittedNeuralForecaster` joblib-picklable değil).

## Sözleşme (donacak)

- `models/catalog/neural_ts.yaml` — `nhits` / `patchtst` / `tft`.
- `RunConfig`: `neural_ts_min_series: int = 20`, `neural_ts_min_history_mult: float = 3.0` (additive).
  `neural_enabled`/`neural_device`/`neural_determinism`/`neural_search*` yeniden kullanılır.
- `pyproject` `neural-ts = ["neuralforecast>=1.7"]`.
- `engines/timeseries/neural_ts.py` — `run_neural_ts_reports` + `FittedNeuralForecaster` +
  `refit_neural_ts`.
- `run_core_pipeline(run_neural_ts=)` (additive-optional).
- `persistence.bundle` `_NEURAL_TS_DIR` dallanması.
- `Candidate`/`EngineResult`/`ModelBundle` **değişmez**.

## Bağımlılık notu (kritik)

`neuralforecast` 3.2.1 → `torch>=2.9.1` + `pytorch-lightning<2.6.0`. ADR 0030'un `torch 2.6.0+cu124`'ü
yetmez. **Çözüm:** `torch 2.14.0+cu126` (`--index-url .../cu126`) — 6 modeli + pytabkit + pytorch_tabular
hepsini karşılar (lightning 2.5.6 ile pytabkit/pytorch_tabular hâlâ çalışıyor, doğrulandı).
`[neural]` + `[neural-nas]` + `[neural-ts]` **tek venv'de** çalışır ama kırılgan — üretimde ayrı
venv önerilir (ADR 0004 `autogluon` extra notu gibi).

## Kapsam dışı / sonra

- Exogenous/covariate (TFT'nin asıl gücü) → ADR 0009 `Dataset.relations` açılınca (v1.2).
- `neural_ts` GES ensemble (nöral-TS + klasik ortak backtest) → v1.1 sonrası.
- Probabilistik forecasting (DeepAR, quantile) → v1.2 (`quantiles` alanı var ama nöral-TS'de yok).
- Foundation TS modeller (Chronos/TimesFM) → ADR 0033 (in-context, fit yok).
- Multivariate (MLPMultivariate, StemGNN) → v1.2.

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `models/catalog/neural_ts.yaml` — 6 sabit + 3 `Auto*` (`auto_nhits`/`auto_patchtst`/`auto_tft`).
- `engines/timeseries/neural_ts.py`: `run_neural_ts_reports` (ADR 0023 şablonu — `NeuralForecast.
  cross_validation`, pencere-indeksli fold, `_adaptive_windows`, `_MAX_CV_WINDOWS=2`) +
  `FittedNeuralForecaster` (`Predictor`, tarih-hizalı; fit-sonrası h → daha ileri istek fallback) +
  `refit_neural_ts`. `quiet_cwd` (ADR 0031 → `torch_env`'e taşındı).
- `engines/core.run_core_pipeline(run_neural_ts=)` + `_native_panel` filtresi.
- `engines/champion._neural_ts_bundle`.
- `models/neural_gate.prepare_neural_candidates`: `neural_ts` ailesi de kapıdan geçer
  (`neural_enabled` auto→GPU şart; `neural_ts_min_series` kontrolü); `Auto*` yalnız `neural_search=True`.
- `models/estimator._build_nf_model`: `random_seed`/`accelerator`/`n_series` (multivariate) enjekte;
  `Auto*` → `num_samples` (bütçe/60). Model alias `None` → `type(m).__name__`.
- `RunConfig` `neural_ts_min_series` (20) / `neural_ts_min_history_mult` (3.0).
- `persistence.bundle`: `_NEURAL_TS_DIR` sidecar — `NeuralForecast.save`/`load` (pipeline tümü
  dizinden; `_last` fallback yeniden dolmaz, kabul).
- `pyproject` `neural-ts` extra + mypy override; manifest `_ENV_PACKAGES` += neuralforecast/pytorch-tabular.
- Testler: `test_neural_ts.py` (katalog, kapı, is_neural_ts + GPU e2e CV+refit+serving+bundle round-trip).

**Benchmark → tüm nöral blok (0030-0033) bitince (kullanıcı kararı).**
