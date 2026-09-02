# ADR 0030 — nöral tablo katmanı (RealMLP · TabM · RealTabR)

**Durum:** Kabul · 2026-09-03 (kullanıcı kilitledi)

**v1.1 ADR sırası (kullanıcı onayı):** 0030 nöral tablo · **0031 nöral mimari arama (layer
optimizer)** · 0032 nöral forecasting · 0033 foundation modeller · 0034 L2 stacking · 0035 seçim
temizliği · 0036 sınıflandırma GES · 0037 açıklanabilirlik. Her ADR kilitlendikçe sonraki.

> **0031 ileri-uyum notu:** Nöral mimari arama (Auto-PyTorch tarzı — katman sayısı/genişlik/
> aktivasyon/residual + HP ortak arama, multi-fidelity) ADR 0031'de `fine_tuners` genişlemesi
> olarak gelecek. Backend: `pytorch-tabular` (yapılandırılabilir modeller), ayrı `[neural-nas]`
> extra. Bu ADR (0030) pytabkit sklearn wrapper'larıyla **sabit-mimari güçlü defaultları** kurar;
> 0031 bunun üzerine arama ekler. RealMLP-default + GBDT havuzda kalır → arama bir şey bulamazsa
> ensemble güçlü defaulta döner (regresyon yok).

Kaynak: v1'de tek nöral model `sklearn.MLPRegressor` (`family: neural`, meta-tune yok) — güncel
tablo derin öğrenme manzarasının çok gerisinde. TALENT / TabArena (NeurIPS 2025) bulguları:
**iyi ayarlanmış DNN'ler (özellikle RealMLP, ModernNCA) GBDT ansambllarını yakalıyor/geçiyor**;
ensembling nöralde de kazandırıyor (TabM = MLP ansamblı). `pytabkit` (1.7.x) bu modelleri
**sklearn arayüzüyle** veriyor (GPU auto, kategorik tespit, kendi ön-işleme + train/val erken
durdurma). AutoGluon 2026 preset'leri `REALMLP` / `TABM`'i portföye aldı.

## İlke

Nöral tablo modelleri **yeni engine değil** — mevcut `TabularCoreEngine` + katalog girişleri
(ADR 0012/0015). `pytabkit` sklearn wrapper'ları `build_estimator`/`fit_estimator` sözleşmesine
zaten uyuyor. Çekirdek **torch'suz kalır** (ADR 0003); nöral yalnız `[neural]` extra ile gelir.

## Karar

### 1. Modeller (`configs/model_catalog/neural.yaml`, `[neural]` extra)

| key | sınıf (`pytabkit.models.sklearn_interfaces`) | not |
|---|---|---|
| `real_mlp` | `RealMLP_TD_Classifier` / `RealMLP_TD_Regressor` | meta-tune edilmiş MLP + "bag of tricks"; tek en güçlü DNN |
| `real_mlp_s` | `RealMLP_TD_S_*` | hızlı/küçük varyant (`family_policy=minimal`, düşük bütçe) |
| `tab_m` | `TabM_Classifier` / `TabM_Regressor` | paylaşımlı-parametreli MLP ansamblı (ICLR 2025) |
| `real_tab_r` | `RealTabR_D_Classifier` / `RealTabR_D_Regressor` | derin + en-yakın-komşu (TabR ailesi) |

- Hepsi `family: neural`, `requires: [pytabkit]`, `predict_kind: [point, proba]` (sınıflandırma),
  `wrap: false` / `scale: false` (model kendi ön-işlemesini yapar), `modalities: [tabular, timeseries]`
  (reduction regresörü olarak forecasting'e de girerler — pytabkit sklearn regresörü).
- **FT-Transformer / transformer ailesi** bu ADR'de YOK (KARAR: ertelendi): `pytabkit` sklearn
  wrapper'ı yok; TALENT'te ensemble'lara karşı istatistiksel üstünlüğü yok; >100 özellikte
  ölçeklenmiyor. → ADR 0031 sonrası ayrı mini-ADR (`rtdl_revisiting_models` sarımı), gerekirse.
- `mlp` (sklearn) katalogda **kalır** — `pytabkit` yoksa nöral aile tümden boş kalmasın; `real_mlp_s`
  kuruluysa `mlp` `enabled: false`'a düşürülür (registry, `real_mlp` deneyimi > sklearn MLP).

### 2. Determinizm (ADR 0003 — v1 tam deterministik idi) — KARAR: varsayılan `best_effort`

`RunConfig.neural_determinism: Literal["strict", "best_effort", "off"] = "best_effort"`.

- `models/torch_env.py` `configure_torch(seed, mode)`: `torch.manual_seed` + `numpy`/`random` seed,
  `torch.backends.cudnn.deterministic=True` / `benchmark=False`, tek-thread dataloader.
- `strict` → ek `torch.use_deterministic_algorithms(True)` (bazı op'lar CUDA'da hata verir → nöral
  aday **atlanır**, WARNING; CPU'da sorun yok).
- `best_effort` (varsayılan) → `use_deterministic_algorithms(True, warn_only=True)`; GPU atomik
  toplamlar nedeniyle bit-düzeyi tekrar garanti edilmez, **manifest'e `neural_determinism: best_effort`
  ve torch/cuda sürümleri yazılır** (ADR 0009 fingerprint mantığı: koşum tekrarlanabilirliği
  raporlanır, sessiz değil).
- `off` → yalnız seed; en hızlı.

Çekirdek deterministik iddiası **modalite-koşullu** hale gelir: "tablo/TS deterministik; nöral
`[neural]` extra ile `best_effort`" — `00_overview.md` güncellenir.

### 3. Cihaz + kaynak + varsayılan (KARAR — kullanıcı 2026-09-03)

- `RunConfig.neural_device: Literal["auto", "cpu", "cuda"] = "auto"` — `pytabkit` zaten auto seçer;
  bu alan override + manifest kaydı için.
- `RunConfig.neural_enabled: Literal["auto", true, false] = "auto"`:
  - **`"auto"` (varsayılan): GPU (CUDA) tespit edilirse nöral adaylar havuza girer; CPU'da GİRMEZ.**
    Varsayılan fit hızlı kalır, GPU'lu kullanıcı otomatik faydalanır (AutoGluon değil — daha muhafazakâr).
  - `true` → cihazdan bağımsız dahil (CPU'da bilinçli yavaşlık).
  - `false` → hiç dahil etme.
- `neural_min_rows: int = 500` · `neural_max_rows: int | None = None` (üstünde nöral adaylar
  atlanır; GPU varsa `dynamics` bu eşiği 4×'ler).
- `dynamics.planner`: nöral aday, `_neural_applicable(profile, config, has_gpu)` → `neural_enabled`
  kapısı + `neural_min_rows ≤ n_rows ≤ (neural_max_rows or ∞)` iken havuza girer. RealMLP'nin
  meta-tune defaultları güçlü → `hpo_level=none`'da bile yarışırlar (sklearn MLP'nin aksine).
- GPU tespiti: `models/torch_env.py::has_cuda()` (torch import'u lazy; `[neural]` yoksa `False`).

### 4. Eğitim akışı (mevcut sözleşme korunur)

- `build_estimator`: `family == "neural"` + `requires` karşılanıyorsa `configure_torch` çağrılır,
  sonra sklearn wrapper instance. `wrap`/`scale` sarımı YOK.
- `fit_estimator`: `pytabkit` modelleri **kendi iç train/val split'iyle erken durur** →
  `supports_early_stopping: false` (fold-içi ES devre dışı; `n_epochs`/`val_metric_name`
  default'ları yeterli). `fidelity` yok (v1.1'de epoch-fidelity SH → v1.2).
- **k-fold bagging (ADR 0022) + GES (ADR 0021): normal katılır.** TabM zaten ansambl — GES onu
  tek üye olarak alabilir (çakışma yok).
- **Leakage:** `pytabkit` ön-işlemesi (kategorik encode, quantile) `fit`'te yalnız o fold'un
  train'inde çalışır (sklearn `fit(X_train)` sözleşmesi) → ADR 0011 korunur. `FeaturePipeline`
  nöral adaylar için minimal (yalnız `committed_ops` — impute/drop; encode/scale'i model yapar) —
  `family_policy["neural"] = "minimal"` (v1'de `"full"` idi; değişiyor).

## Sözleşme (donacak)

- `configs/model_catalog/neural.yaml` — `real_mlp` / `real_mlp_s` / `tab_m` / `real_tab_r`.
- `RunConfig`: `neural_enabled: bool`, `neural_device`, `neural_determinism`, `neural_min_rows`,
  `neural_max_rows` (hepsi additive, güvenli defaultlar).
- `pyproject.toml` `[project.optional-dependencies]` `neural = ["pytabkit>=1.7", "torch>=2.2"]`.
- `models/torch_env.py` `configure_torch(seed, mode) -> str` (seçilen device döner).
- `Candidate` sözleşmesi **değişmez** (`family` serbest string; `requires` zaten var).
- `family_policy["neural"]`: `"full"` → `"minimal"`.
- `RunManifest.env`: `torch_version` / `cuda_version` / `neural_determinism` / `neural_device`.

## Kapsam dışı / sonra

- FT-Transformer / ExcelFormer / AutoInt (transformer ailesi) → ayrı mini-ADR, `rtdl` sarımı.
- Epoch-fidelity multi-fidelity (SH/Hyperband nöralde) → v1.2.
- Foundation modeller (TabPFNv2) → ADR 0032 (ayrı: in-context learning, fit yok).
- Nöral forecasting (NHITS/PatchTST/TFT) → ADR 0031 (`neuralforecast`, `[neural-ts]`).
- GPU multi-worker / distributed → v2 (`EngineRunner` container tier).
- Nöral için `SubprocessRunner` izolasyonu (torch import ağır) → InProcess v1.1'de yeterli, ölçülür.

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `models/catalog/neural.yaml`: `real_mlp` (RealMLP_TD) · `tab_m` (TabM_D) · `real_tab_r` (RealTabR_D).
  Hepsi `pytabkit.<Class>_{Regressor,Classifier}`, `requires: [pytabkit]`, `n_cv: 1` (iç ensemble
  kapalı — bizim bagging/GES devrede). `real_tab_r` yalnız tabular (KNN → panelde pahalı).
- `models/torch_env.py`: `torch_available` / `has_cuda` / `cuda_device_name` / `torch_versions` /
  `resolve_device` / `configure_torch(seed, mode, device)` (idempotent global; torch yoksa `cpu` no-op).
- `models/neural_gate.py::prepare_neural_candidates(candidates, profile, config)`: çalışma-zamanı
  kapısı — `neural_enabled` (auto/on/off) × GPU × satır bandı; pytabkit havuza girince `mlp` düşer;
  `neural_device` override enjekte edilir. `engines/core.run_core_pipeline` çağırır + nöral aday
  varsa `configure_torch`.
- `RunConfig`: `neural_enabled` (`Literal["auto","on","off"]`), `neural_device`, `neural_determinism`,
  `neural_min_rows` (500), `neural_max_rows`. `tabular_fast` preset → `neural_enabled: "off"`.
- `EnvInfo.accelerator: dict[str,str]` (additive); `manifest._accelerator_info` doldurur;
  `_ENV_PACKAGES` += `torch`, `pytabkit`.
- `_FAMILY_POLICY["neural"]`: `"full"` → `"minimal"` (v1: bilgi amaçlı — henüz tüketilmiyor).
- `build_estimator`/`fit_estimator` **değişmedi** — pytabkit sklearn wrapper `wrap:false`/`scale:false`,
  `supports_early_stopping:false` (kendi iç val'i). `Candidate`/engine sözleşmeleri değişmez.
- mypy override += `torch.*` / `pytabkit.*` / `pytorch_tabular.*`.
- `tests/unit/models/test_neural_tier.py` (12): torch-yok güvenliği, katalog-atla, kapı matrisi,
  cihaz enjeksiyonu, config + preset.

### Kurulum + doğrulama (RTX 4060, 2026-09-03) ✅

```
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 2.6.0+cu124
pip install pytabkit skorch                                            # 1.7.3 + lightning 2.6
```

- `torch.cuda.is_available() → True`, `NVIDIA GeForce RTX 4060`, `cuda 12.4`. **PyTorch RTX 4060'ı
  ilk denemede gördü — debug gerekmedi** (Keras/TF GPU dertleri PyTorch'a bulaşmıyor).
- `pytabkit.RealMLP_TD_Regressor` doğrudan fit/predict GPU'da OK.
- `AutoRagML().fit(..., neural_enabled="on")` e2e (**doğrusal-olmayan sentetik, 700 satır**):
  ```
  şampiyon: real_mlp  (RMSE 0.392)
  tab_m 0.483 · weighted_ensemble 0.549 · lightgbm 0.785 · hist_gbm 0.849 · ... · ridge 2.686
  ```
  **RealMLP GBDT'yi net geçti** (0.39 vs 0.79) — araştırmanın vaadi doğrulandı. Her ikisi GPU'da
  eğitildi, GES (`weighted_ensemble`) nöralleri aldı, serving çalıştı, `accelerator` manifest doldu.
- `real_tab_r`: `skorch` + `faiss` zinciri gerektiriyor → `requires: [pytabkit, skorch, faiss]`
  (opt-in-opt-in, `[neural]` extra'ya girmez); faiss yoksa registry temiz atlar.
- `configure_torch`: RTX 40xx için `torch.set_float32_matmul_precision("high")` (Tensor Core, determinizmi bozmaz).
- Testler torch present/absent iki senaryoya da dayanıklı (skipif markerleri) — CI (`[neural]` yok)
  "absent" dalını, lokal "present" dalını koşar.

### Kalan (opsiyonel, bu ADR dışı)

- Benchmark: 1. dalga `--hpo none` GPU'da — RealMLP GBDT'yi yakalıyor mu? (ADR 0031 sonrası birlikte)

### v1 sınırı (açık)

`FeaturePipeline` nöral adaylara da tam (one-hot + quantile) uygular → pytabkit kendi
ön-işlemesini bunun üstüne yapar (kategorik bilgisi kaybolur, çift-transform). Per-candidate
minimal pipeline → v1.1 (`family_policy["neural"]` tüketimi).
