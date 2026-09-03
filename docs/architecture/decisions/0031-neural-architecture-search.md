# ADR 0031 — nöral mimari arama (`architecture_search` tuner modu)

**Durum:** Kabul · 2026-09-03 (kullanıcı 3 kararı kilitledi)

**Kilitli kararlar:** (1) kapsam = **aile taraması → kazanan ailede derin mimari arama** (iki
aşama); (2) etkinleşme = **yalnız açık `neural_search=True`** (hpo_level/preset bağlaması yok);
(3) backend = **`pytorch-tabular`** (`[neural-nas]` extra), serving = `TabularModel` dizini bundle'a
gömülür (persistence sözleşmesi genişler).

Kaynak: kullanıcı isteği — "parametre optimizasyonu gibi ama daha kapsamlı; parametreleri test
eden, **farklı layerlar ekleyen**, bu değişikliklere göre sonuç üreten bir optimizer" (layer
optimizer). Kullanıcı işlem süresi maliyetini açıkça kabul etti.

## Araştırma özeti (2026-09)

| araç | ne yapıyor | not |
|---|---|---|
| **Auto-PyTorch Tabular** (AutoML-Freiburg) | mimari (ağ tipi, katman sayısı, genişlik, aktivasyon, dropout, skip) + eğitim HP + ön-işlemeyi **ortak** optimize eder; **BOHB** (Bayesian opt + Hyperband) + meta-portföyler; "small"/"full" iki uzay | kanonik referans |
| **pytorch-tabular** `model_sweep` + `Tuner` | önce tüm mimari aileleri default HP ile tara → en iyi mimariyi seç; sonra HP ara; tüm knoblar config'te açık | **aktif bakımlı**, torch |
| NNI (Microsoft) | multi-trial + one-shot NAS (DARTS/ENAS) | ~2023'ten beri arşiv |
| AutoKeras | Keras/TF, hyper-graph Bayesian opt | TF GPU dertleri — kullanmıyoruz |

**2026 uzlaşısı (önemli):** Saf NAS çoğu ekip için pahalı; kazananlar DL'i ensemble yığınında bir
model sınıfı gibi ele alıyor. RealMLP makalesi: **meta-öğrenilmiş default'lar HP-optimize edilmiş
metotlarla yarışıyor**. Mimari araması **azalan getiri** — ama büyük veri / sıra dışı özellik
yapısı / bol compute'ta net fayda sağlıyor.

## İlke

**Sıfırdan NAS yazmıyoruz.** Mevcut `fine_tuners` çok-fideliteli makinesini (SH/Hyperband —
`halving.py`, eta=3) **yapılandırılabilir bir nöral model** üzerinde **koşullu mimari+HP arama
uzayı**yla sürüyoruz. Backend: `pytorch-tabular` (`[neural-nas]` extra) — torch, tüm knoblar açık.
**RealMLP-default + GBDT havuzda kalır** → arama daha iyisini bulamazsa ensemble güçlü default'a
döner (ADR 0021 GES + ADR 0014 1-SE). **Regresyon garantisi yok değil — var.**

## Karar

### 1. Etkinleştirme (KİLİTLİ: yalnız açık bayrak)

`RunConfig.neural_search: bool = False` + `neural_search_space: Literal["small", "full"] = "small"`
+ `neural_search_budget_seconds: int | None = None`.
- **Yalnız `neural_search=True`** ile çalışır. `hpo_level`/preset **tetiklemez** (sürpriz uzun
  koşum yok). `neural_enabled` kapısı (ADR 0030) önce geçilmeli (GPU / satır bandı).
- Yalnız `family: neural` adaylarını etkiler; GBDT/linear normal HPO'da kalır.

### 2. İki aşama (KİLİTLİ: sweep → derin arama)

**Aşama A — aile taraması** (`pytorch_tabular` model aileleri, meta-tune default'lar, kısa bütçe):
`MLP` (Category Embedding) · `ResNet` · `GANDALF` · `FT-Transformer` · `TabNet` · `NODE`
(katalogda `neural_nas.yaml`, kullanıcı kısaltabilir). Her aile 1× düşük-epoch fit → iç-val
metriği → **en iyi 1-2 aile** seçilir.

**Aşama B — derin mimari + HP arama** (kazanan aile[ler] üzerinde, multi-fidelity):
koşullu arama uzayı `configs/search_spaces/neural_arch_{small,full}.yaml` (override edilebilir):

**small** (~30-60 konfig):
```
n_layers        int         [1, 4]
layer_width     categorical [64, 128, 256, 512]
dropout         float       [0.0, 0.4]
learning_rate   loguniform  [1e-4, 1e-2]
```
**full** (~100-200 konfig):
```
+ activation     categorical [relu, gelu, mish, leaky_relu]
+ residual       bool                      # skip-connection (MLP → ResNet)
+ normalization  categorical [batch, layer, none]
+ weight_decay   loguniform  [1e-6, 1e-2]
+ batch_size     categorical [256, 512, 1024]
+ embedding_dim  int         [8, 64]       # kategorik gömme
+ layer_width_scaling categorical [const, pyramid, funnel]
+ (FT-T kazandıysa) n_heads / attn_dropout / ff_multiplier
```
Aile-özel HP'ler `condition` ile aktive olur (ör. `n_heads` yalnız `family == ft_transformer`).

### 3. `SearchDim` sözleşmesi genişler (additive)

Koşullu HP için: `SearchDim.condition: dict[str, object] | None = None`
(ör. `{"param": "residual", "eq": true}` — residual seçilmezse `residual_block_depth` örneklenmez).
`fine_tuners/space.py` örnekleme sırasında koşulu değerlendirir; koşul sağlanmazsa param atlanır.

### 4. Multi-fidelity

Fidelity ekseni = **epoch** (`max_epochs`). `halving.build_schedule` (eta=3): düşük bütçe
(ör. 15 epoch) → çok konfigürasyon; her turda en iyi 1/eta yukarı bütçeye (45 → 135 epoch).
`pytorch-tabular` `trainer_config.max_epochs` fidelity parametresi; `checkpoints` ile ara-durum
korunur (ASHA benzeri erken durdurma pytorch-lightning callback'iyle).

### 5. `ArchitectureSearchTuner` (`fine_tuners/arch_search.py`)

`Tuner` protokolüne uyar (candidate + dış-fold train frame + plan + config → `TunerOutcome`).
- **Nested CV korunur (ADR 0010/6):** aile taraması + derin arama yalnız dış-fold train'in iç
  resample'ında. Dış fold yalnız skorlar.
- Aşama A: her aile 1× (düşük epoch) → iç-val → en iyi ≤2 aile.
- Aşama B: kazanan aile(ler) üzerinde SH/Hyperband — config → `pytorch_tabular`
  `MLPConfig`/`GANDALFConfig`/`FTTransformerConfig`/... → `TabularModel.fit` → iç-val metriği.
  SH turları arası `neural_search_budget_seconds` (yoksa varsayılan 3600s) zorlanır (kill).
- Çıktı: en iyi mimari+HP config `TunerOutcome.best_params` içinde; `candidate.key`
  `neural_arch_search` olarak scoreboard'a girer (`family: neural`, `_FAMILY_COMPLEXITY` = 5).
- **Serving:** `pytorch_tabular` `TabularModel` — `Predictor` protokolü sarımı
  (`FittedNeuralArchPipeline`). joblib picklable DEĞİL → `persistence` genişler (aşağıda).

### 5b. Persistence genişlemesi (KİLİTLİ)

`save_bundle` (ADR 0018): şampiyon `neural_arch_search` ise `champion.joblib` yanında
`champion_neural/` dizini — `TabularModel.save_model(dir)` (torch state + config + datamodule).
`load_bundle`: dizin varsa `TabularModel.load_model` + `FittedNeuralArchPipeline` sarımı.
`ModelBundle.metadata.params` mimari config'i (n_layers, family, ...) taşır — manifest'te görünür.
Sözleşme: `BundleMetadata` değişmez; `persistence.bundle` `_NEURAL_DIR = "champion_neural"` sabiti +
save/load dallanması (additive).

### 6. Determinizm + kaynak

- `configure_torch` (ADR 0030) + `pytorch_tabular` `seed`. `best_effort`.
- `budget.total_max_seconds` yoksa `neural_search` için varsayılan tavan (ör. 3600s) enjekte edilir
  (aksi halde kullanıcıyı şaşırtacak kadar uzun sürebilir) — WARNING ile.
- VRAM koruması: `batch_size` OOM → otomatik yarıya indir + retry (pytorch-lightning
  `auto_scale_batch_size` veya kendi try/except).

## Sözleşme (donacak)

- `RunConfig`: `neural_search: bool = False`, `neural_search_space: Literal["small","full"] = "small"`,
  `neural_search_budget_seconds: int | None = None` (additive).
- `SearchDim.condition: dict[str,object] | None = None` (additive — mevcut düz uzaylar etkilenmez;
  `{"param": ..., "eq"|"ne"|"ge"|"in": ...}`).
- `configs/search_spaces/neural_arch_small.yaml` + `neural_arch_full.yaml` + `neural_families.yaml`
  (aile listesi) — pakete gömülü, override edilebilir.
- `fine_tuners/arch_search.py::ArchitectureSearchTuner`; `resolve_tuner`: `neural_search` iken
  `family: neural` adaylara bunu, diğerlerine normal tuner'ı (heterojen tuner seçimi —
  `validators.run_validation_suite` aday-başı tuner destekler).
- `pyproject` `neural-nas = ["pytorch-tabular>=1.1"]` (torch/pytabkit üzerine).
- `Candidate` key `neural_arch_search` (sentetik). `_FAMILY_COMPLEXITY["neural"]` = 5 korunur.
- `persistence.bundle`: `_NEURAL_DIR` + `TabularModel.save_model`/`load_model` dallanması;
  `FittedNeuralArchPipeline` (`Predictor`, joblib-picklable **değil** — dizinden yüklenir).
- `Candidate`/`EngineResult`/`ModelBundle`/`BundleMetadata` sözleşmeleri **değişmez**.

## Kapsam dışı / sonra

- **One-shot NAS** (DARTS/ENAS/ProxylessNAS — süper-ağ, gradyan tabanlı) → v2 (torch-heavy,
  determinizm zor).
- **Cross-table meta-learning portföyleri** (Auto-PyTorch'un asıl gücü — 100+ dataset'te öğrenilmiş
  başlangıçlar) → v1.2.
- **Transformer mimari araması** (FT-T katman/head sayısı) → önce FT-Transformer mini-ADR.
- **RL-NAS / evrimsel NAS** → kapsam dışı (compute:fayda oranı kötü).
- **Nöral forecasting mimari araması** → ADR 0032 (neuralforecast) sonrası ayrı değerlendirme.
- **Auto-PyTorch backend / BOHB / cross-table portföyler** → v1.2 (ağır, kendi CV'si bizimkiyle çakışır).

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `models/neural_arch.py::TabularModelEstimator` — sklearn-uyumlu `pytorch_tabular.TabularModel`
  sarımı (`fit(X,y)`/`predict`/`predict_proba`/`save`/`load`). Aileler: `mlp`
  (`CategoryEmbeddingModelConfig`), `gandalf` (`GANDALFConfig`), `ft_transformer`
  (`FTTransformerConfig`). `_layer_widths` (const/pyramid/funnel). Kendi train/val split'i + ES.
- `models/estimator.build_estimator`: `class_path == "__neural_arch__"` → `TabularModelEstimator`
  (`task_kind` enjekte).
- `models/catalog/neural.yaml` `neural_arch_search` (`fidelity: max_epochs`, `requires: [pytorch_tabular]`).
- `contracts/candidate.SearchDim.condition` (additive) + `fine_tuners/space.sample_params` iki-tur
  koşullu örnekleme (`eq`/`ne`/`ge`/`in`).
- `fine_tuners/_spaces/neural_arch_{small,full}.yaml`.
- `fine_tuners/arch_search.py::ArchitectureSearchTuner` — Aşama A (3 aile × `_SWEEP_EPOCHS=20`) →
  en iyi ≤2; Aşama B (`_n_configs` 12/24 × SH `build_schedule`, budget kill). `_ARCH_KEY` dışı →
  `fallback` tuner. Nested CV korunur (`evaluate_trial` yeniden kullanımı → `build_estimator` routing).
- `fine_tuners.resolve_tuner`: `neural_search` → `make_arch_tuner(config, base)`.
- `models/neural_gate.prepare_neural_candidates`: `neural_arch_search` yalnız `neural_search=True` +
  GPU/satır kapısı.
- `engines/champion._fit_pipeline(want_bag=candidate.family != "neural")` — nöral tek-model refit.
- `persistence/bundle`: `_NEURAL_DIR` sidecar — `save_bundle` `TabularModelEstimator`'ı dizine yazar,
  pickle'da `None`; `load_bundle` `.load()` ile geri koyar. `neural_sidecar` bayrağı payload'da.
- `pyproject` `neural-nas` extra + mypy override (`omegaconf.*`).
- `tests/unit/fine_tuners/test_arch_search.py` (koşullu uzay, tuner çözümleme, delege).

### GPU e2e doğrulaması (RTX 4060, 2026-09-03) ✅

- `test_neural_arch.py` (6): 3 aile (mlp/gandalf/ft_transformer) fit+predict GPU'da · `build_estimator`
  routing · `_layer_widths` (const/pyramid/funnel) · **bundle sidecar round-trip** (save→load→predict
  eşleşiyor, `champion_neural/` dizini).
- Uçtan uca (`neural_search=True`, küçük sentetik): `neural_arch_search` **çalıştı + leaderboard'da**
  (RMSE 0.742). Şampiyon `tab_m` (0.572) — arama küçük veri + dar bütçede ADR 0030 default'larını
  geçemedi; **GES/1-SE daha iyi modeli tuttu (regresyon garantisi çalıştı)**.
- **Cache kritik:** ilk implementasyonda arama her dış CV fold'unda tekrar koşuyordu (175+ nöral fit
  runaway). `ArchitectureSearchTuner._cache` → arama bir kez (ilk fold), sonraki fold'lar aynı
  mimariyi değerlendirir (Auto-PyTorch deseni, sızıntı yok).
- **Maliyet notu:** lightning per-fit kurulum maliyeti küçük veride baskın (~10dk). Gerçek fayda
  büyük veri + gerçek bütçe (`neural_search_budget_seconds` vars. 3600s) ile.

### Eski plan (referans)
6. `pytorch_tabular` wrapper (`models/neural_arch.py`?) — config→`TabularModel`, `FittedNeuralArchPipeline`.
7. `persistence.bundle` `_NEURAL_DIR` save/load dallanması.
8. `RunConfig` alanları + `resolve_run_config`.
9. Testler: `condition` örnekleme, `ArchitectureSearchTuner` (mock `TabularModel`), bundle
   save/load round-trip, `neural_search=False` → hiç etki yok.
10. GPU e2e (RTX 4060): `neural_search=True` küçük veri → `neural_arch_search` leaderboard'da +
    serving round-trip; benchmark bir tabular set.
