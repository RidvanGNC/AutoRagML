# ADR 0031 — nöral mimari arama (`architecture_search` tuner modu)

**Durum:** Taslak · 2026-09-03

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

### 1. Etkinleştirme

`RunConfig.neural_search: bool = False` (+ `neural_search_space: Literal["small", "full"] = "small"`).
- Yalnız `family: neural` adaylarına uygulanır; GBDT/linear normal HPO'da kalır.
- Yalnız `neural_search=True` VEYA `hpo_level=thorough` + GPU. Hiçbir varsayılan preset açmaz.
- `neural_enabled` kapısı (ADR 0030) önce geçilmeli (GPU / satır bandı).

### 2. Arama uzayı (`configs/search_spaces/neural_arch_{small,full}.yaml` — override edilebilir)

**small** (hızlı, ~30-60 konfigürasyon):
```
n_layers        int         [1, 4]
layer_width     categorical [64, 128, 256, 512]
dropout         float       [0.0, 0.4]
learning_rate   loguniform  [1e-4, 1e-2]
```
**full** (SOTA hedefi, ~100-200 konfigürasyon):
```
+ activation     categorical [relu, gelu, mish, leaky_relu]
+ residual       bool                      # skip-connection (MLP → ResNet)
+ normalization  categorical [batch, layer, none]
+ weight_decay   loguniform  [1e-6, 1e-2]
+ batch_size     categorical [256, 512, 1024]
+ embedding_dim  int         [8, 64]       # kategorik gömme
+ layer_width_scaling categorical [const, pyramid, funnel]  # per-layer genişlik profili
```

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
- **Nested CV korunur (ADR 0010/6):** arama yalnız dış-fold train'in iç resample'ında.
- Her deneme: config → `pytorch_tabular` `MLPConfig`/`GANDALFConfig`/... → `TabularModel.fit` →
  iç-val metriği. SH turları arası `budget.total_max_seconds` zorlanır (kill).
- Çıktı: en iyi mimari config `TunerOutcome.best_params` içinde; `candidate.key` `neural_arch_search`
  olarak scoreboard'a girer (`family: neural`, `_FAMILY_COMPLEXITY` = 5 — arama = karmaşık).
- Serving: `FittedModelPipeline` içinde `pytorch_tabular` `TabularModel` (kendi predict'i;
  `Predictor` protokolü sarımı — joblib yerine `TabularModel.save_model` + bundle referansı).

### 6. Determinizm + kaynak

- `configure_torch` (ADR 0030) + `pytorch_tabular` `seed`. `best_effort`.
- `budget.total_max_seconds` yoksa `neural_search` için varsayılan tavan (ör. 3600s) enjekte edilir
  (aksi halde kullanıcıyı şaşırtacak kadar uzun sürebilir) — WARNING ile.
- VRAM koruması: `batch_size` OOM → otomatik yarıya indir + retry (pytorch-lightning
  `auto_scale_batch_size` veya kendi try/except).

## Sözleşme (donacak)

- `RunConfig.neural_search: bool` + `neural_search_space: Literal["small", "full"]` (additive).
- `SearchDim.condition: dict | None` (additive — mevcut düz uzaylar etkilenmez).
- `configs/search_spaces/neural_arch_small.yaml` + `neural_arch_full.yaml` (pakete gömülü).
- `fine_tuners/arch_search.py::ArchitectureSearchTuner`; `resolve_tuner` `neural_search` iken
  nöral adaylara bunu, diğerlerine normal tuner'ı verir (heterojen tuner).
- `pyproject` `[project.optional-dependencies]` `neural-nas = ["pytorch-tabular>=1.1"]`.
- `Candidate` key `neural_arch_search` (sentetik — arama sonucu).
- Serving: bundle'da `pytorch_tabular` model dizini (`persistence` `save_bundle` genişler).

## Kapsam dışı / sonra

- **One-shot NAS** (DARTS/ENAS/ProxylessNAS — süper-ağ, gradyan tabanlı) → v2 (torch-heavy,
  determinizm zor).
- **Cross-table meta-learning portföyleri** (Auto-PyTorch'un asıl gücü — 100+ dataset'te öğrenilmiş
  başlangıçlar) → v1.2.
- **Transformer mimari araması** (FT-T katman/head sayısı) → önce FT-Transformer mini-ADR.
- **RL-NAS / evrimsel NAS** → kapsam dışı (compute:fayda oranı kötü).
- **Nöral forecasting mimari araması** → ADR 0032 (neuralforecast) sonrası ayrı değerlendirme.

## Açık sorular (kilitleme öncesi — kullanıcıya)

1. `neural_search` etkinleşme: yalnız açık bayrak mı, yoksa `hpo_level=thorough` da tetiklesin mi?
2. Backend: `pytorch-tabular` (aktif, kolay entegre) vs `Auto-PyTorch` doğrudan (daha güçlü ama
   ağır bağımlılık + kendi CV'si bizimkiyle çakışır) — `pytorch-tabular` öneriyorum.
3. Serving formatı: `pytorch_tabular` model dizini bundle'a gömülür (joblib değil) — `persistence`
   sözleşmesi genişler. Kabul?
