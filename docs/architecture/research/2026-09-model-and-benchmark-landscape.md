# Model + benchmark manzarası (2026-09 akademik tarama)

v1.1 forecasting benchmark'ından sonra: "hep aynı verilerle çalışıp bir şeyler kaçırıyor
olabilir miyiz?" sorusuna cevap. Akademik kaynaklar taranarak (a) hangi ek model
seçeneklerimiz var, (b) hangi standart benchmark suite'lerini kullanmalıyız.

## 1. Tablo ML — TabArena (NeurIPS 2025 Spotlight) rosteri

**TabArena** = güncel "living benchmark" (51 küratörlü dataset, binary/multiclass/regression;
ROC AUC / log-loss / RMSE; Elo). Değerlendirdiği modeller:

| Kategori | Modeller | Bizde? |
|---|---|---|
| Ağaç | CatBoost, LightGBM, XGBoost | ✅ (catboost/xgboost opt) |
| DL | **RealMLP, TabM**, ModernNCA, xRFM | RealMLP/TabM ✅ · ModernNCA/xRFM ❌ |
| Foundation | TabPFNv2/2.5, **TabICL**, TabDPT, LimiX, Mitra | TabPFN dormant · diğerleri ❌ |

**TabArena ana bulgu:** GBM hâlâ güçlü; DL yeterli bütçe + **ensembling ile yetişti**;
foundation küçük veride domine; **"modeller-arası ensemble SOTA'yı ilerletiyor"** →
bizim GES + L2 stacking + joint yaklaşımımızı doğruluyor.

### Yeni model isimleri (akademik, eklenebilir)
- **ModernNCA** — modern Neighbourhood Component Analysis (DL mesafe-tabanlı); orta veri, güçlü.
- **xRFM** — genişletilmiş random feature model (kernel-benzeri, hızlı).
- **TabICL** — in-context tabular; TabPFN'den **daha iyi ölçekleniyor** (10K+ satır), lisans daha açık.
- **TabDPT** — discriminative pretrained transformer.
- **Mitra** — AutoGluon'un kendi sentetik-veri FM'i (autogluon ile gelir).
- **EBM** (InterpretML `interpret`) — cyclic-boosting GAM + otomatik etkileşim; "XGBoost/LightGBM'e
  yakın doğruluk **ama tam yorumlanabilir**". Cam-kutu; `explain()` ile mükemmel eşleşir (ADR 0037).

### Klasik tablo boşluğu (v1.1 listesi, hâlâ yok)
- **KNN** (distance), **SVR/SVC** (kernel), **GAM** (`pygam`), **NGBoost** (olasılıksal GBM — quantile).

## 2. Forecasting — klasik (statsforecast'te VAR, kullanmıyoruz)

`statsforecast` 2.0.3 ekosistemimizde zaten: **AutoCES** (complex ES — bir benchmark'ta MASE 0.73,
AutoARIMA'nın 0.80'ini geçti), **AutoTBATS** (çoklu mevsim), **IMAPA** (kesikli talep — M5, bizim
croston/tsb yanında), Dynamic/Optimized Theta. **Ekleme maliyeti ~sıfır** (aynı native panel yolu).

## 3. Forecasting — foundation (Chronos'a alternatif)

| Model | Sağlayıcı | Paket | Not |
|---|---|---|---|
| **TimesFM** | Google | `timesfm` | decoder-only, production-test, auth'suz |
| **Moirai-2** | Salesforce | `uni2ts` | MoE transformer, çok-değişkenli one-shot |
| **Lag-Llama** | — | `lag-llama` | olasılıksal çıktı |
| Toto (Datadog) · Sundial | — | — | yeni, olgunlaşmamış |
| Chronos-2 | Amazon | `chronos-forecasting` | katalogda var (enabled:false) — covariate |

Ayrıca: **TabPFN-TS zero-shot multivariate** (arXiv 2604.08400) — tabular PFN'i çok-değişkenli TS'e.

## 4. Benchmark suite'leri — dataset sayımızı buradan artıracağız

### Tablo: **AMLB** (AutoML Benchmark) = standart
- **104 görev**: 71 sınıflandırma (binary+multiclass) + 33 regresyon, hepsi OpenML.
- %10 test / %87.5 train / %12.5 val. Framework/dataset eklemek YAML + wrapper.
- Alternatif: **TabArena 51** (daha küratörlü, daha yeni) · **TabRepo** (dev değerlendirme deposu).
- **Bizde 6.** Öneri: AMLB/TabArena'dan ~15-20 seç — küçük/orta/büyük × binary/multiclass/regresyon
  × temiz/karışık/eksik ekseninde yay.

### Forecasting: **GIFT-Eval** (Salesforce) = TS foundation standardı
- **23 dataset**, 144K seri, 177M nokta, **7 alan** (Energy, Finance, Healthcare, Transport,
  Nature, **Sales**, Web/CloudOps), **10 frekans**, 97 görev, kısa/orta/uzun ufuk.
- **Test = her serinin son %10'u, kesinlikle train penceresinden sonra** → **ADR 0038 per-seri
  holdout yaklaşımımızı doğruluyor.**
- Ayrıca **Monash** arşivi (30+ dataset). M-competitions (M3/M4/M5) = bizim kullandığımız.
- **Bizde 3** (m3/m5/tourism — hepsi aylık/günlük perakende-benzeri). GIFT-Eval saatlik/haftalık,
  enerji, web-trafiği, sağlık ekler → tek tip veriden çıkarız.

## 5. Öneri — sıra

1. **Klasik forecasting eklemeleri** (AutoCES/AutoTBATS/IMAPA) — sıfır maliyet, M4/M5-ilgili.
2. **EBM** (tablo) — cam-kutu, `explain()` ile sinerjik, XGBoost-rakip.
3. **KNN/SVR/GAM/NGBoost** — küçük-veri/olasılık boşluğu.
4. **Benchmark dataset genişletme** — AMLB'den ~12 tablo + GIFT-Eval/Monash'tan ~6 forecasting.
5. **TimesFM** — ikinci TS foundation (Chronos yanında, karşılaştırma için).
6. **TabICL / ModernNCA** — TabArena roster tamamlama (opsiyonel, sonra).

## Kaynaklar
- TabArena — arXiv 2506.16791 (NeurIPS 2025 D&B)
- TabPFN-2.5 — arXiv 2511.08667
- AMLB — openml.github.io/automlbenchmark ; arXiv (AMLB paper)
- GIFT-Eval — arXiv 2410.10393 (Salesforce)
- TabRepo — arXiv 2311.02971
- EBM / InterpretML — interpret.ml/docs/ebm.html
- statsforecast — github.com/Nixtla/statsforecast
