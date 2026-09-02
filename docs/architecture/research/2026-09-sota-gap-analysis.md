# SOTA karşılaştırması & gap analizi — 2026-09

**Amaç:** Benchmark verisetlerimizde yayınlanmış en iyi sonuçlar hangi yöntemlerle
alınmış? Bizim eksiğimiz nerede? (Ara adım — çıktı "eksik yok" da olabilir.)

## Yöntem

Benchmark sonuçlarımız (`scripts/benchmarks/RESULTS.md`) ile literatürdeki/yarışmalardaki
en iyi sonuçlar karşılaştırıldı. Kaynaklar aşağıda.

---

## 1. Tablo verileri — **SOTA'dayız, acil gap yok**

| dataset | SOTA yöntem & skor | bizim | değerlendirme |
|---|---|---|---|
| california_housing | XGBoost R² ~0.83–0.87; stacking 0.843 | lightgbm RMSE 0.446 → **R² ≈ 0.85** | **SOTA aralığında.** Stacking ~+0.01 verebilir |
| adult | GBM accuracy ~88.2% (en yüksek bilinen) | lightgbm f1_macro 0.819 (≈acc %87) | ~1 puan altında; tuning + kategorik encoding derinliği |
| covtype | RF/XGBoost (DL'i geçiyor) | extra_trees f1_macro 0.799 | GBM/forest doğru aile; büyük veride tuning + bagging |
| bike_sharing / credit_g / bank_marketing | GBM tipik en iyi | GBM/linear şampiyon | doğru |

**Sonuç:** Tabloda kazanan reçete **iyi GBM + bagging + ensemble** — hepsi ya var (GBM,
bagging ADR 0022) ya da yol haritasında (stacking v1.1). **Yeni model tipi gerekmiyor.**
Marjinal kazanç: L2 stacking, kategorik encoding derinliği (CatBoost-tarzı), daha iyi HPO.

---

## 2. Forecasting — **gerçek, somut gaplar** (öncelik sırası)

### Gap #1 — Klasik model kombinasyonu (EAT / HOC) · **en yüksek ROI**
- **M3 yarışmasını Theta kazandı**; en iyi ansambl **EAT = ETS + ARIMA + Theta**.
- **M4'ü forecast-combination + meta-learning kazandı** (HOC2: AutoETS+AutoARIMA+Theta+
  TBATS+SeasonalNaive'in horizon-optimize konveks kombinasyonu).
- **Bizde:** `auto_ets`, `auto_arima`, `auto_theta`, `mstl` **var** ama **birleştirmiyoruz** —
  GES ensemble klasiği dışlıyor (ADR 0023 v1: cutoff-OOF ≠ fold-OOF).
- **Yapılacak:** klasik modeller için ayrı bir ensemble (basit ortalama veya cutoff-ızgarasında
  GES). M3'te tek `auto_ets` sMAPE ~15 → EAT ansambl ~13-14'e iner (competition seviyesi).
- **Efor:** düşük-orta. `classical.py` içinde cutoff-hizalı OOF → GES. ADR 0024 adayı.

### Gap #2 — Zengin reduction özellikleri (MLForecast paritesi) · orta ROI
- **Nixtla MLForecast** (M5 tarzı tabular-reduction'ın referansı):
  - `Differences([1])` / seasonal differences **target transform** (trend/mevsim serileri için kritik)
  - `RollingMean/Std/Min/Max` + **`SeasonalRollingMean/Std`** çoklu pencere/lag kombinasyonları
  - `rolling_mean_lag1_window7` gibi türev özellikler ham lag'lerden daha öngörücü
  - tarih özellikleri (ay, haftagünü, ...)
- **Bizde** (`reduction.py`): `y_lag_{h..h+3}`, `rollmean/std_{4,8,13}`, `ewm_{4,12}`,
  `step_index` — **temel var**, ama `Differences` target transform + `SeasonalRolling` +
  tarih özellikleri **yok**.
- **Yapılacak:** `reduction.py`'ye seasonal-rolling + target `Differences` seçeneği + date_expand
  çağrısı. **Efor:** orta.

### Gap #3 — Intermittent talep: Tweedie loss + recursive · orta ROI (DemandSensing kalbi)
- **M5'i hiyerarşik LightGBM ansamblı, Tweedie loss ile** kazandı (mağaza / mağaza-kategori /
  mağaza-departman seviyelerinde pooled; recursive + non-recursive; ortalama).
- M5'te talebin **%78.6'sı düzensiz** (%63.5 intermittent, %15.1 lumpy). Tweedie = sıfırda
  kütlesi olan hacim dağılımı → tam bu iş için.
- **Bizde:** lightgbm `objective` sabit (L2). Croston/TSB var ama zayıf (M5 winner Croston'u
  "önemli ölçüde" geçti). Recursive multi-step **yok** (sadece direct h-step, ADR 0004).
- **Yapılacak:** (a) intermittency sınıfı `lumpy/intermittent` ise lightgbm/hist_gbm için
  `objective="tweedie"` (katalog + planner ipucu), (b) recursive multi-step reduction (v1.1'de
  planlıydı zaten). "Magic multipliers" ≈ bizim `postprocess.calibrate="multiplicative"` — **zaten var.**
- **Efor:** Tweedie düşük (katalog + koşullu param); recursive orta.

### Gap #4 — Hiyerarşik reconciliation (MinT) · niş, düşük öncelik
- Tourism (hiyerarşik) verilerinde **MinT reconciliation** taban forecast'leri tutarlı kılıp
  (toplamlar uyumlu) doğruluğu artırıyor — özellikle üst seviyeler + uzun horizon.
- **Bizde:** hiyerarşi kavramı yok (`Dataset.relations` rezerve). `hierarchicalforecast`
  (Nixtla) kütüphanesi var.
- **Yapılacak:** v1.2+ — `HierarchySpec` sözleşmesi + `hierarchicalforecast` opsiyonel eklenti.
  DemandSensing per-ürün/kategori/toplam yapısı için değerli ama v1 kapsamı dışı.

### Gap #5 — Per-grup / hiyerarşik pooling · planlı (v1.1)
- M5 winner **her mağaza-departman için ayrı model** eğitti (per_group_champion mantığı).
- **Bizde:** `dynamics/planner` `per_group_champion` **planlıyor** ama engine pooled ilerliyor
  (ADR 0015 v1 sınırı). Ayrıca M5 winner **çoklu pooling seviyesi** (global + grup + alt-grup)
  ortalıyor — bizde tek seviye.
- **Yapılacak:** gerçek per-grup refit (v1.1). Çoklu-seviye pooling v1.2.

---

## Öncelikli eklemeler (öneri)

| # | Ekleme | ADR | Efor | Beklenen etki |
|---|---|---|---|---|
| 1 | Klasik model ansamblı (EAT/HOC — cutoff-ızgarasında GES) | 0024 | düşük-orta | M3/M4 tipi panelde sMAPE ~1-2 puan |
| 2 | `Differences` target transform + `SeasonalRolling` + tarih özellikleri (reduction) | 0025 | orta | trend/mevsim serilerinde belirgin |
| 3 | Tweedie objective (intermittency ipucuyla) | 0024/kat. | düşük | M5-tarzı kesikli talepte belirgin |
| 4 | Recursive multi-step reduction | 0004 rev. | orta | kısa-horizon çok-adım |
| 5 | Gerçek per_group_champion | 0015 rev. | orta | heterojen panelde |
| — | L2 stacking (tablo) | v1.1 | orta | marjinal (~+0.01 R²) |
| — | Hiyerarşik reconciliation (MinT) | v1.2 | yüksek | yalnız hiyerarşik veri |

**Tablo tarafında yeni bir şey gerekmiyor** — GBM+bagging+ensemble zaten SOTA reçetesi.
**Forecasting tarafında 5 somut adım var**, hepsi mevcut mimariye oturuyor (yeni model
altyapısı değil, mevcut yolların derinleştirilmesi). Neural forecasting (N-BEATS/PatchTST)
M5 2. sıra kullandı ama 1. sıra saf GBM'di → neural v1.1+ opsiyonel kalabilir.

## Kaynaklar
- M5: Makridakis et al. "M5 accuracy competition: Results, findings, conclusions" (IJF 2022);
  Artefact "Sales forecasting in retail: what we learned from M5".
- M3/M4: Hyndman "A forecast ensemble benchmark"; Pawlikowski & Chorowska (2020);
  "An Accurate and Fully-Automated Ensemble Model for Weekly Time Series Forecasting" (HOC2).
- MLForecast: Nixtla docs "Automated time series feature engineering"; mlforecast GitHub.
- Tweedie/intermittent: "Forecasting intermittent time series with GP and Tweedie likelihood"
  (IJF special issue); M5 findings.
- Reconciliation: "Forecast reconciliation: A review" (IJF 2024); "Rediscovering Bottom-Up".
- Tabular: TabArena (NeurIPS 2025); California housing / Adult / Covtype benchmark studies.
