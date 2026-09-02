# ADR 0025 — zengin reduction özellikleri (MLForecast paritesi)

**Durum:** Kabul · 2026-09-02

Kaynak: SOTA gap analizi Gap #2. Nixtla MLForecast (M5 tarzı tabular-reduction referansı):
`Differences` target transform, `SeasonalRollingMean/Std`, çoklu pencere/lag rolling,
tarih özellikleri. Benchmark: m5 wMAPE 105 — temel lag/rolling yetmiyor.

## İlke korunur

**Leakage-safe by construction:** hedef-türevi her özellik `shift(≥ horizon)` tabanlı →
h-adım direkt tahminde test satırı yalnız train dönemi `y`'sini görür. Takvim özellikleri
`time_col`'dan doğrudan (gelecek tarihler için de bilinir — sızıntı yok).

## Eklenen özellikler (`build_reduction_features`)

### 1. Takvim / tarih özellikleri (koşulsuz, `add_calendar=True`)
`time_col`'dan: `month`, `quarter`, `weekofyear`, `dayofweek`, `dayofyear`,
`is_month_start/end`, `is_quarter_start/end` + **döngüsel kodlama** `sin/cos(2π·month/12)`,
`sin/cos(2π·dayofweek/7)`. Ağaç modelleri ham int'i, lineer modeller sin/cos'u kullanır.

### 2. Mevsim-hizalı lag'ler (`season` verilirse)
Mevcut `y_lag_{h..h+3}`'e ek: `y_lag_{H}`, `y_lag_{H+s}`, `y_lag_{H+2s}` — burada
`H = ceil(h/s)·s` (h'yi aşan ilk mevsim katı). Aynı mevsimdeki geçmiş dönemler.

### 3. Mevsimsel rolling (`season` verilirse)
`base_s = [shift(H), shift(H+s), shift(H+2s), shift(H+3s)]` üzerinde grup-içi `mean`/`std`
→ `y_seasonal_rollmean`, `y_seasonal_rollstd`. Aynı-mevsim ortalaması (MLForecast `SeasonalRollingMean`).

### 4. Genişletilmiş rolling
Mevcut `rollmean/std_{4,8,13}`'e ek: `rollmin_{w}`, `rollmax_{w}` ve pencereye `season` eklenir
(varsa). `base = shift(horizon)` üzerinde.

### 5. Fark (difference) özellikleri
- `y_diff1_lag_{h}` = `shift(h) − shift(h+1)` (kısa-vadeli momentum)
- `y_diffs_lag_{h}` = `shift(h) − shift(h+s)` (mevsimsel değişim, `season` verilirse)

Bunlar **özellik**tir — hedef dönüştürülmez (direct h-step'te first-difference'ın tersine
çevrilmesi belirsiz). **Gerçek (seasonal) target differencing transform → v1.1** (`s ≥ h`
şartıyla tersine çevrilebilir; ayrı iş).

## API değişikliği

`build_reduction_features(frame, task, *, horizon, season=1, add_calendar=True)`
→ `(frame, new_cols)`. `TimeSeriesCoreEngine` `season`'ı `_season_length(profile, freq)`'ten
(ADR 0023 helper'ı) geçirir; `pre_transform` `functools.partial` bu argümanları taşır.

Yeni sözleşme yok. `RunConfig` knob yok — özellikler otomatik; model örtük seçim yapar
(ağaçlar gürültülü özelliği görmezden gelir). Lineer/MLP için sayı fazlaysa
`preprocessors` zaten scale/impute uyguluyor.

## Kapsam dışı / sonra
- **Gerçek (seasonal) target differencing** transform (`Differences([s])`) → v1.1.
- **Recursive multi-step** reduction (`shift(1)` + adım-adım besleme) → ADR 0026.
  M5-rekabetçi intermittent için bu + per-grup (Gap #5) gerekli.
- Exogenous (price/promo) özellikler → Dataset.relations rezerve, v1.1.

## Sonuç
- `reduction.py` ~15 → ~40 özellik (takvim + mevsim-hizalı lag + mevsimsel rolling + fark).
- Hepsi leakage-safe (`shift ≥ horizon` veya takvim).
- benchmark ile ölçülür (m3/tourism trend+mevsim; m5 kısmi iyileşme beklenir).
