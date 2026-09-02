# ADR 0026 — recursive multi-step reduction + seasonal target differencing

**Durum:** Kabul · 2026-09-02

Kaynak: SOTA gap analizi Gap #4. M5 winner: recursive + non-recursive LightGBM ortalaması.
AutoGluon-TS: `RecursiveTabular` + `DirectTabular` birlikte. MLForecast: `Differences([s])`
target transform. Benchmark: m5 wMAPE 105 (direct-only + temel özellik).

## Bölüm A — seasonal target differencing (`s ≥ h` şartıyla)

### Karar
`build_reduction_features(..., seasonal_diff=None)` — `season ≥ horizon` ve
`seasonal_diff` aktif → **eğitim hedefi** `y_t − y_{t−s}` olur (per-series). Ham `y`
korunur; ayrı `__y_target__` kolonu üretilir.

**Leakage-safe:** horizon satırı `t` için (`1 ≤ t−cutoff ≤ h ≤ s`) → `t−s ≤ cutoff` →
`y_{t−s}` train aktüeli. Tersine çevirme: `y_hat = pred + y_slag_s` (zaten üretilen
`shift(s)` özelliği).

### Threading
- reduction `__y_target__` + `seasonal_diff_ref` (= `y_slag_s` kolon adı) döndürür.
- `champion._fit_one` / `validators`: hedef `__y_target__` ise onu kullanır, `TargetTransform`
  yine üstüne binebilir.
- `FittedModelPipeline` `seasonal_diff_ref: str | None` slotu — `predict`'te
  `target_transform.inverse` **sonrası** `out += frame[ref]`.
- Otomatik: `s ≥ h` ve (trend_strength yüksek veya freq mevsimsel) → planner `candidate_ops`
  `target` grubuna `seasonal_difference` seçeneği ekler; `hpo_level=none` bunu **varsayılan** yapar.

## Bölüm B — recursive multi-step reduction

### Karar
`RunConfig.forecast_reduction: Literal["direct", "recursive"] = "direct"`.
`recursive` → `build_reduction_features(strategy="recursive")`:
- lag'ler `1..max_lag` (`shift(1)` tabanı), rolling/ewm `shift(1)` üzerinde, mevsim-hizalı
  `s, 2s, ...`, takvim aynı.
- Model **1-adım-ileri** eğitilir/CV'lenir (mevcut `run_validation` değişmeden çalışır —
  reduction frame'inde `y_t`'yi `y_{t−1..}` ile regresyon).
- **Serving: `FittedRecursivePipeline`** — per-series `h` adım döngüsü: adım `k`'de tüm
  seriler için özellik satırı kur (bilinen geçmiş + `k−1` önceki tahmin + gelecek `ds`
  takvimi), batch predict, tahmini geçmişe ekle.

### v1 sınırı (açık)
CV **1-adım** skoru ölçer; recursive-`h` birikimli hata v1'de ölçülmez (model seçimi için
1-adım güçlü proxy — AutoGluon `RecursiveTabular` da 1-adım eğitir). Recursive-h CV → v1.1.
`weighted_ensemble` recursive + direct'i **karıştırmaz** (OOF farklı anlam) — v1'de
`forecast_reduction` tek strateji seçer; "both" (AutoGluon deseni) → v1.1.

### `FittedRecursivePipeline`
`__slots__`: fitted feature pipeline + estimator + target_transform + recipe (max_lag,
season, calendar) + group_col/time_col/horizon/target + `seasonal_diff_ref`. `Predictor`
protokolü. joblib picklable (reduction recipe saf param).

## Sözleşme
- `RunConfig.forecast_reduction`.
- `FittedModelPipeline.seasonal_diff_ref: str | None` (additive slot).
- `build_reduction_features` → `(frame, new_cols, target_col, sdiff_ref)` (target_col =
  `y` veya `__y_target__`).

## Kapsam dışı / sonra
- Recursive-h CV (birikimli hata ölçümü) → v1.1.
- Direct + recursive **ensemble** (AutoGluon deseni) → v1.1.
- İlk-fark (`Differences([1])`) target transform — direct h-step'te tersi belirsiz → yalnız
  recursive'de anlamlı, v1.1.

## Sonuç
- `reduction.py` `strategy` + `seasonal_diff` parametreleri.
- `engines/timeseries/recursive.py` — `FittedRecursivePipeline` + recursive predict döngüsü.
- `champion` / `core_engine` `forecast_reduction`'a göre yol seçer.
- benchmark: m5 (recursive), m3/tourism (seasonal_diff).
