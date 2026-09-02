# ADR 0026 — recursive multi-step reduction + seasonal target differencing

**Durum:** Kabul · 2026-09-02

Kaynak: SOTA gap analizi Gap #4. M5 winner: recursive + non-recursive LightGBM ortalaması.
AutoGluon-TS: `RecursiveTabular` + `DirectTabular` birlikte. MLForecast: `Differences([s])`
target transform. Benchmark: m5 wMAPE 105 (direct-only + temel özellik).

## Bölüm A — seasonal target differencing (`s ≥ h` şartıyla) · UYGULANDI (commit `bc4c248`)

### Karar (uygulanan biçim)
`seasonal_difference`, **`TargetTransform` seçeneği** olarak eklendi (ayrı `__y_target__`
kolonu değil — hedef dönüşüm katmanı zaten bu iş için var, ADR 0011).
`FittedTargetTransform.forward(y, ref) = y − ref` / `inverse(y, ref) = y + ref`;
`ref` = reduction'ın ürettiği `y_sdiff_ref` kolonu (`= grouped.shift(H)`,
`H = ceil(h/s)·s ≥ h`).

**Leakage-safe:** horizon satırı `t` için `t − H ≤ cutoff` → `y_{t−H}` train aktüeli.

### Threading (uygulanan)
- `reduction.build_reduction_features` → seasonal + `direct` iken `y_sdiff_ref` kolonu üretir.
- `validators.frame_ops.sdiff_ref(frame, target, choice)` — kolonu `_Arr | None` döndürür;
  `runner` ve `champion._fit_one` warmup `ref=NaN` satırlarını düşürür, `tt.forward(y, ref)`
  ile eğitir, `tt.inverse(pred, ref)` ile geri çevirir.
- `FittedModelPipeline._target_ref_col: str | None` slotu — `predict`'te transform sonrası
  frame'den `ref` çekip `target_transform.inverse(raw, ref=ref)`.
- Otomatik: `dynamics.planner._seasonal_diff_applicable` → `candidate_ops` `target` grubuna
  `seasonal_difference` ekler; `s ≥ h` + (mevsim gücü ≥ 0.3 veya trend ≥ 0.3) → **varsayılan**.

## Bölüm B — recursive multi-step reduction · UYGULANDI

### Karar (uygulanan)
`RunConfig.forecast_reduction: Literal["direct", "recursive"] = "direct"`.
`recursive` → `build_reduction_features(strategy="recursive")`:
- lag'ler `1..k_max` (`k_max = max(h, 3s, 12)`), rolling/ewm/min-max `shift(1)` tabanı,
  mevsim-hizalı `H, H+s, H+2s`, fark `y_diff1_lag_1`/`y_diffs_lag_1`, takvim aynı.
- `y_sdiff_ref` **üretilmez** (seasonal target differencing yalnız `direct`).
- Model **1-adım-ileri** eğitilir; hedef dönüşümü yalnız `none`/`log1p`.

### CV — recursive-`h` (spec'i aşıyor)
`engines/timeseries/recursive.run_recursive_reports`: rolling-origin fold'da 1-adım model
`aug.iloc[train_idx]` üzerinde fit → test bloğu **recursive-h** tahmin edilir (her adım
özellikler yeniden kurulur, tahmin geri beslenir). OOF = birikimli-hata skoru → model seçimi
gerçek serving davranışını ölçer. (ADR taslağı bunu v1.1'e ertelemişti; birlikte geldi.)

`weighted_ensemble` recursive modda **devre dışı** — recursive şampiyon ansambl refit yolundan
(`_fit_one` direct özellik kurar) geçemez. `forecast_reduction` tek strateji seçer; direct +
recursive "both" (AutoGluon deseni) → v1.1.

### `FittedRecursivePipeline`
`__slots__`: fitted feature pipeline + estimator + target_transform + `RecursiveRecipe`
(`task`, `season`, `add_calendar`, `horizon`) + `feature_cols` + `reserved`. `Predictor`
protokolü (`predict`, `feature_cols`). `predict(frame)` → her seri **son `horizon` satır**
recursive; kalan satırlar `NaN`. joblib picklable (recipe saf param, `TaskSpec` pydantic).

## Sözleşme (uygulanan)
- `RunConfig.forecast_reduction: Literal["direct", "recursive"]`.
- `TargetTransform`/`FittedTargetTransform` `seasonal_difference` seçeneği +
  `forward/inverse(y, ref=None)`.
- `FittedModelPipeline.target_ref_col: str | None` (additive slot).
- `build_reduction_features(..., strategy=, max_lag=)` → `(frame, new_cols)` imzası korunur;
  seasonal + direct iken `y_sdiff_ref` kolonu `new_cols` içinde.
- `validators.frame_ops.sdiff_ref` / `sdiff_ref_col`.
- `run_core_pipeline(..., recursive=, recursive_season=)`;
  `champion.refit_champion(..., recursive_season=)`.

## Kapsam dışı / sonra
- Direct + recursive **ensemble** (AutoGluon deseni) → v1.1.
- İlk-fark (`Differences([1])`) target transform — direct h-step'te tersi belirsiz → yalnız
  recursive'de anlamlı, v1.1.
- Recursive modda linear/knn adayları (lag NaN warmup) — engine yolunda planner `impute`
  ekler; standalone yolda GBDT-dışı adaylar fold try/except ile atlanır.

## Sonuç
- `reduction.py` `strategy`/`max_lag` parametreleri + seasonal differencing kolonu.
- `preprocessors/target.py` `seasonal_difference` + ref'li forward/inverse.
- `engines/timeseries/recursive.py` — `RecursiveRecipe` + `FittedRecursivePipeline` +
  `run_recursive_reports` (recursive-h CV) + `fit_recursive_champion`.
- `core.py` / `core_engine.py` / `champion.py` `forecast_reduction`'a göre yol seçer.
- Testler: `test_reduction.py` (+recursive lag leakage), `test_planner.py` (+seasonal_diff
  default), `test_engines_e2e.py` (+recursive uçtan uca).
- benchmark: m5 (recursive), m3/tourism (seasonal_diff) — sonraki adım.
