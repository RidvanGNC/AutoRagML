# ADR 0029 — anchored klasik forecaster'lar arbitrary gelecek penceresini serve edebilir

**Durum:** Kabul · 2026-09-02

Kaynak: m5 benchmark. Orchestrator (ADR 0020) `frame_full`'dan nihai holdout'u carve eder,
engine `train = full − holdout` üzerinde CV + şampiyon fit yapar, şampiyon serving'e döner.

- **Feature-tabanlı modeller (reduction/tabular):** `predict(frame)` verilen frame'in lag/takvim
  özelliklerinden çalışır → arbitrary gelecek penceresini **doğru serve eder** (repro ile
  doğrulandı: benchmark holdout wMAPE ≈ orchestrator holdout wMAPE). Son `holdout` dönemi kadar
  eğitim eksik ama etki küçük.
- **`FittedClassicalForecaster` (anchored, `StatsForecast`): serving BOZUKTU.** `predict` sabit
  `sf.predict(h=self._h)` çağırıyordu → yalnız **fit sonrası ilk `h` adım**. Şampiyon
  `full − holdout`'ta fit edilince bu pencere = holdout dönemi; gerçek gelecek (`full` sonrası)
  hiç üretilmiyor → tarih-merge tutmaz → tüm satırlar "son değer" fallback'i. (m5: 3 farklı
  klasik şampiyonda byte-identik wMAPE 105.2.)

## Karar

`FittedClassicalForecaster.predict(frame)` istenen tarih penceresini kapsayacak kadar ileri
tahmin eder:

- `_train_end` (fit verisinin son `ds`'i) + `_freq` saklanır.
- `_horizon_for(target_max)` = `train_end → target_max` arası freq-periyot sayısı, `[self._h,
  self._h·24 + 366]` aralığına clamp (runaway koruması). `target_max ≤ train_end` → `self._h`.
- `sf.predict(h=_horizon_for(...))` → istenen pencere forecast kolonlarında; merge artık hit eder.

Orchestrator akışı / `EngineResult` / `ModelBundle` **değişmez** — tek noktalı düzeltme
(`engines/timeseries/classical.py`). Reduction modellerinin "son holdout kadar eksik eğitim"
kaybı v1'de kabul (küçük; tam-veri refit → v1.1, orchestrator `champion_refit_full` aşaması).

## Sözleşme

- `FittedClassicalForecaster.__init__(..., freq: str)` (additive-required — tek çağıran
  `_fit_forecaster`).
- Davranış: `predict` artık `frame`'in `ds` aralığına göre değişken ufuk kullanır.

## Kapsam dışı / sonra (v1.1)

- Orchestrator `champion_refit_full`: nihai holdout skorlandıktan sonra şampiyonu `frame_full`
  üzerinde yeniden fit (feature-tabanlı modellere de güncellik kazandırır; metrikler küçük-veri
  skorunda kalır). Tasarımı: `run_core_pipeline` refit closure'ı + `finalize_champion` (skor →
  swap sırası kritik). Segmented ile etkileşimi ayrıca çözülür.
- Benchmark harness `champion_test_score` artık klasik için de anlamlı olacak — RESULTS güncellenir.
