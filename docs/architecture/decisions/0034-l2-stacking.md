# ADR 0034 — L2 stacking (stacker katmanı)

**Durum:** Kabul · 2026-09-03 (kullanıcı 3 kararı kilitledi)

**Kilitli kararlar:**
1. **K1 = saf stacking** — L2 yalnız L1 OOF tahmin matrisi `Z` üstünde (orijinal öznitelik yok,
   FeaturePipeline gerekmez). Restacking → kapsam dışı.
2. **K3 = tüm L1 tipleri + tek katman** — başarılı L1 adaylarının her **çeşit** tipi (class_path'e
   göre tekilleştirilmiş) bir L2 stacker olur. `stack_levels` alanı rezerve (L3+ yok).
   **İstisna:** `neural` / `foundation` aileleri stack katmanına girmez (ne üye ne stacker) —
   6-sütunlu `Z` üstünde nöral anlamsız + joblib-picklability riski; stacking'in fayda gösterdiği
   yer yapılandırılmış tablo (GBDT/linear/forest).
3. **K4+K5 = muhafazakâr auto + hafif guard** — `stacking_enabled="auto"`: `n_rows ≥
   stacking_min_rows(2000)` + `≥ stacking_min_families(4)` çeşitli L1 ailesi + `≥5` fold; altında
   atlanır. **Guard:** bir L2 stacker OOF'u en iyi L1 OOF'unu (primary metrik) geçmiyorsa o stacker
   hiç önerilmez. Ek koruma: ADR 0020 holdout + ADR 0014 1-SE + `_FAMILY_COMPLEXITY["stack"]=6`.

Kaynak: v1.1 sırası (ADR 0033 sonrası). SOTA gap analizi (2026-09): "Tablo: SOTA'dayız...
**Marjinal: L2 stacking.**" AutoGluon'un çok-katmanlı stack ensembling'i tek en büyük mimari
fark; bizde ADR 0021 (GES) + ADR 0022 (k-fold bagging) var ama **stacker katmanı yok**.

## Araştırma özeti (2026-09)

- **AutoGluon multi-layer stacking:** L1 taban modelleri → OOF tahminleri **concat** → L2 stacker
  modelleri (aynı model tipleri, aynı HP). Stacker girdisi = **orijinal öznitelikler + alt-katman
  OOF tahminleri** ("restacking"). Son katman = penultimate layer üstünde GES.
- **Leakage-free:** L2 yalnız L1 **OOF** tahminleri üstünde eğitilir (k-fold bagging her satır için
  OOF tahmin üretir → L2 aynı veri miktarını kullanır). ADR 0011/0022 bunu zaten sağlıyor.
- **Stacked overfitting** (gerçek risk): L1 OOF tahminleri hâlâ hafif iyimser → L2 bunu ezberleyip
  holdout'ta çöker. AutoGluon **dynamic stacking** ekledi (iç holdout'ta stacked-overfitting tespit
  → stacking'i kapat). Bizde ADR 0020 orchestrator holdout + ADR 0014 1-SE zaten kısmi koruma.
- **TS forecasting:** "Multi-layer Stack Ensembles for Time Series Forecasting" (arXiv 2511.15350,
  2025) — aynı desen rolling-origin OOF ile forecasting'e uygulanıyor.

## İlke

- **Yeni engine yok.** `ensembling/stacking.py` — `build_weighted_ensemble` (ADR 0021) ikizi:
  hizalı L1 OOF matrisi → L2 stacker adayları → sentetik `ValidationReport` + `Candidate` →
  `engines/core` bunları `reports`/`candidates`'e ekler → `score_reports` (1-SE) hepsini yarıştırır
  (`_FAMILY_COMPLEXITY["stack"]` ensemble'dan da yüksek → eşitse basit kazanır).
- **Hizalama kısıtı ADR 0021 ile aynı:** yalnız nested-CV suite adayları (reduction/tabular);
  klasik/nöral-TS/foundation-TS OOF'u cutoff-tabanlı → L2'ye giremez (GES gibi).
- **Leakage-safe by construction (ADR 0011):** L2 CV, L1 ile **aynı splitter/fold** üstünde;
  her fold'da stacker yalnız o fold'un L1-OOF sütunlarıyla eğitilir. `fit` yalnız `validators`/
  `stacking` içinde.

## Açık kararlar (kullanıcıya sorulacak)

### K1 — Restacking: L2 girdisi
- **(a) Saf stacking:** L2 yalnız L1 OOF tahminleri (`Z`, tamamen sayısal, FeaturePipeline gerekmez).
  Basit, düşük overfit riski, ~40 satır. Meta-öğrenici klasik anlamı.
- **(b) Restacking (AutoGluon):** L2 girdisi = orijinal öznitelikler + L1 OOF. Daha güçlü ama
  FeaturePipeline'ı L2 için threading + daha yüksek stacked-overfitting riski.
- **Öneri: (a)** — SOTA analizi "marjinal" diyor; saf stacking temiz kazanç, restack fayda
  gösterirse v1.1+ genişletme.

### K2 — Katman derinliği
- **(a) Tek L2 katmanı** (L3+ yok). **(b) Yapılandırılabilir derinlik** (`stack_levels: int`).
- **Öneri: (a)** — L3 getiri iyice azalıyor, maliyet katlanıyor; `stack_levels` alanı rezerve edilir.

### K3 — Stacker model seti
- **(a) Küratörlü küçük set:** `lightgbm` + `ridge`/`linear` + (varsa) `real_mlp` — meta-öğrenici
  için yeterli, hızlı. **(b) Tüm L1 tipleri** (AutoGluon).
- **Öneri: (a)** — meta-katmanda 2-3 çeşitli öğrenici yeterli; tam set maliyeti L2'de haklı değil.

### K4 — Etkinleşme
- **(a) Opt-in bayrak** (`stacking_enabled: bool = False`). **(b) Auto** (veri boyutu + L1 çeşitliliği
  kapısı; `hpo_level=thorough` veya `n_rows ≥ eşik` + `≥4` çeşitli L1 ailesi).
- **Öneri: (b) auto ama muhafazakâr** — `n_rows ≥ 2000` + `≥4` başarılı L1 ailesi + `≥5` fold;
  altında atlanır. `stacking_enabled: "auto"|"on"|"off"`.

### K5 — Stacked-overfitting koruması
- **(a) Mevcut mekanizmalar yeter:** ADR 0020 holdout + ADR 0014 1-SE + `_FAMILY_COMPLEXITY`
  (stack en karmaşık → 1-SE bandında basit L1/GES kazanır).
- **(b) Dynamic-stacking tespiti:** iç holdout'ta L2-OOF ↔ L2-holdout sapması ölç, eşik aşılırsa
  L2 adaylarını karantinaya al.
- **Öneri: (a) + hafif guard** — `stacking.py` L2 şampiyon OOF'u L1 en iyisini `> noise_floor`
  geçmiyorsa L2'yi hiç önermez (GES zaten var). Tam dynamic stacking → v1.1+.

## Sözleşme (donacak)

- `ensembling/stacking.py` — `build_stack_layer(reports, candidates, frame, plan, profile, task,
  config, tuner) -> list[tuple[ValidationReport, Candidate]]` + `STACK_KEY_PREFIX = "stack_"`.
- `Candidate` (mevcut alanlar yeter): `family="stack"`, `class_path="__stack__"`,
  `ensemble_members` = L1 üye anahtarları (ağırlık yerine "hepsi girdi" → 1.0), `default_params`
  `{"base_model": "lightgbm"}` (hangi sklearn stacker).
- `RunConfig` (additive): `stacking_enabled: Literal["auto","on","off"] = "auto"`,
  `stacking_min_rows: int = 2000`, `stacking_min_families: int = 4`, `stack_levels: int = 1` (rezerve).
- `engines/core.run_core_pipeline` — GES'ten **önce** `build_stack_layer` (L2 adayları GES havuzuna
  da girer → "GES over penultimate layer"). `recursive` modda kapalı (ADR 0026 gibi).
- `engines/champion` — `_stack_bundle`: L1 üyeleri (bagged) + L2 stacker refit →
  `FittedStackPipeline` (`engines/stack_pipeline.py`): `predict` = L1 üye tahminleri → `Z` → L2.
- `persistence` — `FittedStackPipeline` joblib-picklable (sklearn L1+L2) → sidecar gerekmez.
- `scoring/selection._FAMILY_COMPLEXITY["stack"] = 6` (ensemble=5'ten yüksek).
- `Candidate`/`EngineResult`/`ModelBundle` **değişmez**.

## Kapsam dışı / sonra

- Restacking (orijinal öznitelik + L1 OOF) → v1.1+.
- L3+ katmanlar → `stack_levels` rezerve.
- Dynamic stacking tespiti → v1.1+.
- `neural`/`foundation` ailelerini stack'e katma (üye veya stacker) → v1.1+ (picklability + değer).
- **GES over penultimate layer**: v1'de L2 stacker'lar GES üyesi değil (GES = L1 blend);
  stacker'lar 1-SE seçiminde **doğrudan** yarışır → v1.1+.
- Klasik/nöral-TS/foundation-TS'i stack'e katma (ortak backtest ızgarası) → v1.1 sonrası.
- Sınıflandırma stacking (proba OOF) → ADR 0036 (sınıflandırma GES) ile birlikte.

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `ensembling/stacking.py` — `build_stack_layer(reports, candidates, task, config) ->
  list[(ValidationReport, Candidate)]`. Hizalı L1 OOF (`_aligned_reports`, ADR 0021 helper) →
  `Z` (nan_to_num) → her **çeşit** taban (family|class_path tekil) için L2 k-fold OOF
  (`_l1_fold_indices` = L1 fold sınırlarını yeniden kullan → stacker OOF L1 ile hizalı kalır).
  Guard: L2 OOF, en iyi taban-L1 OOF'unu geçmiyorsa stacker atlanır. `_EXCLUDED_FAMILIES` =
  ensemble/stack/neural*/foundation*.
- `engines/stack_pipeline.FittedStackPipeline` — L1 üye tahminleri → `Z` → L2 estimator →
  (opsiyonel) stack-düzeyi postprocess. `pre_transform` bir kez (TS reduction). joblib-picklable.
- `engines/core.run_core_pipeline` — GES'ten önce `build_stack_layer`; stacker'lar
  `reports`/`candidates`'e eklenir (1-SE havuzu). `recursive` modda kapalı.
- `engines/champion._stack_bundle` — L1 üyeleri `_fit_pipeline` (bagged, postprocess'siz) +
  L2 = `build_estimator(base_cand)` L1 **OOF** matrisi üstünde fit. `refit_champion`
  `family == "stack"` → `_stack_bundle`. Recursive guard'a `stack` eklendi.
- `ensembling/__init__.build_weighted_ensemble` — `family == "stack"` GES üyeliğinden çıkarıldı.
- `scoring/selection._FAMILY_COMPLEXITY["stack"] = 6`.
- `RunConfig` (additive): `stacking_enabled` (`auto`/`on`/`off`), `stacking_min_rows` (2000),
  `stacking_min_families` (4), `stack_levels` (1, rezerve — `le=1`).
- Testler: `tests/unit/ensembling/test_stacking.py` (6 — katalog/kapı/guard/refit round-trip/
  zorla-şampiyon serving).

**Benchmark → tüm v1.1 eklemeleri bitince (kullanıcı kararı).**
