# ADR 0033 — foundation model katmanı (TabPFN · Chronos)

**Durum:** Kabul · 2026-09-03 (kullanıcı 3 kararı kilitledi)

**Kilitli kararlar:**
1. **Kapsam: ikisi de** — tablo: **TabPFN** (PriorLabs, in-context PFN, sklearn sarımı — ADR 0030
   deseni). TS: **Chronos** (Amazon, zero-shot panel forecasting — ADR 0032 deseni). Nöral bloğu
   (0030-0033) tamamlanır.
2. **TS foundation slotu = yalnız Chronos.** `tabpfn-time-series` **dışlandı**: varsayılanı bulut
   client (kullanıcı verisi Prior Labs sunucusuna gider) → çekirdek "ağ yok" ilkesine (ADR 0003)
   aykırı. Chronos tamamen yerel + auth'suz ağırlık indirmesi.
3. **Ayrı bayrak + muhafazakâr auto.** Yeni `foundation_enabled: auto|on|off` (nöral bayrağından
   ayrı — lisans-token + büyük HF indirmesi hikâyesi farklı). `auto`: TabPFN yalnız satır/öznitelik
   bandında **ve** GPU varken; Chronos yalnız GPU varken. `on` → banttaysa/uygunsa her zaman.

Kaynak: v1.1 sırası (ADR 0032 sonrası). Foundation modeller ADR 0005'te "opsiyonel eklenti,
çekirdek bağımsız" olarak v1.2'ye konmuştu; kullanıcı v1.1 nöral bloğuna çekti. "Opsiyonel eklenti"
ilkesi korunuyor — `[foundation]` / `[foundation-ts]` extra'ları, çekirdek torch'suz.

## Araştırma özeti (2026-09)

- **TabPFN** (`pip install tabpfn`, PriorLabs): prior-data fitted network; `fit` = context'i belleğe
  al, `predict` = tek/az forward geçiş. sklearn uyumlu `TabPFNClassifier` / `TabPFNRegressor`.
  - **Limitler (TabPFN-3, Mayıs 2026 vars.):** 1M×200 · 100K×2000 · 1K×20000 (satır×öznitelik takası);
    sınıflandırmada ≤10 sınıf (değişmez).
  - **Lisans token:** ağırlık indirmesi için `TABPFN_TOKEN` (ilk kullanımda tarayıcıdan lisans kabulü;
    headless'ta ux.priorlabs.ai → token → env). İndirilen ağırlık **yerelde cache** → sonrası offline.
    Offline kurulum: `download_all_models.py` veya `TABPFN_MODEL_CACHE_DIR`.
  - GPU önerilir (~8GB VRAM yeter); CPU yalnız orta boyutta uygulanabilir, çok yavaş.
  - TabArena/TabPFN literatürü: **küçük-orta veride (≲10K satır) GBDT'yi ve tune-edilmiş nöralleri
    geçer**; büyük veride avantaj kapanır.
- **Chronos** (`pip install chronos-forecasting`, Amazon): T5 tabanlı, pretrained, **zero-shot**
  (fit yok). `torch>=2.2,<3` + `transformers>=4.49`.
  - **Chronos-Bolt** (9M–205M): patch-encoding, non-autoregressive → tek forward'da tüm quantile'lar;
    orijinal Chronos'tan ~250× hızlı, ~20× bellek-verimli, ~%5 daha düşük hata. Auth'suz HF indirmesi.
  - **Chronos-2** (120M): univariate→multivariate + covariate; güncel öneri. Bizde covariate yok
    (ADR 0009) → **Chronos-Bolt yeterli**; Chronos-2 opsiyonel katalog girişi.
  - `predict_df()` (chronos-forecasting ≥2.1) — long-format pandas (`id_column`/timestamp/target),
    quantile çıktısı aynı formatta. `BaseChronosPipeline.from_pretrained(name, device_map=)`.

## İlke

**Yeni engine yok.**
- **TabPFN** → `models/catalog/foundation.yaml`, sklearn sarımı (`FoundationTabEstimator`), mevcut
  tabular akışa girer (reduction dahil — forecasting'de lag-frame üstünde). ADR 0030 `real_mlp` gibi.
- **Chronos** → `engines/timeseries/foundation_ts.py`, native panel yolu (ADR 0023/0032 deseni).
  **fit yok** → "CV" = rolling-origin pencerelerde zero-shot tahmin + skor; serving = tek `predict_df`.
- Çekirdek torch/transformers'sız (ADR 0003) — yalnız `[foundation]` / `[foundation-ts]` extra.

## Karar

### 1. Modeller

**`models/catalog/foundation.yaml` (`[foundation]` extra) — tablo:**

| key | sınıf | not |
|---|---|---|
| `tabpfn` | `__foundation_tab__` → `models.foundation_tab.FoundationTabEstimator` | reg + clf (≤10 sınıf); in-context, HPO yok |

- `class_path` sentinel `__foundation_tab__` (ADR 0031 `__neural_arch__` deseni; registry `__x__`
  bypass). `family: foundation`. `requires: [tabpfn]`. `fidelity` yok (in-context — SH anlamsız).
- `default_params`: `{random_state: 42, n_estimators: 4}` (TabPFN iç ensemble tekrar sayısı; hız/kalite).
- HPO: **yok** (PFN'de hiperparametre minimal; `RandomSearchTuner` bu adayı pas geçer — `search_space`
  boş).

**`models/catalog/foundation_ts.yaml` (`[foundation-ts]` extra) — TS:**

| key | sınıf/checkpoint | not |
|---|---|---|
| `chronos_bolt` | `amazon/chronos-bolt-base` | varsayılan zero-shot forecaster |
| `chronos_bolt_small` | `amazon/chronos-bolt-small` | küçük panel / hız |
| `chronos_2` | `amazon/chronos-2` | opsiyonel — güncel, covariate (bizde kullanılmaz) |

- `family: foundation_ts`. `requires: [chronos]` (import adı `chronos`). Katalog `checkpoint:` alanı.
- `neural_search` **etkisiz** (zero-shot — aranacak mimari yok; `chronos_bolt_base` sabit).
- Model boyutu seçimi: `auto` → panel < ~50 seri veya kısa geçmiş → `_small`, aksi `_base`.

### 2. Entegrasyon

**TabPFN (`models/foundation_tab.py`):**
- `FoundationTabEstimator` — sklearn arayüzü (`fit(X, y)` / `predict` / `predict_proba` / `save` /
  `load`). İçte `TabPFNClassifier` | `TabPFNRegressor` (task'e göre). joblib-picklable **değil**
  (torch modülü) → `persistence.bundle` `_FOUNDATION_DIR` sidecar (ADR 0031 `_NEURAL_DIR` deseni).
- `estimator.build_estimator`: `class_path == "__foundation_tab__"` → `FoundationTabEstimator(
  **merged, task_kind=..., token_env=config.foundation_token_env)`.
- Token: `Settings.get(config.foundation_token_env)` → `TABPFN_TOKEN` ortam değişkenine yazılır
  (import öncesi). Yoksa + ağırlık cache'te yoksa → aday **atlanır** (tek WARNING; registry değil
  runtime kapısı — cache kontrolü `foundation_gate`'te).
- Bagging: nöral gibi **kapalı** (`champion._fit_pipeline` `want_bag=family not in {"neural",
  "foundation"}`). GES: OOF verir → normal yarışır (fit ucuz değil ama var).

**Chronos (`engines/timeseries/foundation_ts.py` — ADR 0023/0032 şablonu):**
- `run_foundation_ts_reports(frame, profile, task, config, candidates) -> (reports, extra_cands)`.
  Her aday için: rolling-origin pencereler (`_adaptive_windows`, `_MAX_CV_WINDOWS = 2`) →
  her pencerede `pipeline.predict_df(context=geçmiş, prediction_length=h)` → OOF birleştir →
  `ValidationReport` (pencere-indeksli fold gruplama — klasik/nöral-TS ile aynı).
- `FittedChronosForecaster` (`Predictor` protokolü) — `_context_df` (train sonu geçmiş) + `_pipeline`
  saklar; `predict(frame)` → `predict_df(prediction_length = _horizon_for(istenen ds))` + tarih-merge
  + fallback `_last` (ADR 0032 `FittedNeuralForecaster` ikizi). joblib-picklable **değil** →
  `_FOUNDATION_TS_DIR` sidecar: `_pipeline` yeniden `from_pretrained` (checkpoint adı + `_meta.npz`
  context/last).
- `run_core_pipeline(..., run_foundation_ts=config.foundation_enabled != "off")`. `_native_panel`
  filtresi genişler (classical ∪ neural_ts ∪ foundation_ts).
- `champion._foundation_ts_bundle` — refit yok (zero-shot); yalnız context'i tüm-veri sonuna kaydır.
- GES: cutoff-OOF sınıfı (klasik/nöral-TS ile aynı) → v1: ayrı yarışır.

### 3. Kapı + kaynak (`models/foundation_gate.py`)

Yeni modül (nöral kapıdan ayrı — farklı band + token/cache mantığı). `resolve_candidates` sonrası
`engines/core` çağırır (nöral kapı gibi).

- `foundation_enabled`: `auto` → GPU şart; `on` → her zaman; `off` → hiç.
- **TabPFN band (auto):** `n_rows ≤ foundation_tab_max_rows` (vars. 50_000) **ve**
  `n_features ≤ foundation_tab_max_features` (vars. 500) **ve** (clf ise) `n_classes ≤ 10`.
  `on` modda band tavanı 1M×200'e (kütüphane limiti) esner + CPU'ya izin (yavaş, kullanıcı seçti).
- **TabPFN token/cache:** token env çözülemiyor **ve** `TABPFN_MODEL_CACHE_DIR` / vars. cache boş →
  atla (WARNING: "TABPFN_TOKEN yok ve ağırlık cache boş").
- **Chronos band (auto):** GPU şart; `foundation_ts_min_series` (vars. 1 — tek seri de olur) +
  `foundation_ts_min_history_mult` (vars. 3× season). Model boyutu auto-seç.
- `configure_torch` (ADR 0030) determinizm — Chronos-Bolt tek forward → deterministik; TabPFN
  `random_state` sabit → deterministik.
- `foundation_device` override (`auto|cpu|cuda`) `default_params`'a / `device_map`'e enjekte.

### 4. Serving / persistence

`persistence.bundle`:
- `_FOUNDATION_DIR` (tablo) — `FoundationTabEstimator.save(dir)` (TabPFN `save_fit_state` veya
  fit context'i `.npz`); `saved_pipeline._estimator = None`. Sidecar kind `"tab"`.
- `_FOUNDATION_TS_DIR` (TS) — checkpoint adı + `_meta.npz` (context, `_last`); load → `from_pretrained`.
  Sidecar kind `"ts"`.
- Payload `foundation_sidecar` = kind (`"tab"` | `"ts"` | `""`). `load_bundle` ADR 0031/0032
  dallanmasına eklenir.

## Sözleşme (donacak)

- `models/catalog/foundation.yaml` (`tabpfn`) + `models/catalog/foundation_ts.yaml`
  (`chronos_bolt` / `chronos_bolt_small` / `chronos_2`).
- `RunConfig` (additive):
  - `foundation_enabled: Literal["auto","on","off"] = "auto"`
  - `foundation_device: Literal["auto","cpu","cuda"] = "auto"`
  - `foundation_token_env: str = "TABPFN_TOKEN"`
  - `foundation_tab_max_rows: int = 50_000` · `foundation_tab_max_features: int = 500`
  - `foundation_ts_min_series: int = 1` · `foundation_ts_min_history_mult: float = 3.0`
  - `tabular_fast` / hızlı preset'ler → `foundation_enabled: "off"`.
- `models/foundation_tab.py` — `FoundationTabEstimator` (sentinel `__foundation_tab__`).
- `models/foundation_gate.py` — `prepare_foundation_candidates(candidates, profile, task, config)`.
- `engines/timeseries/foundation_ts.py` — `run_foundation_ts_reports` + `FittedChronosForecaster` +
  `refit_foundation_ts` (context kaydırma).
- `engines/core.run_core_pipeline(run_foundation_ts=)` (additive-optional) + `_native_panel` genişler.
- `engines/champion` — `_foundation_ts_bundle`; `want_bag` `foundation` ailesini dışlar.
- `persistence.bundle` `_FOUNDATION_DIR` / `_FOUNDATION_TS_DIR` sidecar dallanması.
- `models/registry._class_exists` — `__foundation_tab__` zaten `__x__` bypass'ından geçer.
- `models/estimator.build_estimator` — `__foundation_tab__` dalı.
- `pyproject`: `foundation = ["tabpfn>=2.5"]` · `foundation-ts = ["chronos-forecasting>=2.1"]`.
  mypy override += `tabpfn.*`, `chronos.*`, `transformers.*`.
- `persistence.manifest._ENV_PACKAGES` += `tabpfn`, `chronos-forecasting`.
- `Candidate` / `EngineResult` / `ModelBundle` **değişmez**.

## Bağımlılık notu

- `chronos-forecasting>=2.1` → `torch>=2.2,<3` + `transformers>=4.49`. Mevcut `torch 2.14.0+cu126`
  (ADR 0032) karşılar. `transformers` yeni bağımlılık — pytorch-tabular/neuralforecast ile çakışmaz.
- `tabpfn>=2.5` → torch (mevcut yeter). `TABPFN_TOKEN` ağırlık indirmesi için (bkz. karar 3).
- Dört nöral/foundation extra (`neural` + `neural-nas` + `neural-ts` + `foundation` + `foundation-ts`)
  tek venv'de çalışır ama kırılgan — üretimde ayrı venv (ADR 0032 notu geçerli).

## Kapsam dışı / sonra

- `tabpfn-client` (bulut inference, kendi GPU'suz) — çekirdek "ağ yok" ilkesi; **hiç**.
- TabPFN-TS (yerel mod) → gerekirse v1.2 ADR (şu an Chronos yeterli).
- TabPFN fine-tuning / TabPFN-3 ekstra özellikleri → v1.2.
- Chronos fine-tuning (`chronos-forecasting` ≥2.1 destekliyor) → v1.2.
- Chronos covariate (Chronos-2) → ADR 0009 `Dataset.relations` açılınca.
- TimesFM / Moirai / TabPFN-TS bulut → v1.2 opsiyonel.
- Foundation TS + klasik + nöral-TS ortak GES backtest → v1.1 sonrası.

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `models/catalog/foundation.yaml` (`tabpfn`, `__foundation_tab__`) +
  `models/catalog/foundation_ts.yaml` (`chronos_bolt` / `chronos_bolt_small` / `chronos_2`=disabled).
- `models/foundation_tab.py` — `FoundationTabEstimator` (sklearn sarımı; `fit` context'i saklar,
  `predict`/`predict_proba`; `save`/`load` = context `.npz` + yeniden fit). `ensure_tabpfn_token`
  (`.env` → `TABPFN_TOKEN`) + `tabpfn_weights_cached` (offline cache yoklaması).
- `models/estimator.build_estimator` — `__foundation_tab__` dalı.
- `models/foundation_gate.prepare_foundation_candidates(candidates, profile, task, config)` —
  `foundation_enabled` (auto=GPU şart) × TabPFN bandı (`foundation_tab_max_rows/features` +
  ≤10 sınıf; `on` modda 1M×200'e esner) × token/cache kapısı × Chronos `foundation_ts_min_series` +
  model boyutu auto-seç (küçük panel → `_small`). Cihaz/token `default_params`'a enjekte.
- `engines/timeseries/foundation_ts.py` — `run_foundation_ts_reports` (rolling-origin zero-shot
  `predict_df` OOF; `_MAX_CV_WINDOWS=2`; `BaseChronosPipeline.from_pretrained`) +
  `FittedChronosForecaster` (`Predictor`; context'ten `predict_df`, tarih-hizalı, `_last` fallback;
  `save`/`load` = checkpoint adı + `context.parquet` + `_meta.npz`) + `refit_foundation_ts`
  (fit yok — context'i tüm-veri sonuna kaydır).
- `engines/core.run_core_pipeline(run_foundation_ts=)` + `_native_panel` genişledi
  (classical ∪ neural_ts ∪ foundation_ts).
- `engines/champion._foundation_ts_bundle`; `want_bag` `foundation` ailesini dışlar; recursive
  yol guard'ı `{neural_ts, foundation_ts}` ailelerini de dışlar.
- `engines/timeseries/core_engine.py` — `_run_pooled` + `_run_recursive` `run_foundation_ts` geçirir.
- `persistence/bundle.py` — `_FOUNDATION_DIR` / `_FOUNDATION_TS_DIR` sidecar; `neural_sidecar`
  discriminator `foundation_tab` / `foundation_ts` değerleri eklendi; `_sidecar_estimator` /
  `_sidecar_pipeline` genelleştirildi.
- `persistence/manifest._ENV_PACKAGES` += `tabpfn` / `chronos-forecasting` / `transformers`.
- `RunConfig` (additive): `foundation_enabled` / `foundation_device` / `foundation_token_env` /
  `foundation_tab_max_rows` / `foundation_tab_max_features` / `foundation_ts_min_series` /
  `foundation_ts_min_history_mult`. `tabular_fast` preset → `foundation_enabled: "off"`.
- `pyproject` `foundation` / `foundation-ts` extra + mypy override (`tabpfn.*` / `chronos.*` /
  `transformers.*`).
- Testler: `tests/unit/models/test_foundation.py` (10 — katalog, kapı bandı/token/boyut-seçimi,
  `is_foundation_ts`, lib-yok atlama). tabpfn/chronos kurulu değil → e2e skipif ile ertelendi.

**Benchmark → tüm nöral blok (0030-0033) bitince (kullanıcı kararı).**
