# AutoRagML

Modalite-agnostik, **deterministik** AutoML çekirdeği. İlham kaynağı: bir talep-tahmini
(demand sensing) MLOps hattının modelleme bel kemiği — model kataloğu → rolling-origin
backtest → guardrail'li çok-metrikli skorlama → per-group champion.

> **Durum: v1.1 — geniş model havuzu + robustluk turu tamam (pre-alpha, henüz sürüm yayımlanmadı).**
> 15 katman (`contracts` → `interfaces`) + ensembling/stacking + nöral tablo/forecasting +
> foundation modeller (TabICL, Chronos, TimesFM) + hiyerarşik reconciliation + split-conformal
> tahmin aralıkları + açıklanabilirlik. Ayrıntı: [`docs/architecture/decisions/`](docs/architecture/decisions/)
> (46 ADR) ve [`CHANGELOG.md`](CHANGELOG.md).
>
> ```python
> from autoragml import AutoRagML
>
> model = AutoRagML(preset="tabular_fast")
> result = model.fit(df, target="sales")
> result.leaderboard()
> result.predict(new_df)
> result.predict_interval(new_df, coverage=0.9)   # split-conformal (ADR 0044)
> result.explain(new_df)                          # SHAP/permutation atıf (ADR 0037)
> ```

## Hedef

- **pip paketi veya container** ile hızlı çıkarılabilir; kolay entegre; **tool olarak**
  kullanılabilir (Python API + CLI + ajan tool şeması).
- **Bulut zorunluluğu yok.** Tracking/registry/storage ve LLM sağlayıcı = opsiyonel eklenti;
  local dosya + no-op varsayılan. Foundation modeller (TabICL/Chronos/TimesFM) local çalışır,
  yalnız ilk indirmede Hugging Face'e erişir.
- **Çok modalite yol haritası:** v1 tablo + zaman serisi (tam); v1.1+ genişleme (bu sürüm);
  v1.2 metin/görsel/ses + kovaryant/exogenous destek; v2 RAG/agent üst katmanı.
- **Çapraz platform:** çekirdek saf-Python (numpy/pandas/pyarrow/scikit-learn/lightgbm/pydantic),
  Windows/macOS/Linux CI. Ağır DL bağımlılıkları (torch tabanlı) tamamen opsiyonel extra'larda.

## Neler var

**Tablo:** sklearn ailesi + LightGBM/XGBoost/CatBoost + EBM (glassbox GAM) + NGBoost + KNN/SVR +
TabICL/TabPFN (in-context foundation) + nöral tier (RealMLP/TabM, opsiyonel mimari arama) +
Caruana greedy-ensemble + k-fold bagging + **L2 stacking** (ADR 0034) + sınıflandırma GES
(olasılık OOF, ADR 0036).

**Zaman serisi (forecasting):** leakage-safe reduction (lag/rolling/takvim, ~40 özellik) +
native klasik aile (AutoARIMA/AutoETS/Theta/CES/TBATS/Croston/IMAPA/ADIDA, `StatsForecast`) +
nöral tier (NHITS/PatchTST/TFT/NBEATSx/iTransformer/TSMixer/**TiDE**, `neuralforecast`) +
foundation-TS (**Chronos-Bolt**, **TimesFM 2.5** — zero-shot) + klasik+reduction ortak GES
(joint ensemble) + recursive/direct multi-step + seasonal target differencing + SBC
intermittency-tabanlı **segmentasyon** (ADR 0028) + deneysel **per-series şampiyon** (ADR 0046) +
**hiyerarşik reconciliation** (MinTrace/wls_struct, ADR 0045).

**Seçim/robustluk:** 1-SE kuralı + aile-karmaşıklığı tie-break + kararsız-CV filtresi (ADR 0038)
+ ince-kanıt/kontamine-OOF marj guard'ı (foundation_ts, ADR 0042) + metrik-duyarlı promotion
(kesikli talepte yüzde-tavan atlanır, ADR 0039).

**Serving:** postprocess zinciri (calibrate → clip → **split-conformal aralık** → round →
business-rule, ADR 0017 + 0044) + `explain()` (SHAP veya model-agnostik permutation, ADR 0037) +
`champion_refit_full` (train+holdout'ta son fit, ADR 0035).

## Kilitlenen mimari kararlar

Ayrıntı: [`docs/architecture/decisions/`](docs/architecture/decisions/) (ADR 0001–0046).

| aşama | kapsam |
|---|---|
| **v1 iskelet** (ADR 0001–0020) | Paket/kapsam kararları · `RunConfig`/`Dataset`/`AdaptivePlan` sözleşmeleri · 15 katman (contracts→interfaces) · leakage-safe validasyon · 1-SE seçim · persistence/reporting |
| **Ensembling + robustluk** (ADR 0021–0029) | Caruana GES + bagging · native klasik forecasting · recursive/seasonal-diff · guardrail/segmentasyon düzeltmeleri |
| **v1.1 model genişlemesi** (ADR 0030–0037) | Nöral tablo/mimari-arama/forecasting · foundation modeller (TabPFN/TabICL/Chronos) · L2 stacking · seçim temizliği · sınıflandırma GES · `explain()` |
| **v1.1 robustluk + akademik tarama** (ADR 0038–0043) | Panel holdout/seçim düzeltmeleri (m3/m5/tourism benchmark bulguları) · EBM/KNN/NGBoost/TimesFM/**TiDE** ekleme · foundation-TS OOF güven guard'ı |
| **v1.1 kapanış** (ADR 0044–0046) | **Split-conformal `predict_interval()`** · **hiyerarşik reconciliation** (MinTrace) · deneysel per-series şampiyon |

## Yol haritası

### v1.1 — kalan küçük kalemler (bu fazda)

| kalem | not |
|---|---|
| Isotonic/linear kalibrasyon | `CalibrateConfig.method` şu an yalnız `off`/`additive_bias`/`multiplicative` |
| GAM (`pygam`) modeli | Yalnız EBM (glassbox GAM ailesi) var; bağımsız `pygam` tabanlı model yok |
| ModernNCA | Akademik taramada (ADR 0040) değerlendirildi, implemente edilmedi |
| String-etiketli sınıflandırma hedefi oto-encode | Çekirdekte yok — yalnız benchmark harness'ı elle kodluyor |
| `predict_interval()`/`explain()` — native forecaster/stack/segmented şampiyonlar | Şu an yalnız tablo + reduction-forecasting (ADR 0044-B/0045-B) |
| MinTrace `mint_shrink` (daha optimal reconciliation) | `OOFArrays`'e zaman damgası (`ds`) eklenmesini gerektiriyor |
| Tam 23-dataset benchmark doğrulaması | v1.1'in genel kapanış ölçümü |

### v1.2 — modalite + veri genişlemesi

- **Gerçek exogenous/kovaryant desteği** (`Dataset.relations` açılması, ADR 0009 rezervi) —
  fiyat/promosyon/tatil gibi dış değişkenler; TiDE'nin ve hiyerarşik reconciliation'ın kovaryant
  desteğini gerçekten kullanabilmek için önkoşul.
- **Metin / görsel / ses modalite** — çekirdek modalite-agnostik tasarım zaten buna göre (ADR 0002).
- Gerçek çok-tablo (`relations`) desteği, hiyerarşik seviyeler arası tahmin erişimi.

### v2 — RAG/agent üst katmanı + kullanıcı arayüzü

- **LLM/agent orkestrasyonu** — `llm/` sağlayıcı soyutlaması (OpenAI/Anthropic/Bedrock/Azure/local)
  üzerine RAG/agent katmanı; `interfaces/agent_tools.py` şeması bugünden hazır.
- **Kullanım arayüzleri — üç seviye:**
  1. **Shell/CLI** (bugün var) — `autoragml run --data ... --target ...`.
  2. **Kolay-kullanım katmanı** — kullanıcı yalnızca veri path'i verip **varsayılan akışla**
     çalıştırabilir; detaylandırmak isterse parametreleri tek tek açabilir (preset → override
     zinciri zaten bu modeli destekliyor, ADR 0016).
  3. **UI** — yukarıdaki ikisinin görsel karşılığı; tüm parametreler arayüzden de değiştirilebilir.
     Tasarım + implementasyon **RAG/LLM fazında (v2)** ele alınacak — bu madde şimdilik yalnız
     **bilgi/niyet notu**, sonradan çok daha fazla ayrıntı eklenecek.

## Kurulum

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,timeseries]"
```

**Extra'lar** (hepsi opsiyonel — çekirdek sıfır ağır bağımlılıkla çalışır):

| extra | ne getirir |
|---|---|
| `timeseries` | Klasik forecasting ailesi (`statsforecast`) |
| `hierarchical` | Hiyerarşik reconciliation (`hierarchicalforecast`, ADR 0045) |
| `xgboost` / `catboost` / `interpret` / `ngboost` | Ek tablo modelleri (varsayılan katalogda opt-in olanlar dahil) |
| `neural` | RealMLP/TabM (`pytabkit` + `torch`, ADR 0030) |
| `neural-nas` | Nöral mimari arama (`pytorch-tabular`, ADR 0031) |
| `neural-ts` | NHITS/PatchTST/TFT/NBEATSx/iTransformer/TSMixer/TiDE (`neuralforecast`, ADR 0032/0043) |
| `foundation` | TabPFN — lisans/token gerekir, **varsayılan katalogda kapalı** |
| `foundation-tabicl` | TabICL — auth'suz in-context tablo (ADR 0040) |
| `foundation-ts` / `foundation-ts-timesfm` | Chronos-Bolt / TimesFM 2.5 zero-shot forecasting |
| `hpo` / `tracking` / `report` / `explain` | Optuna / MLflow / matplotlib grafikleri / SHAP |
| `llm-*` | v2 RAG/agent katmanı için sağlayıcı istemcileri (çekirdek bağımsız) |

Ağır extra'lar (nöral/foundation) ayrı bir venv'de kurulmayı önerir; hepsi tek venv'de de
çalışır ama torch sürüm uyumu kırılgan olabilir (bkz. ADR 0030/0032 notları).

**macOS:** LightGBM için OpenMP runtime gerekir — `brew install libomp`
(kurulu değilse `lightgbm` sessizce aday havuzundan düşer, log'da uyarı çıkar).

## Kullanım

```python
from autoragml import AutoRagML

# Tablo
model = AutoRagML(hpo_level="light")
result = model.fit(df, target="sales")

# Forecasting — panel (long format: her satır bir seri × zaman noktası)
model = AutoRagML()
result = model.fit(df, target="y", time_col="ds", group_col="unique_id")

# Hiyerarşik forecasting (ADR 0045) — group_col en-alt seviye, hierarchy_cols üstü
model = AutoRagML(hierarchy_cols=["state", "zone"])
result = model.fit(df, target="y", time_col="ds", group_col="region")

result.leaderboard()                              # tüm adayların sıralı skor tablosu
result.predict(new_df)
result.predict_interval(new_df, coverage=0.9)      # yalnız tablo + reduction-forecasting şampiyonları
result.explain(new_df)

# Serving (diskten yükle)
champion = AutoRagML.load("outputs/.../models/champion.joblib")
champion.predict(new_df)
```

CLI: `autoragml run --data df.csv --target sales [--time-col ds --group-col id --preset ... ]`

## Benchmark

`scripts/benchmarks/` — OpenML/sklearn tablo (16 set) + M3/M4/M5/Tourism/ETT forecasting (7 set),
naive baseline karşılaştırmalı. Tam sonuçlar: [`scripts/benchmarks/RESULTS.md`](scripts/benchmarks/RESULTS.md).

```bash
python -m scripts.benchmarks.run --profile dev --hpo none    # ~12 dataset, hızlı geliştirme sinyali
python -m scripts.benchmarks.run --hpo none                  # tam 23 dataset (~saatler)
```

Öne çıkanlar: tablo setlerinde naive'e göre **+61% ila +862%** iyileşme (TabICL foundation
modeli birçok sette GBM'i geçiyor); forecasting'de segmentasyon + joint-ensemble + foundation-TS
ile m5 kesikli talepte **+22–24%**, tourism'de **+15–19%** (bkz. RESULTS.md için tam tablo ve
metodoloji).

## Yapı

```
src/autoragml/
  contracts/       katmanlar arası tipli omurga (ÖNCE bu dondurulur)
  config/          şema + YAML + katmanlı merge + preset
  io/              yükleyiciler + lazy Dataset (strict fingerprint)
  analyzers/       veriyi anlama -> DataProfile + TaskSpec (intermittency/mevsimsellik dahil)
  dynamics/        veriye-özel strateji -> AdaptivePlan (segmentasyon, hiyerarşi, model ipuçları)
  preprocessors/   fit/transform, leakage-safe
  models/          katalog (YAML) + registry + foundation/neural kapıları
  fine_tuners/     HPO (random/Optuna/mimari-arama) + early stopping + bütçe
  validators/      split stratejileri + nested-CV koşucu + leakage testi
  ensembling/      Caruana GES + bagging + L2 stacking
  scoring/         metrikler + guardrail + 1-SE seçim
  engines/         orkestrasyon (tabular/timeseries: reduction, klasik, nöral, foundation,
                   segmented, hierarchical, joint-ensemble, stack) + runners/
  postprocessors/  calibrate/clip/split-conformal/round/business-rule
  explain/         SHAP / permutation öznitelik atıfı
  persistence/     ModelBundle (+ nöral/foundation sidecar) + RunManifest
  reporters/       EDA / model card / grafik / karşılaştırma
  tracking/        opsiyonel (JSONL varsayılan, MLflow opsiyonel)
  llm/             LLMProvider soyutlaması (v2)
  interfaces/      api.py · cli.py · agent_tools.py
scripts/benchmarks/  OpenML + M3/M4/M5/Tourism/ETT karşılaştırma harness'ı
docs/architecture/   genel bakış + katman taslakları + karar kayıtları (decisions/)
```

## Geliştirme

```bash
ruff check . && mypy && pytest    # tam kalite kapısı — ağır modeller (EBM/NGBoost/SVR/AutoTBATS/
                                   # TabICL) varsayılan olarak test'te kapalı (conftest.py, hızlı suite)
pytest -m full_catalog            # yalnız tam-katalog testleri (ağır modeller dahil)
```

## Lisans

[Apache License 2.0](LICENSE) — ticari kullanım dahil serbestçe kullanılabilir, değiştirilebilir
ve dağıtılabilir; koşul: telif/lisans bildirimleri (ve varsa [`NOTICE`](NOTICE) içeriği) korunur.
Üçüncü parti atıflar: [`NOTICE`](NOTICE).
