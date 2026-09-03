# ADR 0040 — akademik tarama model eklemeleri (klasik forecasting + EBM/KNN/SVR/NGBoost)

**Durum:** Kabul · 2026-09-03 (kullanıcı onayı — araştırma sırası 1-3)

Kaynak: `docs/architecture/research/2026-09-model-and-benchmark-landscape.md` (TabArena/AMLB/
GIFT-Eval taraması). Katalog additive genişletme (ADR 0012 deseni) — mimari değişiklik yok.

## Eklenenler

### Forecasting klasik (`timeseries.yaml`, `[timeseries]` extra — zaten var, sıfır maliyet)

| key | sınıf | aile | not |
|---|---|---|---|
| `auto_ces` | `AutoCES` | statistical | complex ES — bir benchmark'ta AutoARIMA'yı (MASE 0.73 < 0.80) geçti |
| `auto_tbats` | `AutoTBATS` | statistical | çoklu mevsim + Box-Cox; per-seri **yavaş** (ETS ~10×) |
| `dynamic_theta` | `DynamicOptimizedTheta` | statistical | M3 winner Theta ailesi, dinamik ağırlık |
| `imapa` | `IMAPA` | intermittent | Intermittent Multiple Aggregation — M5 kesikli talep |
| `adida` | `ADIDA` | intermittent | Aggregate-Disaggregate Intermittent Demand |

`_build_model` `season_length`'i oto-enjekte eder. Hepsi native `StatsForecast` yolundan
(ADR 0023), `classical_ensemble` (EAT, ADR 0024) + `joint_ensemble` (ADR 0035) havuzuna girer.

### Tablo (`tabular.yaml`)

| key | sınıf | aile | extra | not |
|---|---|---|---|---|
| `ebm` | `interpret.glassbox.ExplainableBoosting{Regressor,Classifier}` | **glassbox** | `[interpret]` | cyclic-boosting GAM — XGBoost-rakip **+ tam yorumlanabilir**; ADR 0037 `explain()` sinerjisi |
| `knn` | `sklearn.neighbors.KNeighbors{Regressor,Classifier}` | distance | çekirdek | `scale: true`; küçük veri / yerel yapı |
| `svr` | `sklearn.svm.SVR` | distance | çekirdek | `enabled: false` (O(n²) — kullanıcı açar, <20K) |
| `svc` | `sklearn.svm.SVC` | distance | çekirdek | `enabled: false` (O(n²) + `probability=True` pahalı) |
| `ngboost` | `ngboost.NGB{Regressor,Classifier}` | gbdt | `[ngboost]` | olasılıksal GB — quantile/dağılım; `fidelity: n_estimators` |

- `scoring/selection._FAMILY_COMPLEXITY` += `glassbox: 2` (EBM eşitlikte GBDT'ye tercih — yorumlanabilir).
- `pyproject` `interpret` / `ngboost` extra. `manifest._ENV_PACKAGES` += `interpret-core` / `ngboost`.
- `interpret-core` + `ngboost` + `pygam` kuruldu — **`ngboost`/`pygam` scipy 1.18 → 1.16 düşürdü**
  (gevşek pin; core suite yeşil kaldı, sorun yok).

## Ertelenenler / kapsam dışı

- **GAM** (`pygam`) — sklearn API'si tam uyumlu değil (`get_params` yok, `terms` spec, NaN yok);
  ayrıca EBM zaten bir GAM (cam-kutu). Gerekirse wrapper ile sonra.
- **TabICL / ModernNCA / xRFM / Mitra** (TabArena roster) → araştırma sırası 6 (opsiyonel).
- **TimesFM / Moirai** → araştırma sırası 5 (ayrı iş).
- SVR/SVC katalogda ama `enabled: false` — büyük veride O(n²) tehlikeli.

## Doğrulama

- 5 klasik + `knn`/`ebm`/`ngboost` çözülüyor + fit/predict smoke ✅.
- `test_registry.test_adr0040_academic_additions_resolve`.
- Fayda ölçümü → benchmark (araştırma sırası 4'ten sonra).
