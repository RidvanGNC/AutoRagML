# ADR 0012 — Model kataloğu (YAML) + registry

**Durum:** Kabul · 2026-09-01

## Karar

Model kataloğu **YAML** ile tanımlanır (DemandSensing `machine_learning:` deseni).
Okuması/yönetmesi kolay, standarda uygun. `class_path` gerçek estimator'ı verir.

### Katalog entry şeması
```yaml
lightgbm:
  name: LightGBM
  class_path:                    # str VEYA task-ailesi -> path map
    regression: lightgbm.LGBMRegressor
    classification: lightgbm.LGBMClassifier
  family: gbdt
  modalities: [tabular, timeseries]
  tasks: [regression, binary_classification, multiclass_classification,
          forecasting, quantile_regression]
  predict_kind: [point, proba, quantile]
  supports_early_stopping: true
  requires: [lightgbm]           # kurulu değilse entry atlanır (tek uyarı)
  default_params: {...}
  search_space:                  # HPO uzayı (ADR 0013)
    learning_rate: {type: loguniform, low: 0.01, high: 0.3}
    num_leaves:    {type: int, low: 15, high: 255}
  fidelity: n_estimators         # multi-fidelity ekseni
  enabled: true
```

### Yerleşim
- `configs/model_catalog/*.yaml` — paketle gelen yerleşik katalog (tabular, timeseries,
  intermittent, baselines)
- Kullanıcı kendi YAML'ını **üstüne merge** eder (RunConfig katmanlaması gibi):
  `enabled: false`, param/`search_space` değiştir, **yeni entry ekle** (importable
  `class_path` yeter)

### registry/
Katalog YAML'larını okur → `class_path` importable + `requires` kurulu mu doğrular →
`TaskSpec.task` ile uyumlu entry'leri `Candidate` nesnesine çevirir. Eksik bağımlılık →
entry sessizce atlanır, tek seferlik WARNING (DemandSensing deseni).
Entry-points = **ikincil** yol (üçüncü parti plugin paketi); YAML kanonik.

### v1 kataloğu
sklearn linear/ridge/lasso/elastic_net/RF/ET/GBM · lightgbm · xgboost(opsiyonel) ·
catboost(opsiyonel) · naive/seasonal_naive/trend_naive/stl_auto baseline ·
statsforecast AutoARIMA/AutoETS/MSTL/AutoTheta(opsiyonel `[timeseries]`) ·
Croston/TSB/SBA (intermittent, aday havuzu genişletme — ADR 0010)

## Sonuç
- `contracts.Candidate` alanları `01_contracts.md`'de dondurulur.
- `models/` YAML okur; estimator üretimi + wrap (imputer/scaler) burada.
- Vendor: AutoGluon problem-type çıkarımı yardımcıları (ADR 0004) `analyzers` destekler.
