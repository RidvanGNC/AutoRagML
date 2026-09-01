# ADR 0008 — RunConfig varsayılanları ve çıkarım politikası

**Durum:** Kabul · 2026-09-01

## Bağlam

`RunConfig` sözleşmesi dondurulmadan önce dört politika netleşmeliydi: bütçe modeli,
split seçimi, hedef/zaman/grup kolonu çıkarımı, sırların yeri.

## Kararlar

### 1. Bütçe — çok eksenli, cömert varsayılan, sessiz kesme YOK
Gerçek bir tek `fit` büyük veride 40+ dakika sürebilir (ör. XGBoost). Varsayılan
bütçe bunu **öldürmemeli**.
- `budget.total_max_seconds = null` (global tavan yok)
- `budget.per_model_max_seconds = null` (trial sayısı yönetir)
- `budget.max_trials_per_model = 15`, `min_trials_per_model = 3`
- `budget.per_fold_timeout_seconds = null` (fold iptali yok)
- `budget.runtime_projection_warn_seconds = 7200` — ilk fold bitince `fine_tuners`
  toplam süreyi ekstrapole eder; bu eşiği aşarsa **uyarır ama devam eder**
  (otomatik küçültme = auto mode, v2).
- Kullanıcı yalnız kısıtlamak istediği ekseni verir.

### 2. Split — katmanlı (analyzers taban + kullanıcı override); auto mode v2
- `RunConfig.split_policy` opsiyonel ve **kısmi**: verilen alan kazanır, verilmeyen
  `analyzers.SplitRecommendation`'dan gelir.
- Leakage-tehlikeli override (`kind: kfold` ama zaman serisi) → `validators/leakage_checks`
  reddeder.
- **v2:** `RunConfig.autopilot = true` → `analyzers` split dahil her şeye karar verir.

### 3. Hedef/zaman/grup kolonu — v1'de AÇIK (çıkarım yok)
- `target` **zorunlu**.
- `time_col` forecasting task'inde **zorunlu**.
- `group_col` opsiyonel: yoksa pooled, varsa per-group champion mümkün.
- v1'de otomatik tahmin yok — yanlış tahmin sessizce yanlış modele gider, riskli.
- **v2:** LLM katmanı geldiğinde bu kolonlar kullanıcıya **soru** olarak sorulur
  (`autopilot` / interaktif onay).

### 4. Sırlar — her zaman `.env`, koda asla sızmaz
- `RunConfig` yalnız **env-var adı** taşır: `api_key_env`, `endpoint_env`, `uri_env`.
- Değerleri `Settings` (pydantic-settings) runtime'da `.env`'den okur; `Settings`
  asla serialize edilmez.
- `RunManifest` serialize'ında `SecretStr` alanları maskelenir / atlanır.
- `.env` `.gitignore`'da (zaten var).

## Sonuç

- `AutoRagML().fit(df, target="y")` → cömert varsayılanlarla çalışır, hiçbir şeyi
  sessizce kesmez, uzun sürerse uyarır.
- `contracts/run_config.py` alan tablosu `01_contracts.md`'de dondurulur.
- `autopilot` alanı v1'de tanımlı ama `false` sabit; v2 auto mode'un giriş noktası.
