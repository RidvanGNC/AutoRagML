# ADR 0009 — Dataset + io sözleşmesi

**Durum:** Kabul · 2026-09-01

**Motto (tüm adımlar):** Zamanın miktarı önemsiz, sağlıklı başarı kesin ölçüdür.
Detaylar kaçırılamaz.

## Kararlar

### 1. fingerprint — STRICT (varsayılan ve zorunlu)
Dataset kimliği tüm hücrelerin hash'idir. Structural/örneklem hash **yeterli değil** —
"aralarda satır kaçamaz".
- Kanonik form: `group_col`+`time_col` varsa ona göre sırala, yoksa satır sırasını koru;
  şema kısmı için kolon adları alfabetik normalize.
- `fingerprint = SHA256( schema_repr ‖ hash_pandas_object(canonical_frame) )`
- Lazy/chunked büyük veride bile **tek streaming geçişte** hesaplanır (uzun sürebilir, sorun değil).
- `RunManifest` içine `fingerprint` + `fingerprint_spec` (nasıl hesaplandığı) yazılır.
- Ek alan `fingerprint_fast` (structural) yalnız hızlı drift sinyali için — kimlik değil.

### 2. Kanonik zaman serisi formatı — LONG
`group_col, time_col, target [, exogenous feature'lar]`.
- **Wide** kabul edilir → otomatik `melt` → `layout = "wide_converted"`, işlem loglanır.
- Wide'da yalnız hedef geçmişi vardır: **exogenous feature yok**, model havuzu
  baseline + univariate reduction ile **sınırlıdır** (`analyzers` `layout`'a bakıp
  aday modelleri kısıtlar). Kullanıcı bilgilendirilir: "wide girdi → basit modelleme".
- Feature içeren veri **long zorunlu**.
- `single_series` (grup kolonu yok, tek `date,value`) da `layout` değeri.

### 3. Çoklu tablo — yapı ne bekliyorsa o
v1 kapsamı (tablo + zaman serisi, deterministik çekirdek) **tek analitik tablo** bekler:
kullanıcı join/ETL'i upstream'de yapar (SQL/pandas/dbt).
- `Dataset.relations` alanı **REZERVE** — v1'de `None`. Çok-tablo + leakage-safe as-of
  join geldiğinde (ayrı modül `io/relations` veya bir `dynamics/recipe`) sözleşme
  kırılmadan doldurulur. Detay şimdi düşünüldü, v1'e taşınmadı.

### 4. Girdi türleri (v1)
DataFrame · `.csv` · `.parquet` · csv/parquet klasörü (parçalı). **DB opsiyonel**
(`[db]` extra, SQLAlchemy adaptörü).

### 5. Materialization — otomatik
`io/` kaynak boyutunu yoklar (dosya byte / DB `count(*)` / `df.memory_usage`).
`RunConfig.io.eager_max_bytes` (default `min(1 GiB, RAM %25)`) altında → pandas (eager);
üstünde → pyarrow dataset / chunk iterator (lazy). Override edilebilir.
- Not: lazy modda `Dataset.shape.n_rows` yine **tam sayımdır** (strict prensibi — tahmin yok).

## Sonuç
- `contracts/dataset.py` alan tablosu `01_contracts.md`'de dondurulur.
- `io/` sorumlulukları `02_layers.md`'de güncellenir.
- Motto `00_overview.md` İlkeler'e eklenir.
