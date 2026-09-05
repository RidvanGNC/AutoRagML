# ADR 0046 — gerçek per-series (per-group) şampiyon refit — deneysel

**Durum:** Kabul (deneysel özellik, varsayılan DEĞİL) · 2026-09-05

Kaynak: kullanıcı kararı "araştırma segmented'in genelde daha iyi olacağını gösteriyor ama yine
de dene". Ayrıca bir isim netliği sorunu: mevcut `structure="per_group_champion"` alanı aslında
ADR 0028'in **segment**-seviyesi kümelemesini çalıştırıyor (SBC/intermittency sınıfına göre ≤4
küme), seri-başı DEĞİL — "per group" adı var ama davranışı "per segment".

## Karar

**Yeni `structure` değeri: `"per_series_champion"`** — mevcut `"per_group_champion"`'dan ayrı
(o segment-seviyesinde kalır, geriye dönük davranış değişmedi). Seçilince `_resolve_segments()`
kümeleme yapmaz, **her seri kendi tek-üyeli segmenti** olur → `run_segmented` mekanizması
(ADR 0028) **aynen** kullanılır — yeni motor kodu yok.

- **Yalnız açık bildirim** (`dynamics.structure="per_series_champion"`) — `"auto"` hiçbir zaman
  buna çözülmez.
- **Maliyet tavanı yok** (kullanıcı kararı — `neural_search` gibi diğer pahalı opt-in modlarla
  tutarlı: sert limit yerine yalnız bilgilendirici log, "kullanıcı süre maliyetini kabul eder").
- **Gerçek bug bulundu+düzeltildi (implementasyon sırasında):** `run_segmented`'ın `n_series < 2`
  guard'ı ADR 0028'in kümeleme senaryosu için doğruydu ("tek-serilik = yetersiz kümeleme") ama
  per-series segmentler TASARIM GEREĞİ tek seri — bu guard onları hep atlayıp "hiçbir segment
  doğrulanamadı" hatası veriyordu. Fix: `SegmentSpec.source == "per_series"` iken eşik 1'e düşer.

## Ampirik sonuç (tourism_large alt-kümesi, 5 seri, `--hpo none`, klasik/nöral/foundation kapalı)

| yaklaşım | test wMAPE | vs naive (18.34) | süre | şampiyon |
|---|---|---|---|---|
| `auto` (mevcut varsayılan → pooled, panel yeterince kesikli değil) | **17.00** | **+7.3%** ✅ | 41s | GES ensemble (`extra_trees`) |
| `per_series_champion` | **19.09** | **-4.1%** ❌ | 227s (5.5×) | 5 ayrı şampiyon: `mlp`, `weighted_ensemble`×3, **`dummy_median`** |

**Per-series naive'i bile geçemedi, hem daha yavaş hem daha kötü.** Bir seride (`DBCOth`)
izole validasyonda hiçbir gerçek model naive'i geçemedi — şampiyon baseline'a (`dummy_median`)
düştü. Cross-series öğrenme olmadan (her seri kendi başına), az veri + gürültüde model seçimi
güvenilmez hale geliyor — tam olarak [Local vs Global Models for Intermittent Time Series
Forecasting](https://arxiv.org/abs/2601.14031) (ADR 0043 kaynağı) makalesinin öngördüğü sonuç.

**Sonuç:** `per_series_champion`'ın varsayılan olmaması (yalnız açık opt-in) kararı bu ölçümle
doğrulandı. Özellik **var** (kullanıcı özellikle test etmek isteyen senaryolar için — ör. gerçekten
heterojen, birbirinden bağımsız serilerden oluşan az-sayıda-seri panelleri) ama **önerilmiyor**;
varsayılan yol hâlâ `auto`/`per_group_champion` (ADR 0028 segmentasyonu).

## Uygulama

- `RunConfig`/`DynamicsConfig`/`AdaptivePlan.structure` += `"per_series_champion"` literal.
- `dynamics/planner._resolve_segments_per_series`: kümeleme yok, `SegmentSpec(name=seri_id,
  group_ids=[seri_id], source="per_series")` — her seri için.
- `engines/segmented.py`: `n_series < 2` guard'ı `source == "per_series"` iken `< 1`'e düşer.
- `engines/core.py`: tek-anlamlı-segment mesajı her iki yapı değerini de kapsar.

## Doğrulama

- `tests/unit/dynamics/test_planner.py` (+2): her seri kendi segmenti, `"auto"` asla
  `per_series_champion`'a çözülmez.
- `tests/unit/engines/test_engines_e2e.py` (+1): 3-serili panel → `FittedSegmentedPipeline`,
  3 segment, `source == "per_series"`, NaN'sız serving.
- Manuel e2e: 3 seri gerçek uçtan uca (`extra_trees`/`ngboost`/`ngboost` — genuinely farklı
  seri-başı şampiyonlar, mekanizma doğru çalışıyor).
- Ampirik karşılaştırma (yukarıda) — tourism 5 seri, auto +7.3% vs per_series -4.1%.
- ruff + mypy(147) yeşil.

## Kapsam dışı

- Varsayılan olarak önerilmiyor — `auto`/segmented (ADR 0028) yol haritasında kalmaya devam
  ediyor. Bu ADR bir **kapasite** ekliyor, bir **yön değişikliği** değil.
- Maliyet tavanı / erken-durdurma — kullanıcı kararıyla eklenmedi.
