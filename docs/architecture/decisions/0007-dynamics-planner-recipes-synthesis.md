# ADR 0007 — dynamics: deterministik planner + custom recipe plug-point + v2 synthesis

**Durum:** Kabul · 2026-09-01

## Bağlam

`dynamics` katmanının kapsamı belirsizdi. İki farklı ihtiyaç aynı isim altında toplanmıştı:

1. **Veriye-özel ön işleme** — büyük bir veri setinde modele girmeden önce yapılması
   gereken, o veri setine özgü dönüşümler. Pratikte bunların çoğu genel bir desenin
   örneğidir (yüksek kardinalite → target encode; ağır kuyruk → log1p; tarih → takvim
   özellikleri; sparse cari-ürün matrisi → per-group champion).
2. **LLM'in kod üretebileceği alan** — kataloğa uymayan gerçekten biricik dönüşümler
   (domain kuralı, beklenmedik alan parse'ı, özel türetme) için üretilen kod ve bunun
   modele nasıl bağlanacağı.

## Karar

`dynamics` üç parçaya ayrılır:

### `dynamics/planner.py` — v1, deterministik
`DataProfile` + `TaskSpec` + `RunConfig` → `AdaptivePlan`. Sabit bir **op kataloğundan**
kural/eşikle seçim yapar. Kod üretmez. LLM yok. Fit yok — karar üretir.

### `dynamics/recipes/` — v1, custom transform plug-point
Kataloğa uymayan dönüşümler için kayıt yeri. Bir recipe:
- `preprocessors` ile **aynı arayüze uyar**: `fit(train_df) -> self`, `transform(X) -> X'`
- serialize edilebilir (joblib), `ModelBundle`'a girer
- `registry` üzerinden isimle çözülür (`parse_recipe:"acme_sales_parser"`)
- v1'de insan yazıp buraya koyar

### `dynamics/synthesis.py` — v2, LLM üretimi
LLM bir recipe sınıfı üretir → `engines/runners` (Subprocess/Container, ADR 0006)
içinde çalıştırılıp doğrulanır → `recipes/`'e kaydedilir. Çıktı **her zaman arayüze
uyan bir transform'dur**; ayrı bir entegrasyon yolu yoktur.

## Modele entegrasyon

Custom kod **modelin içine girmez.** Pipeline adımı olur:

```
[dataset-özel recipe'ler] → [standart preprocessors] → [estimator] → [postprocessors]
```

- `validators` recipe'i **yalnız train fold'unda** `fit` eder → leakage yapısal olarak imkânsız.
- Test'e sızan kod (LLM üretse bile) `validators/leakage_checks`'e takılır.
- Çıktı `scoring/guardrails` denetiminden geçer (negatif/aşırı tahmin, metrik tavanı).
- Tüm zincir tek `ModelBundle`'a serialize edilir → serving'de aynen çalışır.

## Sonuç

- v1: `planner` + `recipes` (manuel). `synthesis` yok.
- `AdaptivePlan` sözleşmesi hem katalog op'larını hem isimli recipe'leri referanslar.
- LLM kod üretimi ADR 0003 (deterministik çekirdek / RAG v2) ile tutarlı: ayrı yetenek,
  aynı arayüz, aynı doğrulama/guardrail süzgeci.
- `preprocessors` ile `dynamics/recipes` arasındaki tek fark **niyet**: recipes veri
  setine özgü ve opsiyonel; preprocessors genel ve katalog. Arayüz aynı.
