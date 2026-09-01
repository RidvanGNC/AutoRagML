# AutoRagML benchmark harness

Gerçek verisetlerinde uçtan uca `AutoRagML().fit()` koşumu + **harici test setinde**
naive baseline ile karşılaştırma. Süre önemli değil; ölçü "sağlıklı başarı".

```bash
python -m scripts.benchmarks.run --list          # kayıtlı setler
python -m scripts.benchmarks.run                  # ilk dalga (tümü), hpo=light
python -m scripts.benchmarks.run --only adult --hpo none
```

Çıktı: `scripts/benchmarks/_runs/<timestamp>/` → `summary.json` + `summary.md`
(+ her dataset'in kendi `outputs/` run dizini).

## İlk dalga (OpenML / sklearn — otomatik indirme, ~15 MB, `~/scikit_learn_data/`)

| dataset | görev | stres |
|---|---|---|
| california_housing | regresyon | temiz sayısal |
| bike_sharing | regresyon | sayısal + kategorik |
| adult | ikili | kategorik + eksik + dengesiz |
| credit_g | ikili | küçük veri |
| bank_marketing | ikili | ağır dengesiz |
| dry_bean | çok-sınıf | 7 sınıf |

## Bilinen v1 sınırı

Sınıflandırma hedefi **string** ise benchmark onu `pd.Categorical(...).codes` ile kodlar
(`target_encoded=true` işaretlenir). AutoRagML `split_xy` şu an hedefi sayısala zorluyor
→ string-etiket sınıflandırması için otomatik label-encoding **v1.1**.

## Sonraki dalgalar

- 2. dalga: TS panel (Nixtla `car_parts`, tourism) + M5 (Kaggle — elle indirilip yol verilir).
- 3. dalga: yüksek boyut / seyrek, ordinal, quantile.
