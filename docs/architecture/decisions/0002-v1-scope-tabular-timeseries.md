# ADR 0002 — v1 kapsamı: tablo + zaman serisi

**Durum:** Kabul · 2026-09-01

## Bağlam
İlham kaynağı hat (demand sensing) tablo + haftalık zaman serisi. Metin/görsel/ses
torch ekosistemi = ağır bağımlılık + platform hassasiyeti + daha geniş sözleşme riski.

## Karar
v1 yalnız tablo + zaman serisi. Sözleşmeler bu iki modalitede stabilize edilir,
sonra genişletilir (v1.1 metin, v1.2 görsel/ses).

## Sonuç
- `contracts` erken ve gerçek kullanımla stabilize olur.
- `engines/tabular` + `engines/timeseries` v1 hedefi.
- `TaskSpec.modality` alanı baştan genişlemeye açık tasarlanır ama v1'de 2 değer.

## Güncelleme

`requires-python >= 3.12` (PEP 695, modern tip stub'ları). CI matrisi 3.12/3.13.
