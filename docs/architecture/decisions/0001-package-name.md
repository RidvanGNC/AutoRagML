# ADR 0001 — Paket adı: AutoRagML

**Durum:** Kabul · 2026-09-01

## Bağlam
Paket adı importable olmalı, işi açıklamalı, PyPI'da ayırt edici olmalı.

## Karar
Paket: `autoragml`. Repo/klasör: `AutoRagML`. Dağıtım adı: `autoragml`.
Ad, deterministik AutoML çekirdeği + ileride eklenecek RAG/agent katmanını birlikte anlatır.

## Sonuç
- `src/autoragml/` src-layout.
- CLI komutu: `autoragml`.
- v1'de "Rag" kısmı henüz pasif (v2 hedefi) — README'de açıkça belirtilir.
