# ADR 0006 — EngineRunner tier'ları (izolasyon eskalasyonu)

**Durum:** Kabul · 2026-09-01

## Bağlam
Kütüphane arttıkça sürüm çakışması riski (özellikle torch + transformers + AG +
tensorflow aynı env'de). "Her şeyi mini-container ağına koy" fikri dep hell'i çözer
ama in-process çağrıyı network RPC'ye çevirir, "kolay pip / tool" şartını kırar,
Windows'ta Docker/WSL2 sürtünmesi getirir.

## Karar
`engines/runners/` altında katmanlı `EngineRunner` protokolü: `run(engine_spec,
data_ref) -> result_ref`. Sınır formatı **1. günden Arrow/Parquet + manifest**.

| Tier | Mekanizma | Varsayılan? |
|---|---|---|
| 0 | Tek env, kilitli pinning | ✅ (v1 kapsamında çakışma yok) |
| 1 | Opsiyonel extras + import guard | ✅ |
| 2 | Subprocess, ayrı venv, container YOK | iskele v1, kullanım v1.1 |
| 3 | Container / remote engine servisi | v2+, opt-in infra |

## Sonuç
- Orchestrator hangi runner olduğunu bilmez.
- Container-mesh dayatma değil, en üst eskalasyon tier'ı.
- Varsayılan her zaman in-process.
