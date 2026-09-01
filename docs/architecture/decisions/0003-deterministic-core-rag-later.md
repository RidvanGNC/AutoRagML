# ADR 0003 — v1 tamamen deterministik; RAG/agent ayrı üst katman (v2)

**Durum:** Kabul · 2026-09-01

## Bağlam
Agentic AutoML (MLZero, AIDE, MLE-STAR) güçlü ama: LLM API bağımlılığı, maliyet,
tekrarüretilemezlik, sandbox/container yükü, kayıtlı kütüphane dışına çıkamama.

## Karar
v1 çekirdeği LLM bilmez. Bileşen orkestrasyonu deterministik. RAG bilgi tabanı +
agent döngüsü v2'de ayrı katman/paket olarak `interfaces/agent_tools.py` ve `llm/`
üstünde oturur, çekirdeği tool olarak çağırır.

## Sonuç
- `llm/` v1'de yalnız `base.py` + `registry.py` + `null` provider iskeleti.
- `interfaces/agent_tools.py` çekirdek fonksiyonların JSON-schema'sını üretir (v2 tüketir).
- Çekirdek testleri LLM'siz, hızlı, deterministik.
