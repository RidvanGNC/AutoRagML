# ADR 0005 — LLM sağlayıcı soyutlaması (kullanıcı seçer)

**Durum:** Kabul · 2026-09-01

## Bağlam
"Bulut zorunluluğu yok" = backend takılabilir olmalı. Kullanıcı Bedrock, Azure
OpenAI, OpenAI-uyumlu, Anthropic veya local (Ollama/llama.cpp/vLLM) seçebilmeli.

## Karar
`llm/base.py` içinde `LLMProvider` protokolü: `complete`, `stream`, `embed`.
Her sağlayıcı `llm/providers/` altında ayrı dosya + ayrı extra (`llm-openai`,
`llm-anthropic`, `llm-bedrock`, `llm-azure`, `llm-local`). Varsayılan `null`.
Seçim config'te: `llm: {provider, model, endpoint, api_key_env}`. Sırlar env'den.

## Sonuç
- Çekirdek bağımlılıklarında hiçbir LLM SDK'sı yok.
- `embed()` de aynı soyutlamada (RAG bilgi tabanı embedding'i kullanıcı seçimine tabi).
- v1'de yalnız iskele; gerçek kullanım v2.
