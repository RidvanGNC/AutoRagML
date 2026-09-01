# Model kataloğu (ADR 0012)

Paketle gelen yerleşik katalog. Her `*.yaml` bir grup model tanımı içerir.
Kullanıcı kendi YAML'ını `RunConfig.model_catalog_override` ile üstüne merge eder:
- `enabled: false` ile devre dışı bırak
- `default_params` / `search_space` değiştir
- yeni entry ekle (importable `class_path` yeter)

Entry şeması için ADR 0012'ye bakın. Bu dosyalar contract'lar dondurulduktan sonra doldurulacak.
