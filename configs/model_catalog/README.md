# configs/model_catalog/

**Kullanıcı model kataloğu override'ları için.** Bu dizin pakete girmez.

- Yerleşik katalog pakete gömülü: `src/autoragml/models/catalog/*.yaml`.
- Kendi YAML'ınızı buraya koyup `RunConfig.model_catalog_override: [configs/model_catalog/benim.yaml]`
  ile devreye alın. Entry key bazında deep-merge edilir: `enabled: false` ile kapatın,
  `default_params`/`search_space` değiştirin, yeni entry ekleyin (importable `class_path` yeter).
- Şema: ADR 0012 / `src/autoragml/models/catalog/tabular.yaml`.
