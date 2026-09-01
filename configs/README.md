# configs/

**Kullanıcı proje config'leri için.** Bu dizin pakete girmez.

- Yerleşik presetler artık pakete gömülü: `src/autoragml/config/presets/*.yaml`
  (`autoragml.config.list_presets()` ile listelenir).
- Kendi koşum config'inizi buraya koyup `resolve_run_config(config_file="configs/benim.yaml")`
  veya CLI `--config` ile kullanın. Alan adları `RunConfig` ile birebir (bkz. ADR 0016 / 01_contracts.md).
- `model_catalog/` — kullanıcı model kataloğu override'ları (yerleşik katalog: ADR 0012, models katmanında).
