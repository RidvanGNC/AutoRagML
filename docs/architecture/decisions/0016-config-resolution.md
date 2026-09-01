# ADR 0016 — config çözümleme (katmanlı merge + provenance + Settings)

**Durum:** Kabul · 2026-09-01
**Motto:** zaman değil sağlıklı başarı; detay kaçmaz.

## Katmanlı merge

`RunConfig` sırayla şu katmanlardan kurulur (sonra gelen kazanır):

1. **defaults** — `RunConfig` pydantic alan varsayılanları (ayrı YAML yok)
2. **preset** — `preset=` ile seçilen yerleşik reçete (varsa `extends:` zinciri önce)
3. **file** — `config_file=` kullanıcı YAML'ı
4. **overrides** — runtime kwargs / CLI (`overrides={...}`)

### Merge kuralları (detay — ADR gereği net)
- `dict + dict` → **özyinelemeli** birleşme (`split_policy` gibi kısmi objeler doğal çalışır)
- `scalar | list + herhangi` → **değiştir** (append yok)
- YAML'da açık `key: null` → sonraki katman `None` ile **ezer**; anahtarın **yokluğu** = miras
- Bilinmeyen anahtar → merge sonrası `RunConfig.model_validate` **ValidationError** verir
  (`extra="forbid"`)
- Meta anahtarlar merge öncesi soyulur: `preset`, `extends`, `description`

### `extends`
Preset başka preset'i genişletebilir (`demandsensing` → `timeseries_rolling`).
Zincir kökten yaprağa çözülür; **döngü tespiti** var.

## Provenance (izlenebilirlik)

`resolve_run_config` bir `ConfigResolution` döndürür:
- `config: RunConfig` — doğrulanmış nihai nesne
- `provenance: dict[str, str]` — yaprak alan yolu → kaynak katman
  (`"budget.max_trials_per_model" -> "preset:timeseries_rolling"`, ayarlanmamış → `"default"`)
- `layers: list[str]` — uygulanan katmanlar, sırayla

`RunManifest.config_snapshot` bu provenance'ı da taşır — "n_folds neden 6?" cevaplanabilir.

## Preset konumu

Yerleşik presetler **pakete gömülü**: `src/autoragml/config/presets/*.yaml`,
`importlib.resources` ile okunur (wheel'de de çalışır).
Repo kökündeki `configs/` = **kullanıcı proje config'leri** için örnek/alan; pakete girmez.
(Aynı ilke model kataloğu için de: `src/autoragml/models/catalog/` — ADR 0012 uygulaması.)

## Settings (sırlar)

`config/settings.py` — `Settings(BaseSettings)` (pydantic-settings), `.env` + ortam
değişkenlerinden okur. `SecretStr` alanları. **Asla serialize edilmez.**
`resolve_secret(env_name: str) -> SecretStr | None` — `RunConfig.*_env` adlarını çözer.
`pydantic-settings` çekirdek bağımlılığa eklenir (hafif, yalnız pydantic'e bağlı).

## Sonuç
- `config/`: `__init__.py` (`resolve_run_config`), `merge.py`, `loaders.py`, `presets.py`,
  `settings.py`
- `contracts`: `ConfigResolution` eklenir (`config`, `provenance`, `layers`)
- presetler `src/autoragml/config/presets/` altına taşınır; repo-kök `configs/` örnek olur
