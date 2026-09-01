# ADR 0019 — reporters + tracking: bağımlılıksız çıktı & gözlem sink'leri

**Durum:** Kabul · 2026-09-01

Kaynak: Mitchell et al. "Model Cards for Model Reporting" (kanonik bölümler);
2025 pratiği — kartın çoğunu pipeline metadata'sından **otomatik üret**, yargı gerektiren
bölümleri (intended use / limitations / ethics) şablon placeholder bırak. Tracking
soyutlaması: NVIDIA FLARE `LogWriter` deseni (`log_params`/`log_metrics`/`log_artifact`,
çoklu writer tek arayüz).

## Neden tek ADR

`reporters` ve `tracking` **ortak sürücüyü** paylaşır: (1) orchestrator'ın çağırdığı
**yan-etki sink'leri**, (2) bitmiş contract nesnelerini salt-okunur tüketir, (3)
opsiyonel — kapalıyken/eksik bağımlılıkta **no-op'a düşer**, (4) **çekirdekte ağ yok**,
(5) `persistence`'ın açtığı run dizinine yazar. Sıkı bağlı → gruplanır (ADR granülerlik
kararı, [[0018-persistence]] ayrı kaldı çünkü disk-format sözleşmesi farklı değişim hızında).

---

## reporters/

### Çıktılar (`paths.reports/` içine)

| Dosya | Koşul | İçerik |
|---|---|---|
| `run_report.html` | **her zaman** | tek dosya, **CDN/harici asset yok**; inline `<style>`. Koşum özeti (manifest) + EDA özeti (`DataProfile`) + leaderboard (`ScoreBoard`) + şampiyon kartı + karşılaştırma testleri + uyarılar |
| `model_card.md` | **her zaman** | Mitchell bölümleri; metadata'dan otomatik + yargı bölümleri `TODO` placeholder |
| `leaderboard.csv` | **her zaman** | `ScoreBoard.rows` → pandas (çekirdekte var) |
| `plots/*.png` | `[report]` extra varsa | actual-vs-pred · fold metrikleri · feature importance (estimator destekliyorsa). matplotlib import edilemezse **atlanır** (WARNING, hata değil) |

### Kararlar
- **HTML bağımlılıksız** — jinja2 YOK; küçük yerel string/format yardımcısı. `report`
  extra'sı yalnız `matplotlib` (jinja2 kaldırıldı).
- **Markdown** model card — git-diff'lenebilir, elle düzenlenebilir.
- Tüm kullanıcı string'leri (`html.escape`) kaçışlanır; yalnız `config.model_dump(mode="json")`
  (zaten sırsız) okunur, ondan da seçili alanlar.
- Deterministik: zaman damgaları yalnız `manifest.created_at`'ten; sabit sıralama.
- `ModelBundle.pipeline is None` → importance grafiği atlanır (bundle diskten metadata-only yüklenmiş olabilir).

### API
```
reporters/
  __init__.py    write_reports(engine_result, manifest, paths, *, reports=None, config=None) -> dict[str,str]
  html.py        render_run_report_html(...) -> str
  model_card.py  render_model_card_md(...) -> str
  tables.py      scoreboard_to_frame(scoreboard) -> pd.DataFrame
  plots.py       maybe_plots(engine_result, paths, *, reports=None) -> list[Path]   (matplotlib lazy)
```
`write_reports` → artifacts sözlüğü (rel yol → dosya adı), manifest'e merge edilir.

---

## tracking/

### Protokol (`Tracker`)
```python
class Tracker(Protocol):
    def start_run(self, run_id: str, *, project: str, config: dict[str, Any]) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None: ...
    def log_artifact(self, path: Path, *, name: str | None = None) -> None: ...
    def end_run(self, *, status: str = "ok") -> None: ...
```

### Implementasyonlar
- **`NullTracker`** — tümü no-op (`backend == none`).
- **`JsonlTracker`** (varsayılan) — `<run_dir>/tracking/events.jsonl`'a JSON satırları
  ekler (`{"ts", "kind": start|params|metrics|artifact|end, ...}`); `end_run`'da
  `tracking/summary.json` (düz params + son metrikler). **Bağımlılıksız, ağsız.**
- **`MlflowTracker`** — `[tracking]` extra (lazy import). mlflow API'sine eşler.
  `config.tracking.uri_env` → `Settings.resolve_secret` (uzak sunucu = kullanıcının
  açık tercihi; çekirdek yine ağsız).

### Resolver
`resolve_tracker(config, *, run_dir) -> Tracker`:
- `none` → `NullTracker`
- `jsonl` → `JsonlTracker`
- `mlflow` ve mlflow **import edilemez** → `ConfigError` (`pip install autoragml[tracking]`
  ipucuyla). **Fail-fast** — koşumdan önce resolve edilir, sessizce jsonl'a düşmez
  (kullanıcı sonuçları beklediği yerde bulamaz).

### Ağ
`NullTracker` + `JsonlTracker` sıfır ağ. `MlflowTracker` yalnız `uri_env` uzak bir
sunucuya işaret ederse ağ yapar — config + env üzerinden açık opt-in.

---

## Ortak

- `persistence.paths` alt dizinlerine **`tracking/`** eklenir (`RunPaths.tracking`).
- Yeni exception yok: mlflow-eksik → `ConfigError`; plot hatası → yakalanır + WARNING.
- **Kapsam dışı (ADR 0020):** orchestrator ne zaman `start_run`/`write_reports` çağırır,
  hangi params/metrics loglanır.

## Sonuç
- `reporters.write_reports` her zaman HTML+MD+CSV; plotlar opsiyonel, asla akışı kırmaz.
- `tracking` protokolü + 3 implementasyon; varsayılan bağımlılıksız JSONL, kapalı = no-op.
- `report` extra'sından jinja2 çıkarıldı; `RunPaths.tracking` eklendi.
