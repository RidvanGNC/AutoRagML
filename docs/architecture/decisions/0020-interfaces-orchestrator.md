# ADR 0020 — interfaces: Orchestrator + AutoRagML facade + CLI

**Durum:** Kabul · 2026-09-01

Kaynak: ADR 0011 (leakage-safe — test'e tek dokunuş), ADR 0014 ("Winning by Peeking" —
seçim yalnız OOF, nihai skor bir kez), ADR 0015 (uçtan uca akış), ADR 0018/0019
(persistence / reporters / tracking sıralaması).

## İlke

`interfaces`, çekirdeğin **tek dışa açık yüzü**: `Orchestrator` (akış), `AutoRagML`
(Python facade), `cli.py` (`autoragml run`). Üçü de aynı `Orchestrator.run`'ı çağırır.
`agent_tools.py` v2 (iskele — JSON-schema tanımı hazır, executor yok).

## Akış (`Orchestrator.run(data, config, *, resolution=None, tracker=None) -> RunResult`)

```
0. create_run_dir(config)                    → RunPaths (run_id sabitlenir)
1. resolve_tracker(config, run_dir) + start_run(run_id, project, config_snapshot)
2. [io]        load_dataset(data, config)    → Dataset (full)          | stage timeline
3. [analyze]   analyze(dataset_full, config) → DataProfile, TaskSpec
4. [holdout]   split_holdout(frame_full, config, task) → HoldoutSplit | None
5. [engine]    select_engine(task, config)
               InProcessRunner().run(engine, train_dataset, config, profile, task) → EngineResult
6. [holdout_score]  hsplit varsa + champion.pipeline: score_holdout(...) →
               champion.metrics_holdout  (**test'e TEK dokunuş — ADR 0014**)
7. [persist]   persist_run(config, dataset_full, engine_result, paths=paths,
               timeline=..., resolution=resolution, realized_seconds=...) → RunManifest
8. [report]    write_reports(engine_result, manifest, paths) → artifacts
9. tracker.log_params + log_metrics(champion OOF + holdout) + log_artifact(*) + end_run(status)
→ RunResult(engine_result, manifest, reports_dir=paths.reports)
```

### Kararlar

1. **Nihai holdout `Orchestrator`'ın işi, engine'in değil.** Engine yalnız `train_dataset`'i
   görür; `analyzers`/`planner` tüm veriyi betimler (fit etmez — ADR 0011 kural 2).
   Holdout, `validators` CV'sinden **ayrı** ve **bir kez** skorlanır.

2. **Holdout ne zaman carve edilir:**
   `n_rows < ceil(validation.min_rows_for_cv / (1 - holdout_fraction))` → **holdout yok**
   (CV'ye yeterli veri kalmalı; WARNING). Aksi halde:
   - **tabular:** seed'li rastgele `holdout_fraction` (varsayılan 0.2).
   - **timeseries:** frame `[group, time]`'a **stable** sıralanır; holdout = son
     `min(horizon, n_distinct_periods − 1)` **dönem** (global cutoff). `shift(horizon)`
     reduction özellikleri bu genişlikte leakage-safe (holdout satırı yalnız train `y`'sini görür).

3. **TS holdout skorlaması:** `pipeline.predict(scoring_frame)` çağrılır — `scoring_frame`
   **tüm sıralı frame** (reduction lag'leri hesaplayacak geçmiş lazım); tahminler
   `holdout_mask` ile seçilir. Reduction içte aynı `[group, time]` sıralamasını
   uyguladığından hizalama korunur.

4. **Hata politikası:** io/analyze hataları **propagate** (veri/config sorunu, fail-fast).
   Engine `EngineStatus.FAILED` döndürürse (runner yakalar, raise etmez) → akış **devam
   eder**: minimal manifest + rapor yine yazılır, `RunResult.engine_result.status = FAILED`,
   CLI çıkış kodu `1`. Her stage `TimelineEntry` (start/end/status) toplar.

5. **`tracker`** akışın başında resolve edilir (`run_dir` gerekli → `create_run_dir` önce).
   Enjekte edilebilir (test/embed); verilmezse `resolve_tracker`.

## `AutoRagML` facade (`interfaces/api.py`)

```python
model = AutoRagML(preset=None, config_file=None, **overrides)
result = model.fit(data, *, target, time_col=None, group_col=None, **more_overrides) -> RunResult
model.leaderboard()  ·  model.predict(X)  ·  model.explain()  ·  model.champion  ·  model.manifest

champ = AutoRagML.load(bundle_path) -> LoadedChampion   # serving; .predict(X) / .metadata
```
- `fit` → `resolve_run_config(target, preset, config_file, overrides)` →
  `Orchestrator().run(...)`. `time_col`/`group_col` override'a eklenir.
- `predict` fit'ten sonra bellekteki `champion.pipeline`'ı kullanır; `load` diskten.

## CLI (`interfaces/cli.py`, `autoragml run`)

```
autoragml run --data PATH --target COL [--preset NAME] [--config FILE]
              [--time-col COL] [--group-col COL] [--output-dir DIR] [--project-name NAME]
```
Leaderboard (ilk 10) + şampiyon + promotion + çıktı dizinini stdout'a yazar. `argparse`,
ek bağımlılık yok. Çıkış kodu: 0 (success/partial) · 1 (failed).

## Kapsam dışı / sonraya
- `agent_tools.py` executor + LLM planlayıcı → v2.
- `explain()` v1'de yapısal özet (seçim gerekçesi + guardrail + noise_floor); SHAP → `[explain]` extra, v1.1.
- Subprocess/Container runner seçimi (ADR 0006) → v1.1; v1 hep `InProcessRunner`.

## Sonuç
- `Orchestrator` tek akış; io/analyze fail-fast, engine failure graceful.
- Nihai holdout orchestrator'da, bir kez, leakage-safe carve.
- `AutoRagML().fit()` + `autoragml run` = kullanıcının iki girişi; ikisi de aynı çekirdek.
