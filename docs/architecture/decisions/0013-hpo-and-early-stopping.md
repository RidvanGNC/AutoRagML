# ADR 0013 — HPO (ensemble-öncelikli, multi-fidelity, nested) + early stopping

**Durum:** Kabul · 2026-09-01

Kaynaklar: AutoGluon (HPO varsayılan kapalı; bagging+stack); multi-fidelity HPO review
2025 (SH → Hyperband → ASHA → BOHB); FLAML CFO/BlendSearch; Optuna HyperbandPruner;
GBM early-stopping best practice.

## HPO

### İlke: doğruluğa birincil yol = ensemble, HPO ikincil
AutoGluon dersi: çeşitli katalog + bagged CV + **Caruana weighted ensemble**
(vendor'landı, ADR 0004) tabloda HPO'yu yener, daha az overfit. HPO bütçe-kapılı
ikincil rafinasyon.

### Backend soyutlaması (`fine_tuners/`)
- `RandomSearch` — çekirdek, bağımlılıksız; üstünde **Successive Halving / Hyperband**
- `Optuna` — `[hpo]` extra; TPE + `HyperbandPruner`
- `FLAML` — opsiyonel; CFO/BlendSearch (maliyet-cimri)

### Detaylar
- Arama uzayı **katalog YAML**'ında (`search_space`), override'lı
- Fidelity ekseni katalogda (`fidelity`): GBM → `n_estimators`; büyük veri → data-subsample;
  erken rung → az fold
- Bütçe (ADR 0008): Hyperband doğal böler; `min_trials_per_model=3` korunur
- **HPO iç resample'da** (ADR 0010/6 nested CV) — dış test'e asla dokunmaz
- Presetler: `hpo: none` (sadece ensemble, AutoGluon tarzı) · `hpo: light` (**default** —
  RandomSearch ~15 trial + Hyperband pruning) · `hpo: thorough`

## Early stopping

### Varsayılan: fold-içi iç-val ES
Her CV fold'unda train partition'dan `early_stopping_fraction` (default 0.1; TS'de
**son parça**, zaman-farkında) ES-val ayrılır. `early_stopping_rounds` katalogdan
(default 50). ES-val fold'un train'i içinde → ADR 0011 uyumlu.

### CV-ES (opt-in, küçük/dengesiz veri)
`lgb.cv`/`xgb.cv` tarzı iç k-fold ile en iyi `n_iter`, sonra refit. Daha sağlam, pahalı.

### Final refit
Şampiyon seçilince tüm train'de yeniden fit. ES modelleri için fold'lardaki
`best_iteration` medyanı sabit `n_estimators` (AutoGluon deseni). ES + CV tek çağrıda
birleştirilmez.

`early_stopping_rounds` HPO edilmez (iyi varsayılanlar).

## Sonuç
- `contracts.TuningResult`: `best_params, trials[], spent_budget, realized_seconds,
  early_stopped, best_iteration_per_fold[], fidelity_schedule`
- `contracts.Candidate`: `search_space`, `fidelity`, `supports_early_stopping`,
  `early_stopping_rounds`
- `fine_tuners/` backend soyutlaması + SH/Hyperband zamanlayıcı
