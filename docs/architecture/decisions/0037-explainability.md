# ADR 0037 — açıklanabilirlik (`explain()` — öznitelik atıfı)

**Durum:** Kabul · 2026-09-03 (kullanıcı 2 kararı kilitledi — hepsi önerilen)

**Kilitli kararlar:**
- **K1 = (a)** — tek `FittedModelPipeline` + ağaç/linear estimator + `shap` kurulu → SHAP
  (dönüştürülmüş öznitelik uzayı, örnek-başı değerler mümkün). Aksi → model-agnostik.
- **K2 = (a)** — opak pipeline (ensemble/stack/segmented/joint): `pipeline.predict` üstünde
  **model-agnostik permutation** (hedef gerekmez: her ham kolonu karıştır → `mean|Δŷ|`).
- **K3** — `feature_cols` boş şampiyonlar (klasik/nöral-TS/foundation-TS forecaster) →
  `method="unavailable"` + yapısal notlar.

Kaynak: v1.1 sırasının **son** adımı. Şu an `RunResult.explain()` yalnız **yapısal özet**
(seçim gerekçesi, guardrail, noise_floor). Gerçek öznitelik atıfı (SHAP / permutation importance)
yok. `[explain]` extra (`shap>=0.44`) `pyproject`'te tanımlı ama kullanılmıyor.

## Araştırma özeti (2026-09)

- **SHAP birleşik `Explainer`**: model tipine göre otomatik — ağaç → `TreeExplainer` (hızlı, tam),
  linear → `LinearExplainer`, diğer → `PermutationExplainer` / `KernelExplainer` (model-agnostik,
  yavaş). Çıktı: örnek-başı SHAP değerleri (n×p) + `base_value`.
- **sklearn `permutation_importance`**: bağımlılıksız, model-agnostik, herhangi bir `.predict`
  üstünde; global önem (ortalama skor düşüşü ± std). Ensemble/stack/opak pipeline için doğal seçim.
- **Global önem** = `mean(|SHAP|)` (öznitelik başına) veya permutation skor düşüşü.
- Uyarı: atıflar **dönüştürülmüş** öznitelik uzayında (one-hot / target-encode sonrası) —
  ham kolona geri eşleme kayıplı.

## İlke

- `explain()` **serving-only, tembel** — koşum sırasında hesaplanmaz (maliyet + veri örneği gerekir).
- Bağımlılık: SHAP **opsiyonel** (`[explain]`); yoksa `permutation_importance` fallback (çekirdek
  `scikit-learn` zaten var). Çekirdek ağsız/hafif kalır (ADR 0003).
- Şampiyon türü ne olursa olsun bir cevap döner (opak → model-agnostik; öznitelik uzayı yok → "yok" + not).

## Açık kararlar (kullanıcıya sorulacak)

### K1 — yöntem seti
- **(a) SHAP (varsa) + permutation fallback** — ağaç/linear şampiyon + `[explain]` kurulu →
  SHAP (hızlı/tam); aksi → `permutation_importance`. En zengin, opsiyonel bağımlılık.
- **(b) yalnız `permutation_importance`** — SHAP hiç yok; tek tip çıktı, bağımlılıksız, model-agnostik,
  daha yavaş ve yalnız global (örnek-başı yok).
- **Öneri: (a)** — SHAP tablo ML'de fiili standart; fallback her durumu kapsar.

### K2 — opak pipeline'lar (ensemble / stack / segmented / joint)
- **(a) Model-agnostik** — `pipeline.predict` üstünde `permutation_importance` (SHAP
  `PermutationExplainer` de olabilir). Tek tutarlı yol, her şampiyonu kapsar.
- **(b) Üye-bazlı + ağırlıklı topla** — her üyeyi ayrı açıkla, ensemble ağırlığıyla birleştir.
  Daha "doğru" ama stack/segmented'de anlamı bulanık, çok daha fazla kod.
- **Öneri: (a)**.

### K3 — öznitelik uzayı yok olan şampiyonlar
Klasik (StatsForecast) · nöral-TS · foundation-TS · `joint_ensemble`'ın klasik parçası →
öznitelik yok. **Öneri:** `Explanation(method="unavailable", notes="<model> öznitelik-tabanlı
değil; yapısal özet + (varsa) model parametreleri")`. `joint_ensemble` → yalnız reduction
üyeleri açıklanır + not.

## Sözleşme (kilitlenince — öneri)

- `explain/` yeni katman (yan-etkisiz, salt-okunur; `reporters`/`tracking` gibi opsiyonel sink değil,
  kullanıcı-tetikli). `explain/attribution.py`.
- `contracts/explanation.py` — `Explanation` (`method: str`, `global_importance: list[FeatureScore]`,
  `base_value: float | None`, `per_sample: list[list[float]] | None`, `feature_names: list[str]`,
  `notes: list[str]`). `FeatureScore(feature: str, importance: float, std: float | None)`.
- `explain.explain_champion(bundle: ModelBundle, data: DataFrame, task, *, method="auto",
  per_sample=False, sample_size=200) -> Explanation`.
- `RunResult.explain(data=None, *, method="auto", per_sample=False, sample_size=200) -> dict` —
  yapısal özet **korunur** + `feature_importance` / `method` / `notes` eklenir. `data=None` ve
  öznitelik gereken şampiyon → `ValueError` (temsili örnek iste).
- `AutoRagML.explain(...)` — aynı imza delege.
- `interfaces/api.LoadedChampion.explain(data, ...)` — diskten yüklenen şampiyon için.
- `pyproject` `explain` extra zaten var; `shap` mypy override.
- `Candidate`/`EngineResult`/`ModelBundle`/`RunConfig` **değişmez**.

## Kapsam dışı / sonra

- Etkileşim (SHAP interaction) değerleri → v1.1+.
- Ham-kolon geri eşleme (one-hot → orijinal) → v1.1+ (şimdilik dönüştürülmüş uzay + not).
- Karşı-olgusal / LIME → kapsam dışı.
- `explain()` çıktısını rapora/HTML'e gömme → ADR 0019 reporter genişletmesi, v1.1+.
- Persistlenen açıklama (koşumda hesapla) → kalıcı olarak tembel (maliyet).

## Sonuç (UYGULANDI — commit [pending], 2026-09-03)

- `contracts/explanation.py` — `Explanation` (`method`, `feature_names`, `global_importance:
  list[FeatureScore]` azalan sıralı, `base_value`, `per_sample`, `notes`) + `FeatureScore`.
- `explain/` yeni katman (kullanıcı-tetikli, salt-okunur):
  - `explain_champion(bundle, data, task, *, method="auto", per_sample=False, sample_size=200)`.
  - **SHAP yolu (K1a):** yalnız `FittedModelPipeline` + `shap` kurulu → `shap.Explainer` (ağaç/linear
    → doğrudan estimator, diğer → `est.predict`); `pipeline._design_matrix` (dönüştürülmüş X);
    çok-sınıf `(n,p,C)` → sınıf-ortalaması `|.|`. Kırılırsa permutation'a düşer.
  - **Model-agnostik (K2a):** `_permutation_output_importance` — her ham (rezerve-olmayan) kolonu
    `n_repeats=5` karıştır → `mean|ŷ_perm − ŷ_base|`. Hedef gerekmez. Opak pipeline + SHAP-yok.
  - **Öznitelik yok (K3):** `feature_cols` boş forecaster → `method="unavailable"` + yapısal notlar.
- `RunResult.explain(data=None, *, method, per_sample, sample_size)` — yapısal özet **korunur**,
  `out["attribution"]` eklenir; `data=None` + öznitelik-tabanlı → `attribution.method="skipped"`.
- `AutoRagML.explain(...)` delege; `LoadedChampion.explain(data, *, target=, time_col=, group_col=)`
  (metadata'dan hedef).
- `pyproject` `explain` extra (mevcut) + mypy `shap.*` override.
- Testler: `tests/unit/explain/test_attribution.py` (5 — permutation sıralama, data-yok hatası,
  forecaster "unavailable", RunResult merge, SHAP skipif).

**v1.1 ADR sırası TAMAM (0030-0037). Sıradaki: tüm v1.1 blok benchmark'ı.**
