# ADR 0041 — TimesFM foundation-TS backend (Chronos'a alternatif)

**Durum:** Kabul · 2026-09-03 (araştırma sırası 5)

Kaynak: `2026-09-model-and-benchmark-landscape.md` — Chronos'un tek foundation-TS backend olması
karşılaştırma imkânını sınırlıyor. TimesFM (Google, arXiv, GIFT-Eval'de güçlü) ikinci backend.

## Karar

`engines/timeseries/foundation_ts.py` **backend-agnostik** hale getirildi. `_ForecastBackend`
protokolü: tek metot `forecast(context_df, h) -> DataFrame[unique_id, ds, _yhat]`.

| backend | adaptör | API | extra |
|---|---|---|---|
| `chronos` (varsayılan) | `_ChronosBackend` | `predict_df` (Bolt/Chronos-2 oto) | `[foundation-ts]` |
| `timesfm` | `_TimesFMBackend` | `TimesFM_2p5_200M_torch.from_pretrained` + `.compile(ForecastConfig)` + `.forecast(h, inputs=[np.ndarray])` — df API'si yok, ds devamı `ds.diff().median()` ile elle | `[foundation-ts-timesfm]` (`timesfm[torch]`) |

Backend `candidate.default_params["backend"]` ile seçilir. `foundation_ts.yaml`:
`timesfm_2p5` (`google/timesfm-2.5-200m-pytorch`, 200M, auth'suz HF).

- `FittedChronosForecaster` → **`FittedFoundationForecaster`** (backend adı + checkpoint saklar;
  sidecar `_meta.npz`'ye `backend` alanı — geri-uyum: yoksa `"chronos"`).
- `foundation_gate` boyut-seçimi (`want_small`) yalnız `size` alanı olan adaylara (Chronos) — TimesFM
  tek-boyut, kapıya takılmaz.
- `persistence.bundle` / `explain.attribution` / `champion` — sınıf adı güncellendi.
- `pyproject` `foundation-ts-timesfm` extra + mypy override (`timesfm.*`). `manifest._ENV_PACKAGES` += `timesfm`.

## Doğrulama

- E2e smoke (RTX 4060): `run_foundation_ts_reports` OOF sMAPE 4.56 sentetik mevsimsel panelde;
  `refit_foundation_ts` + serving gerçek forecast (std>0, fallback değil).
- Testler: `test_gate_timesfm_not_size_gated`, `test_timesfm_reports_and_serving` (skipif).
- Chronos yolu değişmedi (22 foundation/persistence testi yeşil).

## Kapsam dışı

- Moirai-2 / Lag-Llama → gerekirse aynı backend deseniyle sonra.
- TimesFM quantile çıktısı (`forecast` `(point, quantiles)` döner) — v1'de yalnız point; quantile → v1.2.
- TimesFM covariate (XReg) → ADR 0009 `Dataset.relations` açılınca.
- `_TimesFMBackend` ds-devamı düzensiz-frekanslı seride kırılgan (median diff) — panel genelde düzenli.
