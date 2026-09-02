# ADR 0027 — guardrail `prediction_negative` serving-clip'e duyarlı

**Durum:** Kabul · 2026-09-02

Kaynak: m5_subset benchmark (recursive koşumu, `_runs/20260902T141028Z`). Leaderboard:
`classical_ensemble` wMAPE **77.19** (en iyi), `auto_ets` 77.83, `auto_theta` 78.81 — hepsi
`prediction_negative` ile **karantinaya alındı** (%2–5 küçük negatif tahmin, intermittent
talepte ETS/Theta additive doğası). Şampiyon → `tsb` (80.37), yalnız yapısı gereği negatif
üretmediği için. Yani en iyi tahminci, serving'de **kesinlikle 0'a kırpılacak** küçük
negatifler yüzünden eleniyor.

## Karar

`scoring.guardrails.evaluate_guardrails` artık serving'de uygulanacak **negatif-olmayan
kırpma tabanını** (`serving_clip_lower`) bilir:

- `config.postprocess.enabled` + (`clip.lower ≥ 0` **veya** `auto_nonneg` + regresyon/forecasting
  + `target_min ≥ 0`) → `serving_clip_lower = 0.0` (aynen `postprocessors.steps.resolve_clip_lower`).
- Bu taban aktifken `prediction_negative` **karantina bayrağı emit edilmez** — çünkü
  `FittedPostprocessor` served tahmini `≥ 0` garantiler.
- **İstisna (bozuk model koruması):** negatif tahmin oranı `> %50` ise bayrak yine emit edilir
  (çoğunluğu negatif = miskalibre; kırpma maskeler). Ölçek patlamaları zaten
  `prediction_scale_ratio` / `prediction_hard_abs_max` ile yakalanıyor (abs → negatif büyüklük dahil).

OOF metrikleri **ham tahmin üzerinde** kalır (ADR 0017 serving-only kilidi korunur); yalnız
**uygunluk kapısı** garantili kırpmadan haberdar olur. `apply_in_validation` hâlâ v1'de kapalı.

## Sözleşme

- `prediction_health` (`validators.frame_ops`) → `n_pred` + `frac_negative` eklendi (additive).
- `evaluate_guardrails(..., serving_clip_lower: float | None = None)` (additive kwarg).
- `GuardrailConfig` değişmedi.

## Sonuç

- `guardrails.py` + `build_scoreboard` (`resolve_clip_lower` yeniden kullanımı).
- `test_guardrails` — negatif tahmin + aktif nonneg clip → karantina yok; > %50 negatif → karantina.
- benchmark: m5 yeniden koşulur (şampiyon `tsb` → `auto_ets`/`classical_ensemble` beklenir).
