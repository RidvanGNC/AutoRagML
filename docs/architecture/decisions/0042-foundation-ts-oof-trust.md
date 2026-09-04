# ADR 0042 — foundation-TS OOF güveni: ince-kanıt marj guard'ı

**Durum:** Kabul · 2026-09-04 (dev benchmark 1. koşum bulgusu)

Kaynak: `scripts/benchmarks/_runs/20260904T054933Z/ANALYSIS.md` — dev profil benchmark'ta
forecasting 2/4 NO_IMPROVEMENT. **tourism_large gerileme:** ADR 0041 öncesi v4'te şampiyon
`auto_ets` **+%15.9** (seasonal-naive üzeri), genişlemiş havuzla şimdi `timesfm_2p5` **-%18.3**.

## Kök neden

| dataset | şampiyon | iç OOF | gerçek holdout | sapma |
|---|---|---|---|---|
| m3_monthly | timesfm_2p5 | **7.98** | 16.62 | **+108%** |
| tourism_large | timesfm_2p5 | **18.21** | 27.06 | +49% |
| m5_subset | timesfm_2p5 | 72.31 | 74.35 | +3% ✓ |
| m4_hourly | chronos_bolt | 7.02 | 9.15 | +30% |

1. **`_MAX_CV_WINDOWS = 2`** (`engines/timeseries/foundation_ts.py`) — foundation_ts adayları yalnız
   2 rolling-origin penceresinde doğrulanıyordu. 2 noktadan SE = `|w1−w2|/2` → anlamsız
   (m4_hourly chronos SE **0.028**). Zero-shot forecaster **fit yapmaz** → ek pencerenin maliyeti
   tek forward geçiş; 2 pencere kısıtının gerekçesi yoktu.
2. **Ön-eğitim kontaminasyonu:** Chronos ve TimesFM ön-eğitim korpusları M3/M4/tourism gibi standart
   forecasting benchmark'larını içerir. Aylık panellerde rolling-origin OOF "ezberden" iyimser
   (m3: OOF 8 → gerçek 17). Günlük/saatlik (m5, m4_hourly) veride sapma makul → foundation_ts orada
   gerçekten iyi.
3. **Seçimde koruma yok:** `_within_one_se` gürültülü/kararsız lideri **muaf tutuyor**
   (`SE > 2·band` filtresi `best` hariç); `_SIMPLICITY_MAX_COST = 0.03` tavanı tourism'de basitlik
   ikamesini blokluyor (`|18.21−19.64| / 19.64 = %7.3 > %3`) → `elig = [timesfm]`, şampiyon zorunlu.

## Karar (A + C + D)

### A — `_MAX_CV_WINDOWS` 2 → 4 (`foundation_ts.py`)
Klasik yolla (`_MAX_CV_WINDOWS = 3` + `default_rolling_folds`) hizalanır. `_adaptive_windows` kısa
serilerde zaten graceful düşer (`w·h + guard` gereği, `mean(len ≥ req) ≥ 0.6`). Anlamlı SE + daha
dürüst OOF nokta tahmini.

### C — ince-kanıt / kontamine-OOF marj guard'ı (`scoring/selection.py`) — **asıl fix**
`_THIN_OOF_FAMILIES = {"foundation_ts", "neural_ts"}` · `_RELIABLE_FOLDS = 3`.
`_untrusted_oof(row) = row.family ∈ _THIN_OOF_FAMILIES or row.n_folds < _RELIABLE_FOLDS`.

`select_champion`: OOF-en-iyi (`raw_best`) `_untrusted_oof` ise → en iyi **güvenilir** adayı
(`_untrusted_oof` değil) bul (`rival`). `raw_best` yalnız `|raw_best.mean − rival.mean| >
max(raw_best.se, rival.se)` ise `best` kalır; aksi halde `best = rival`. 1-SE kuralı buradan devam
eder — ince-kanıtlı aday `within` bandına girer (bilgi amaçlı `within_1se`'de görünür) ama %3
basitlik tavanında `elig`'den elenir.

`foundation` (sentetik-veri ön-eğitimli tablo: TabPFN/TabICL) kontaminasyon riski taşımaz →
`_THIN_OOF_FAMILIES`'te **değil**; tablo K-fold ile `n_folds = 4` → guard tetiklenmez.

| dataset | raw_best OOF | en iyi güvenilir | marj | max(SE) | sonuç |
|---|---|---|---|---|---|
| tourism | timesfm 18.21 | joint_ensemble 19.64 | 1.43 | 1.70 | **düşürülür** → 1-SE `auto_ets` (v4 davranışı) |
| m3 | timesfm 7.98 | joint_ensemble 11.93 | 3.95 | 0.88 | korunur (m3 zaten çözümsüz — seasonal-naive near-SOTA) |
| m5 | timesfm 72.31 | joint_ensemble 79.67 | 7.36 | 5.98 | korunur ✓ |
| m4_hourly | chronos 7.02 | joint_ensemble 16.22 | 9.20 | 0.03 | korunur ✓ |

### D — `_FAMILY_COMPLEXITY["foundation_ts"]` 2 → 4
`neural_ts` ile eşit (ikisi de GPU + opak pretrained). 1-SE bandında bir klasik/istatistik aday
foundation_ts ile eşdeğerse klasiği tercih eder. `foundation` (tablo) 2'de kalır.

## Doğrulama

- `test_adr0042_thin_evidence_foundation_ts_demoted_when_margin_below_se` (tourism sayıları → `auto_ets`)
- `test_adr0042_thin_evidence_foundation_ts_kept_when_margin_clears_se` (marj ≫ SE → korunur)
- `test_adr0042_no_reliable_rival_keeps_raw_best` (tüm adaylar ince-kanıt → ham en iyi)
- `tests/unit/scoring/` 27 test + `tests/unit/models/test_foundation.py` yeşil. ruff + mypy(145) yeşil.
- Dev benchmark tekrar koşumu (tourism ölçümü) — ADR sonrası.

## Kapsam dışı

- **m3 NO_IMPROVEMENT kalır.** M3 monthly seasonal-naive competition-grade; v4'te de en iyi
  (`patchtst`) -%9.9'du. Havuzda gerçekten daha iyi seçenek yok — bu bir seçim hatası değil.
- Kontaminasyonun kökten çözümü (foundation modeli benchmark-dışı veride ölçme) mümkün değil —
  guard, marjinal vakaların gerilemesini önler, foundation_ts'i yasaklamaz.
- Frekans-duyarlı OOF cezası (monthly → foundation_ts penaltı) — heuristik/kırılgan, reddedildi.
- `neural_ts` pencere sayısı 2'de kalır (gerçekten eğitir → pahalı); guard onu da kapsar.
