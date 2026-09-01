# ADR 0004 — Engine stratejisi: hibrit (primitif çekirdek + opsiyonel motorlar)

**Durum:** Kabul · 2026-09-01

## Bağlam
Ürünün farklılaştırıcısı: rolling-origin CV + guardrail'li skorlama + per-group
champion + scenario döngüsü. AutoGluon'u sarmak bu döngünün kontrolünü ona verir
ve ağır bağımlılık getirir. Ama AG'nin ensemble gücü değerli.

## Karar
- **Çekirdek engine'ler primitiflerle** yazılır (sklearn/lightgbm/xgboost/statsforecast
  + kendi baseline'ları), kendi CV/guardrail/champion döngüsüyle.
- AutoGluon'dan **weighted ensemble selection (Caruana greedy)** ve problem-type
  çıkarımı yardımcıları `_vendor/`'a **kopyalanıp düzenlenir** (Apache-2.0, atıf:
  NOTICE + dosya başlığı + "Modified by"). Paket bağımlılığı değil.
- Tam AutoGluon = **opsiyonel engine eklentisi** (`[autogluon]` extra), tek aday
  ailesi olarak çekirdek ScoreBoard'una girer, aynı guardrail'lere tabi. v1.1 hedefi.

## Sonuç
- Çekirdek hafif, torch'suz, 3 OS yeşil.
- `statsforecast` v1'e dahil (`[timeseries]` extra) — classical TS gücü.
- Her engine aynı `contracts` nesnelerini üretir, aynı seçim süzgecinden geçer.

## Güncelleme (ADR 0010)
DemandSensing `routing.py` deseni (demand-class → pipeline atama) **yumuşatıldı**:
intermittency sınıfı model ailesini kısıtlamaz; zero-heavy seride aday havuzunu
(Croston/TSB/SBA + Tweedie/Poisson) **genişletir** ve birincil metriği etkiler.
Nihai seçim yine holdout + guardrail döngüsü.
