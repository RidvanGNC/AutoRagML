# ADR 0045 — hiyerarşik reconciliation (MinTrace/wls_struct)

**Durum:** Kabul · 2026-09-05 (araştırma: tourism_large — hiyerarşiyi görmezden gelen düz panel)

Kaynak: [Nixtla `hierarchicalforecast`](https://github.com/Nixtla/hierarchicalforecast) araştırması
— zaten kullandığımız Nixtla ekosistemi (`statsforecast`/`neuralforecast`/`mlforecast`) ile aynı aile.
TourismLarge (benchmark) tam bu kütüphanenin gösterim veri seti — toplam→eyalet→bölge×amaç
hiyerarşisi; biz şu ana kadar 555 seriyi bağımsız panel gibi ele alıyorduk.

## Karar

### Temel fikir — yeni motor YOK (K2)
`hierarchicalforecast.aggregate()` ile bottom-level panel üst düğümlerle (toplam, eyalet, bölge...)
**genişletilir** → genişletilmiş panel **mevcut `TimeSeriesCoreEngine`/`run_core_pipeline` akışından
aynen geçer** (klasik/reduction/nöral/foundation hepsi olduğu gibi çalışır, agrega düğümler "daha
fazla seri" olarak görülür) → şampiyon TÜM düğümlerde serving tahmini üretir → **MinTrace(wls_struct)**
tutarlı hale getirir (çocuklar toplamı = ebeveyn) → yalnız bottom-level (orijinal grain) served edilir.

### K1 — hiyerarşi bildirimi
`RunConfig.hierarchy_cols: list[str] | None` — en-agregeden en-alta, **`group_col`'un ÜSTÜ**
(`group_col` otomatik en-alt seviye — tekrar yazılmaz). Örn. `group_col="region"` +
`hierarchy_cols=["state","zone"]` → hiyerarşi `state → state/zone → state/zone/region`.
Validator: `group_col` zorunlu, `hierarchy_cols` içinde olamaz, yinelenen kolon olamaz. Otomatik
hiyerarşi algılama YOK (K5) — yalnız açık bildirimde etkin.

**Önkoşul (dokümante edilen kısıt):** `group_col` **globally unique** olmalı (panelin geri
kalanındaki genel varsayımla aynı — iki farklı üst-düğüm altında aynı `group_col` değeri asla
tekrarlanmamalı). `build_hierarchy` bunu **doğrular** — ihlalde net `ValueError`.

### K4 — yöntem: MinTrace(wls_struct), mint_shrink DEĞİL
İlk araştırma `mint_shrink` öneriyordu ama implementasyon sırasında bulundu: `mint_shrink`/`mint_cov`/
`wls_var` residual kovaryansı için **her düğümün geçmiş gerçek+fitted değer matrisini ortak bir
zaman ızgarasında** ister — bizim `OOFArrays`'te `ds` (zaman damgası) YOK. Bunu eklemek ADR 0044'teki
`group` threading'i büyüklüğünde ayrı bir cross-cutting iş olurdu. **Karar (kullanıcı onaylı):**
`wls_struct` — yalnız `S` matrisinden (düğüm başına kaç bottom-seri toplandığından) ağırlıklandırır,
residual/OOF matrisi gerekmez. Bottom-up/top-down'dan belirgin iyi, `mint_shrink`'ten basit ama daha
az optimal. `mint_shrink`'e geçiş → ADR 0045-B (OOFArrays.ds eklendiğinde).

### K3 — varsayılan çıktı: yalnız bottom-level
`FittedHierarchicalForecaster.predict(frame)` yalnız `frame`'in istediği (orijinal `group_col` ham
değeri, `ds`) çiftlerine karşılık gelen **bottom-level reconciled** sonuçları döner. Agrega
düğümlerin tahminleri dışa açılmıyor (ayrı bir accessor v1'de yok — gerekirse ADR 0045-B).

## Uygulama

- `engines/timeseries/hierarchical.py` (yeni): `build_hierarchy()` (bottom panel → `HierarchySpec`:
  genişletilmiş panel + `S` matrisi + düğüm sırası + composite-bottom↔ham-değer eşlemesi — kendi
  algoritmasıyla `aggregate()`'in "/"-birleştirme kuralını bağımsızca yeniden üretir, doğrular),
  `reconcile()` (saf `MinTrace(wls_struct)` çağrısı), `FittedHierarchicalForecaster` (`Predictor`
  protokolü — iç şampiyonu context+hedef-tarih birleşik frame'le çağırır, reconcile eder, bottom'a
  daraltır).
- `engines/timeseries/core_engine.py`: `TimeSeriesCoreEngine.run()` `config.hierarchy_cols` varsa
  `_run_hierarchical()`'a yönlendirir (segmentasyon/recursive ile birleşmez — **her zaman pooled**,
  v1 basitliği). Profil, genişletilmiş panelde artık var olmayan ham kolonları (aggregate() yalnız
  `[group_col,time_col,target_col]` tutar) referans almasın diye **budanır** (`keep = {group_col,
  time_col, target}`) — budanmadan reduction adayları `FeaturePipeline`'da eksik-kolon hatasıyla
  çöküyordu (implementasyon sırasında bulunan gerçek bug, düzeltildi). `EngineResult.finalize`
  closure'ı da sarmalanır (train+holdout refit sonrası tekrar `FittedHierarchicalForecaster`).
- `RunConfig.hierarchy_cols` + validator (K1).
- `pyproject.toml`: `[hierarchical] = hierarchicalforecast>=1.5` extra, mypy override,
  `manifest._ENV_PACKAGES` += `hierarchicalforecast`.

## Kapsam dışı (ADR 0045-B — takip)

- `mint_shrink`/`mint_cov` (residual-kovaryanslı, daha optimal) — `OOFArrays.ds` eklendiğinde.
- Agrega düğüm tahminlerine erişim (yalnız bottom served ediliyor).
- **Bundle persistence:** `FittedHierarchicalForecaster` iç şampiyon joblib-picklable ise (klasik/
  reduction) normal `save_bundle`/`load_bundle` ile SORUNSUZ çalışır (yeni sidecar kodu gerekmedi —
  slotları DataFrame/array/inner-pipeline, hepsi picklable). İç şampiyon nöral/foundation (torch
  tabanlı, sidecar gerektirir) ise persistence v1'de desteklenmiyor — joblib kendi hatasını verir,
  ayrı bir guard eklenmedi (aynı oturumda fit→predict akışı için sorun değil).
- `predict_interval()` / `explain()` — hierarchical şampiyonlarda desteklenmiyor (ADR 0044-B ile
  aynı kapsam mantığı; `getattr(...,"predict_interval",None)` capability-check zaten `None` döner).
- Segmentasyon (ADR 0028) + hiyerarşi birlikte — v1'de hiyerarşi her zaman pooled.

## Doğrulama

- `tests/unit/engines/test_hierarchical.py` (7): `build_hierarchy` şekilleri, globally-unique
  ihlali → `ValueError`, `reconcile()` matematiksel tutarlılık (çocuklar toplamı = ebeveyn — sahte
  S+y_hat ile doğrudan doğrulandı), `FittedHierarchicalForecaster.predict` (stub iç şampiyon ile
  izole + bilinmeyen grup → NaN).
- `tests/unit/config/test_resolve.py` (4): `hierarchy_cols` validator (group_col zorunlu/
  çakışmasın/tekrarsız, geçerli girdi kabul).
- `tests/unit/engines/test_engines_e2e.py` (1): 2 eyalet × 2 bölge sentetik panel → uçtan uca
  `TimeSeriesCoreEngine` → `FittedHierarchicalForecaster` şampiyon, `predict()` NaN'sız.
- Manuel e2e smoke (`AutoRagML` facade): 10 düğüm (4 bottom + 6 agrega), holdout sMAPE ~10.7,
  `champion_refit_full` (finalize sarmalayıcısı) sorunsuz çalıştı.
- ruff + mypy(147 dosya) yeşil. Tam non-neural regresyon suite koşuyor (doğrulama).
