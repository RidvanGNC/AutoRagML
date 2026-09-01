# ADR 0011 — Leakage-safe by construction

**Durum:** Kabul · 2026-09-01

Kaynak: "A Grammar of Machine Learning Workflows: Rejecting Data Leakage at Call Time"
(arXiv 2603.10742); LeakageDetector (arXiv 2503.14723 / 2509.15971).

## İlke

Sızıntı, kullanıcı disiplinine bırakılmaz — **çerçeve tasarımıyla yapısal olarak
engellenir veya çağrı zamanında yakalanır.**

## Kurallar

### 1. Üç ayrı ilkel işlem
Her dönüşüm bileşeni (`preprocessors`, `dynamics/recipes`) şu ayrımı zorunlu tutar:
- **stateless transform** — parametre öğrenmez (ör. `log1p`)
- **fit(train_frame) -> FittedTransform** — yalnız train partition'dan öğrenir,
  **immutable** bir nesne döndürür
- **apply(X) -> X'** — öğrenilmiş parametreyi uygular, saf

`fit` ve `apply` API'de ayrı. Fitted nesne değiştirilemez, split'ler arası yeniden
kullanılamaz.

### 2. Split sınırını yalnız `validators` görür
Kullanıcı / recipe kodu **hiçbir zaman** train/test ayrımına erişmez. `validators`
fold döngüsünde `fit`'e yalnız train satırlarını verir, `apply`'ı test'e uygular.
`analyzers` ve `dynamics/planner` tüm veriyi görebilir çünkü **fit etmezler** (yalnız
betimleme/karar) — ama regime/encoding gibi fit gerektiren her şey fold içine ertelenir.

### 3. Veri provenance etiketi
Her frame `provenance: train | val | test | full` taşır. Bir `FittedTransform`
hangi provenance'tan fit edildiğini kaydeder. `test`/`val`'dan fit edilmiş nesnenin
`apply`'ı başka partition'a → hata.

### 4. Çağrı-zamanı reddi
`fit(frame)` çağrısı `frame.provenance != "train"` ise (fold bağlamında) reddedilir.
Tip/sözleşme kontrolü — sızıntılı çağrı çalışmadan yakalanır.

### 5. Leakage taksonomisi (`validators/leakage_checks`)
LeakageDetector'ın 3 kategorisi:
- **Overlap** — train/test satır veya **zaman** örtüşmesi (fold split hatası)
- **Preprocessing** — split'ten önce fit (kural 1–4 ihlali)
- **Multi-test** — test setini model seçiminde tekrar tekrar kullanma →
  **dış fold'da seçim yapma**; HPO + op seçimi **iç resample**'da (nested CV)

`analyzers/leakage.scan` (yumuşak, WARNING, `DataProfile.leakage_suspects[]`):
- hedefle |corr|/MI > 0.995 → `near_perfect_predictor`
- hedefin birebir monotonik dönüşümü → `target_transform`
- kolon adı regex `(actual|final|result|outcome|resolved|closed|ground_truth|label|
  target|_post_|future_)` → `suspicious_name`
- satırın kendi `time_col`'undan ileri datetime → `future_dated`
- eksiklik deseni hedefi mükemmel ayırıyor → `missingness_leak`
- forecasting: hedeften türetilmiş shift'siz feature → `unshifted_target_feature`
- hedefe korele ID-benzeri kolon → `sorted_id_leak`

`validators` (sert, BLOCK): Overlap + Preprocessing + Multi-test.

### 6. v2 synthesis kod denetimi
`dynamics/synthesis.py` (LLM üretimi recipe) üzerinde soyut yorumlama:
Test-etiketli değerin `fit`-sınıfı işleme akışı → sızıntı hatası (Drobnjaković et al.
2024, %93 precision). Recipe ayrıca `engines/runners` sandbox'ında doğrulama koşusundan geçer.

## Sonuç
- `preprocessors` / `dynamics.recipes` protokolü kural 1–4'ü şart koşar.
- `contracts`: `FittedTransform` (immutable), `Frame.provenance` alanı.
- `validators` split sınırını yöneten **tek** yer; leakage taksonomisi 3 kategori.
- Nested CV zorunlu: HPO / candidate_op seçimi iç resample'da, dış fold yalnız skorlar.
