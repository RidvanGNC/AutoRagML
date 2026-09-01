"""AutoRagML benchmark harness — gerçek verisetlerinde uçtan uca koşum + başarı değerlendirme.

Süre önemli değil; ölçü "sağlıklı başarı" (motto). Kullanım:

    python -m scripts.benchmarks.run                 # ilk dalga (tümü)
    python -m scripts.benchmarks.run --only adult    # tek dataset
    python -m scripts.benchmarks.run --list          # kayıtlı setleri listele
"""
