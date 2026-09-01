"""preprocessors -- leakage-safe by construction (ADR 0011).

FittedTransform protokolu: stateless transform / fit(train_frame)->immutable / apply(X).
fit'i yalniz validators cagirir; split sinirini gormez. provenance_fitted_on kaydi.

Durum: iskele.
"""
