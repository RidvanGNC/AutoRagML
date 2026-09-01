"""Şema çıkarımı — kolon → dtype string (ADR 0009).

Ham pandas dtype string'i tutulur; `RawDtype`'a eşleme `analyzers` işidir.
"""

from __future__ import annotations

import pandas as pd


def infer_schema(frame: pd.DataFrame) -> dict[str, str]:
    """Her kolonun ham dtype string'ini döndür."""
    return {str(col): str(dtype) for col, dtype in frame.dtypes.items()}


def is_datetime_like(series: pd.Series) -> bool:
    """Kolon datetime dtype mı, ya da string olarak tarihe ayrıştırılabiliyor mu."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


def is_numeric_like(series: pd.Series) -> bool:
    """Kolon numerik dtype mı, ya da string olarak sayıya çevrilebiliyor mu."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_numeric(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.99)
