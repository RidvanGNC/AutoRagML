"""Zaman serisi şekli tespiti + wide→long melt (ADR 0009).

Kanonik format `long`. `wide` (bir tarih kolonu + çok sayıda numerik seri kolonu,
grup kolonu yok) tespit edilip melt edilir; `layout = wide_converted`, işlem loglanır.
Wide melt yalnız **eager** kaynak için (lazy + wide → hata).
"""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.enums import Layout
from autoragml.exceptions import DataLoadError
from autoragml.io.schema import is_datetime_like
from autoragml.logging import get_logger

logger = get_logger(__name__)

_MIN_WIDE_SERIES = 3


def _datetime_columns(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if is_datetime_like(frame[c])]


def looks_wide(frame: pd.DataFrame, *, target: str, time_col: str | None, group_col: str | None) -> bool:
    """Frame wide-zaman-serisi görünümünde mi."""
    if group_col is not None:
        return False
    cols = [str(c) for c in frame.columns]
    if target in cols:
        return False
    dt_cols = _datetime_columns(frame)
    if len(dt_cols) != 1:
        return False
    time_candidate = dt_cols[0]
    if time_col is not None and time_col != time_candidate:
        return False
    others = [c for c in cols if c != time_candidate]
    if len(others) < _MIN_WIDE_SERIES:
        return False
    return all(pd.api.types.is_numeric_dtype(frame[c]) for c in others)


def melt_wide_to_long(
    frame: pd.DataFrame,
    *,
    target: str,
    time_col: str | None,
    group_col: str | None,
) -> tuple[pd.DataFrame, str, str]:
    """Wide frame'i long'a çevir. Döner: (long_df, resolved_time_col, resolved_group_col)."""
    time_candidate = _datetime_columns(frame)[0]
    resolved_time = time_col or time_candidate
    resolved_group = group_col or "series_id"
    long_df = frame.melt(
        id_vars=[time_candidate],
        var_name=resolved_group,
        value_name=target,
    )
    if resolved_time != time_candidate:
        long_df = long_df.rename(columns={time_candidate: resolved_time})
    logger.warning(
        "Wide format tespit edildi -> long'a melt edildi "
        "(seri kolonu=%r, hedef=%r, %d seri). Yalnız hedef geçmişi var: "
        "exogenous feature yok, model havuzu baseline + univariate ile sınırlı.",
        resolved_group,
        target,
        long_df[resolved_group].nunique(),
    )
    return long_df, resolved_time, resolved_group


def determine_layout(
    schema: dict[str, str],
    *,
    target: str,
    time_col: str | None,
    group_col: str | None,
) -> Layout:
    """Long/single_series/n-a ayrımı (wide zaten melt edilmiş varsayılır)."""
    cols = set(schema)
    if time_col is None:
        return Layout.NA
    if time_col not in cols:
        return Layout.NA
    if group_col is not None and group_col in cols:
        return Layout.LONG
    value_cols = cols - {time_col}
    if target in value_cols and len(value_cols) == 1:
        return Layout.SINGLE_SERIES
    return Layout.LONG


def normalize_layout(
    frame: pd.DataFrame,
    *,
    target: str,
    time_col: str | None,
    group_col: str | None,
) -> tuple[pd.DataFrame, Layout, str | None, str | None]:
    """Eager frame için: wide → melt, sonra layout belirle.

    Döner: (frame, layout, resolved_time_col, resolved_group_col).
    """
    if looks_wide(frame, target=target, time_col=time_col, group_col=group_col):
        frame, time_col, group_col = melt_wide_to_long(
            frame, target=target, time_col=time_col, group_col=group_col
        )
        return frame, Layout.WIDE_CONVERTED, time_col, group_col
    schema = {str(c): str(frame[c].dtype) for c in frame.columns}
    layout = determine_layout(schema, target=target, time_col=time_col, group_col=group_col)
    return frame, layout, time_col, group_col


def guard_lazy_not_wide(
    schema: dict[str, str],
    *,
    target: str,
    time_col: str | None,
    group_col: str | None,
) -> None:
    """Lazy kaynak wide görünüyorsa hata — melt streaming'de desteklenmiyor (v1)."""
    if group_col is not None or target in schema:
        return
    numeric_others = [
        c for c, dt in schema.items() if "float" in dt or "int" in dt
    ]
    datetime_like = [c for c, dt in schema.items() if "datetime" in dt]
    if len(datetime_like) == 1 and len(numeric_others) >= _MIN_WIDE_SERIES:
        msg = (
            "Kaynak wide-format görünüyor ve lazy yükleniyor. Wide→long melt yalnız "
            "eager modda desteklenir (v1). `io.eager_max_bytes`'i artırın veya veriyi "
            "önceden long formata çevirin."
        )
        raise DataLoadError(msg)
