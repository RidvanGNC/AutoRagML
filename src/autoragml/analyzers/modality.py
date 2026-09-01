"""Modalite tespiti — tablo mı zaman serisi mi (ADR 0010).

v1: `TABULAR | TIMESERIES`. Kural: hint > time_col/forecasting işareti > datetime kolonu
+ tekrar eden zaman damgaları.
"""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.enums import Layout, Modality, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.io.schema import is_datetime_like


def detect_modality(
    frame: pd.DataFrame,
    config: RunConfig,
    *,
    layout: Layout,
) -> tuple[Modality, list[str]]:
    """Döner: (modalite, uyarılar)."""
    warnings: list[str] = []

    explicit_ts = (
        config.modality_hint is Modality.TIMESERIES
        or config.time_col is not None
        or config.task_hint is Task.FORECASTING
        or layout in {Layout.LONG, Layout.SINGLE_SERIES, Layout.WIDE_CONVERTED}
    )
    if explicit_ts:
        if config.modality_hint is Modality.TABULAR:
            warnings.append(
                "modality_hint=tabular ama time_col/forecasting/long-layout işaretleri var "
                "→ timeseries olarak ele alınıyor"
            )
        return Modality.TIMESERIES, warnings

    if config.modality_hint is Modality.TABULAR:
        return Modality.TABULAR, warnings

    dt_cols = [str(c) for c in frame.columns if is_datetime_like(frame[c])]
    for col in dt_cols:
        parsed = pd.to_datetime(frame[col], errors="coerce", format="mixed")
        if parsed.notna().any() and bool(parsed.duplicated().any()):
            warnings.append(
                f"'{col}' datetime kolonu tekrar eden değerler içeriyor → timeseries varsayıldı. "
                "Farklıysa modality_hint verin."
            )
            return Modality.TIMESERIES, warnings

    return Modality.TABULAR, warnings
