"""Veri kalitesi taraması — dataset düzeyi `QualityFlag`'ler (ADR 0010)."""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.analyzer_config import ThresholdConfig
from autoragml.contracts.data_profile import ColumnProfile, QualityFlag, TargetSummary
from autoragml.contracts.enums import ColumnFlag, SemanticRole


def scan_quality(
    frame: pd.DataFrame,
    *,
    columns: list[ColumnProfile],
    target_summary: TargetSummary,
    target: str,
    thr: ThresholdConfig,
) -> list[QualityFlag]:
    """Dataset düzeyi kalite bayrakları."""
    flags: list[QualityFlag] = []
    n_rows = len(frame)

    if n_rows < thr.tiny_data_rows:
        flags.append(QualityFlag(code="tiny_data", detail=f"{n_rows} satır (< {thr.tiny_data_rows})"))

    dup_rows = int(frame.duplicated().sum())
    if dup_rows > 0:
        flags.append(QualityFlag(code="duplicate_rows", detail=f"{dup_rows} birebir tekrar satır"))

    target_profile = next((c for c in columns if c.name == target), None)
    if target_profile is not None:
        if target_profile.semantic_role is SemanticRole.CONSTANT:
            flags.append(QualityFlag(code="constant_target", detail="hedef tek değer içeriyor"))
        if target_profile.stats.missing_ratio > 0:
            flags.append(
                QualityFlag(
                    code="target_has_missing",
                    detail=f"hedefin %{target_profile.stats.missing_ratio * 100:.1f}'i eksik",
                )
            )

    if target_summary.class_balance:
        min_class = min(target_summary.class_balance.values())
        if min_class < thr.severe_imbalance_ratio:
            flags.append(
                QualityFlag(
                    code="severe_imbalance",
                    detail=f"en küçük sınıf oranı %{min_class * 100:.2f}",
                )
            )

    all_missing = [c.name for c in columns if ColumnFlag.ALL_MISSING in c.flags]
    if all_missing:
        flags.append(QualityFlag(code="all_missing_columns", detail=", ".join(all_missing)))

    dupes = [f"{c.name}~{c.duplicate_of}" for c in columns if c.duplicate_of]
    if dupes:
        flags.append(QualityFlag(code="duplicate_columns", detail=", ".join(dupes)))

    return flags
