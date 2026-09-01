"""Sızıntı taraması — yumuşak `LeakageSuspect` uyarıları (ADR 0011/5).

`analyzers` yalnız **uyarır** ve `ColumnProfile.flags`'e `leakage_suspect` ekler.
Sert kontrol (BLOCK) `validators` işi.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from autoragml.contracts.analyzer_config import ThresholdConfig
from autoragml.contracts.data_profile import ColumnProfile, LeakageSuspect
from autoragml.contracts.enums import ColumnFlag, SemanticRole

_SUSPICIOUS_NAME = re.compile(
    r"(actual|final|result|outcome|resolved|closed|ground[_-]?truth|label|target|_post_|future_)",
    re.IGNORECASE,
)


def _corr_with_target(feature: pd.Series, target: pd.Series) -> float | None:
    fx = pd.to_numeric(feature, errors="coerce")
    ty = pd.to_numeric(target, errors="coerce")
    mask = fx.notna() & ty.notna()
    if int(mask.sum()) < 10 or fx[mask].nunique() < 2 or ty[mask].nunique() < 2:
        return None
    return float(np.abs(np.corrcoef(fx[mask], ty[mask])[0, 1]))


def _is_monotonic_transform(feature: pd.Series, target: pd.Series) -> bool:
    fx = pd.to_numeric(feature, errors="coerce")
    ty = pd.to_numeric(target, errors="coerce")
    mask = fx.notna() & ty.notna()
    if int(mask.sum()) < 10:
        return False
    order = ty[mask].argsort()
    sorted_fx = fx[mask].to_numpy()[order]
    return bool(
        np.all(np.diff(sorted_fx) >= -1e-9) or np.all(np.diff(sorted_fx) <= 1e-9)
    )


def scan_leakage(
    frame: pd.DataFrame,
    *,
    columns: list[ColumnProfile],
    target: str,
    time_col: str | None,
    thr: ThresholdConfig,
) -> list[LeakageSuspect]:
    """Şüpheleri döndür ve ilgili `ColumnProfile.flags`'e `leakage_suspect` ekle."""
    suspects: list[LeakageSuspect] = []
    target_series = frame[target]

    for profile in columns:
        name = profile.name
        if name == target:
            continue
        reasons: list[tuple[str, float]] = []

        if _SUSPICIOUS_NAME.search(name):
            reasons.append(("suspicious_name", 0.5))

        corr = _corr_with_target(frame[name], target_series)
        if corr is not None and corr >= thr.leakage_corr:
            reasons.append(("near_perfect_predictor", min(1.0, corr)))
            if _is_monotonic_transform(frame[name], target_series):
                reasons.append(("target_transform", 0.9))

        if (
            time_col is not None
            and name != time_col
            and SemanticRole.DATETIME is profile.semantic_role
            and time_col in frame.columns
        ):
            feat_dt = pd.to_datetime(frame[name], errors="coerce")
            ref_dt = pd.to_datetime(frame[time_col], errors="coerce")
            mask = feat_dt.notna() & ref_dt.notna()
            if int(mask.sum()) > 0 and bool((feat_dt[mask] > ref_dt[mask]).any()):
                reasons.append(("future_dated", 0.7))

        if (
            profile.semantic_role is SemanticRole.ID
            and corr is not None
            and corr >= 0.9
        ):
            reasons.append(("sorted_id_leak", corr))

        if frame[name].isna().any() and not pdt.is_float_dtype(target_series):
            miss = frame[name].isna()
            if miss.nunique() == 2:
                grp = target_series.astype(str).groupby(miss).nunique()
                if bool((grp <= 1).all()):
                    reasons.append(("missingness_leak", 0.6))

        if reasons:
            top_reason, top_conf = max(reasons, key=lambda r: r[1])
            suspects.append(
                LeakageSuspect(
                    column=name,
                    reason=" + ".join(r for r, _ in reasons),
                    confidence=top_conf,
                )
            )
            if top_conf >= 0.6:
                profile.flags = profile.flags | {ColumnFlag.LEAKAGE_SUSPECT}

    return suspects
