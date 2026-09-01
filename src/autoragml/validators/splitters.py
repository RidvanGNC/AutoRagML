"""Split stratejileri (ADR 0008/2 + 0010/6).

Frame 0..n-1 pozisyonel indeksli varsayılır (runner reset eder). Her splitter
`list[Fold]` verir. `resolve_splitter` — `config.split_policy` (kısmi) + analyzers
tabanlı seçim.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import SplitKind, Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.exceptions import AutoRagMLError

IndexArray = np.ndarray


class SplitError(AutoRagMLError):
    """Split kurulamadı (yetersiz veri, geçersiz politika)."""


@dataclass(frozen=True)
class Fold:
    """Bir dış fold — pozisyonel indeksler."""

    fold_id: int
    train_idx: IndexArray
    test_idx: IndexArray
    train_span: tuple[str, str] | None = None
    test_span: tuple[str, str] | None = None


class Splitter:
    """Split stratejisi tabanı."""

    kind: SplitKind

    def split(self, frame: pd.DataFrame) -> list[Fold]:  # pragma: no cover - taban
        raise NotImplementedError


def _time_order(frame: pd.DataFrame, time_col: str | None) -> IndexArray:
    if time_col and time_col in frame.columns:
        ts = pd.to_datetime(frame[time_col], errors="coerce")
        return np.asarray(ts.argsort(kind="stable"), dtype=int)
    return np.arange(len(frame))


def _span(frame: pd.DataFrame, idx: IndexArray, time_col: str | None) -> tuple[str, str] | None:
    if not time_col or time_col not in frame.columns or idx.size == 0:
        return None
    ts = pd.to_datetime(frame[time_col].to_numpy()[idx], errors="coerce")
    lo, hi = ts.min(), ts.max()
    if pd.isna(lo) or pd.isna(hi):
        return None
    return (lo.isoformat(), hi.isoformat())


class HoldoutSplitter(Splitter):
    kind = SplitKind.HOLDOUT

    def __init__(self, *, test_fraction: float, time_col: str | None = None) -> None:
        self._frac = test_fraction
        self._time_col = time_col

    def split(self, frame: pd.DataFrame) -> list[Fold]:
        order = _time_order(frame, self._time_col)
        n_test = max(1, int(round(len(order) * self._frac)))
        train_idx, test_idx = order[:-n_test], order[-n_test:]
        if train_idx.size == 0:
            raise SplitError("Holdout: train boş")
        return [
            Fold(
                1,
                train_idx,
                test_idx,
                _span(frame, train_idx, self._time_col),
                _span(frame, test_idx, self._time_col),
            )
        ]


class KFoldSplitter(Splitter):
    kind = SplitKind.KFOLD

    def __init__(self, *, n_splits: int, seed: int, stratify_col: str | None = None) -> None:
        self._n = n_splits
        self._seed = seed
        self._stratify = stratify_col

    def split(self, frame: pd.DataFrame) -> list[Fold]:
        n = len(frame)
        if n < self._n:
            raise SplitError(f"KFold: {n} satır < {self._n} fold")
        if self._stratify and self._stratify in frame.columns:
            from sklearn.model_selection import StratifiedKFold

            sk = StratifiedKFold(n_splits=self._n, shuffle=True, random_state=self._seed)
            pairs = sk.split(np.zeros(n), frame[self._stratify].to_numpy())
        else:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self._n, shuffle=True, random_state=self._seed)
            pairs = kf.split(np.arange(n))
        return [
            Fold(i, np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
            for i, (tr, te) in enumerate(pairs, start=1)
        ]


class GroupKFoldSplitter(Splitter):
    kind = SplitKind.GROUP_KFOLD

    def __init__(self, *, n_splits: int, group_col: str) -> None:
        self._n = n_splits
        self._group_col = group_col

    def split(self, frame: pd.DataFrame) -> list[Fold]:
        from sklearn.model_selection import GroupKFold

        groups = frame[self._group_col].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < self._n:
            raise SplitError(f"GroupKFold: {n_groups} grup < {self._n} fold")
        gk = GroupKFold(n_splits=self._n)
        return [
            Fold(i, np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
            for i, (tr, te) in enumerate(gk.split(np.arange(len(frame)), groups=groups), start=1)
        ]


class RollingOriginSplitter(Splitter):
    """DemandSensing deseni — genişleyen train, sabit horizon test, adım adım kaydırma."""

    kind = SplitKind.ROLLING_ORIGIN

    def __init__(
        self,
        *,
        time_col: str,
        group_col: str | None,
        horizon_periods: int,
        step_periods: int,
        n_folds: int,
        min_train_periods: int,
    ) -> None:
        self._time_col = time_col
        self._group_col = group_col
        self._horizon = horizon_periods
        self._step = step_periods
        self._n_folds = n_folds
        self._min_train = min_train_periods

    def _period_index(self, frame: pd.DataFrame) -> tuple[pd.Series, Sequence[pd.Timestamp]]:
        ts = pd.to_datetime(frame[self._time_col], errors="coerce")
        periods = sorted(p for p in ts.dropna().unique())
        return ts, periods

    def split(self, frame: pd.DataFrame) -> list[Fold]:
        ts, periods = self._period_index(frame)
        n_periods = len(periods)
        min_train = self._min_train or max(self._horizon, n_periods // (self._n_folds + 1))
        if n_periods < min_train + self._horizon:
            raise SplitError(
                f"RollingOrigin: {n_periods} dönem < min_train({min_train}) + horizon({self._horizon})"
            )
        max_folds = 1 + (n_periods - min_train - self._horizon) // self._step
        n_folds = max(1, min(self._n_folds, max_folds))

        folds: list[Fold] = []
        for k in range(n_folds):
            train_end = min_train + k * self._step
            test_end = train_end + self._horizon
            train_periods = set(periods[:train_end])
            test_periods = set(periods[train_end:test_end])
            train_mask = ts.isin(train_periods).to_numpy()
            test_mask = ts.isin(test_periods).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            folds.append(
                Fold(
                    k + 1,
                    train_idx,
                    test_idx,
                    _span(frame, train_idx, self._time_col),
                    _span(frame, test_idx, self._time_col),
                )
            )
        if not folds:
            raise SplitError("RollingOrigin: hiç fold üretilemedi")
        return folds


class FixedWindowSplitter(Splitter):
    kind = SplitKind.FIXED_WINDOW

    def __init__(self, *, time_col: str, train_end: str, test_end: str) -> None:
        self._time_col = time_col
        self._train_end = pd.Timestamp(train_end)
        self._test_end = pd.Timestamp(test_end)

    def split(self, frame: pd.DataFrame) -> list[Fold]:
        ts = pd.to_datetime(frame[self._time_col], errors="coerce")
        train_idx = np.flatnonzero((ts <= self._train_end).to_numpy())
        test_idx = np.flatnonzero(((ts > self._train_end) & (ts <= self._test_end)).to_numpy())
        if train_idx.size == 0 or test_idx.size == 0:
            raise SplitError("FixedWindow: train veya test boş")
        return [
            Fold(
                1,
                train_idx,
                test_idx,
                _span(frame, train_idx, self._time_col),
                _span(frame, test_idx, self._time_col),
            )
        ]


def _n_periods(frame: pd.DataFrame, time_col: str | None) -> int:
    if not time_col or time_col not in frame.columns:
        return len(frame)
    return int(pd.to_datetime(frame[time_col], errors="coerce").dropna().nunique())


def resolve_splitter(
    frame: pd.DataFrame,
    config: RunConfig,
    task: TaskSpec,
    profile: DataProfile,
) -> Splitter:
    """`split_policy` (kısmi) + görev/modalite tabanlı splitter seçimi."""
    vc = config.validation
    policy = config.split_policy
    n_rows = len(frame)
    kind = policy.kind if policy else None

    is_ts = task.task is Task.FORECASTING and task.time_col is not None
    group_col = task.group_col if (task.group_col and task.group_col in frame.columns) else None

    if kind is SplitKind.FIXED_WINDOW:
        if not (policy and policy.min_train_periods and task.time_col):
            raise SplitError("fixed_window: time_col + eşdeğer tarih sınırları gerekli")
        raise SplitError("fixed_window v1'de explicit tarih parametreleri gerektirir (henüz UI yok)")

    if is_ts or kind in {SplitKind.ROLLING_ORIGIN, SplitKind.TIME_SERIES}:
        assert task.time_col is not None
        horizon = (policy.horizon if policy and policy.horizon else None) or task.horizon or 4
        step = (policy.step if policy and policy.step else None) or vc.default_rolling_step or horizon
        n_folds = (policy.n_folds if policy and policy.n_folds else None) or vc.default_rolling_folds
        pol_min = policy.min_train_periods if policy and policy.min_train_periods else None
        min_train = pol_min or vc.default_min_train_periods
        n_periods = _n_periods(frame, task.time_col)
        if n_periods < 2 * horizon + max(min_train, horizon):
            return HoldoutSplitter(test_fraction=vc.holdout_fraction, time_col=task.time_col)
        return RollingOriginSplitter(
            time_col=task.time_col,
            group_col=group_col,
            horizon_periods=int(horizon),
            step_periods=int(step),
            n_folds=int(n_folds),
            min_train_periods=int(min_train),
        )

    if n_rows < vc.min_rows_for_cv or kind is SplitKind.HOLDOUT:
        frac = policy.test_size if (policy and policy.test_size) else vc.holdout_fraction
        return HoldoutSplitter(test_fraction=float(frac), time_col=task.time_col)

    n_splits = (policy.n_folds if policy and policy.n_folds else None) or vc.default_kfold_splits
    if group_col is not None or kind is SplitKind.GROUP_KFOLD:
        n_groups = int(frame[group_col].nunique()) if group_col else 0
        if group_col and n_groups >= n_splits:
            return GroupKFoldSplitter(n_splits=int(n_splits), group_col=group_col)

    stratify = None
    if kind is SplitKind.STRATIFIED_KFOLD or task.task in {
        Task.BINARY_CLASSIFICATION,
        Task.MULTICLASS_CLASSIFICATION,
    }:
        stratify = task.targets[0]
    return KFoldSplitter(n_splits=int(n_splits), seed=config.seed, stratify_col=stratify)


def iter_folds(splitter: Splitter, frame: pd.DataFrame) -> Iterator[Fold]:
    yield from splitter.split(frame)
