"""Plan üretici — `DataProfile` + `TaskSpec` + `RunConfig` → `AdaptivePlan` (ADR 0007/0010/0015).

**Karar üretir; fit etmez.** `committed_ops` her zaman uygulanır; `candidate_ops` HPO
arama uzayında seçilir. Intermittency = ipucu (havuzu genişletir), router değil.
"""

from __future__ import annotations

import statistics
from typing import Literal

from autoragml.contracts.adaptive_plan import (
    AdaptivePlan,
    CandidateOpGroup,
    ColumnOp,
    RegimeDef,
    SegmentSpec,
)
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dynamics_config import DynamicsConfig
from autoragml.contracts.enums import (
    ColumnFlag,
    IntermittencyClass,
    SemanticRole,
    SpecialType,
    Task,
)
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.dynamics.recipes import validate_recipes
from autoragml.logging import get_logger

logger = get_logger(__name__)

# Model ailesi → dönüşüm yoğunluğu (ADR 0010: ağaç minimal, lineer kapsamlı).
_FAMILY_POLICY: dict[str, str] = {
    "gbdt": "minimal",
    "forest": "minimal",
    "linear": "full",
    "neural": "full",
    "distance": "full",
    "baseline": "none",
    "statistical": "none",
    "intermittent": "none",
}


def _feature_columns(profile: DataProfile, task: TaskSpec) -> list[str]:
    reserved = {c for c in (task.time_col, task.group_col, *task.targets) if c}
    return [c.name for c in profile.columns if c.name not in reserved]


def _committed_ops(profile: DataProfile, task: TaskSpec, cfg: DynamicsConfig) -> tuple[list[ColumnOp], list[str]]:
    ops: list[ColumnOp] = []
    notes: list[str] = []
    by_name = {c.name: c for c in profile.columns}

    for col in _feature_columns(profile, task):
        cp = by_name[col]
        role = cp.semantic_role

        # Yapısal düşürme
        if role is SemanticRole.CONSTANT or ColumnFlag.ALL_MISSING in cp.flags:
            ops.append(ColumnOp(op="drop", column=col, params={"reason": "constant_or_all_missing"}))
            continue
        if cp.duplicate_of is not None:
            ops.append(ColumnOp(op="drop", column=col, params={"reason": f"duplicate_of:{cp.duplicate_of}"}))
            continue
        if role is SemanticRole.ID:
            ops.append(ColumnOp(op="drop", column=col, params={"reason": "id_like"}))
            continue
        if SpecialType.TEXT in cp.special_types:
            ops.append(ColumnOp(op="drop", column=col, params={"reason": "text_v1_unsupported"}))
            notes.append(f"'{col}' metin kolonu düşürüldü — metin modalitesi v1.1.")
            continue
        if cfg.drop_leakage_suspects and ColumnFlag.LEAKAGE_SUSPECT in cp.flags:
            ops.append(ColumnOp(op="drop", column=col, params={"reason": "leakage_suspect"}))
            notes.append(f"'{col}' sızıntı şüphesiyle düşürüldü (drop_leakage_suspects=true).")
            continue

        # Datetime string → takvim özellikleri
        if SpecialType.DATETIME in cp.special_types:
            ops.append(ColumnOp(op="date_expand", column=col))
            continue

        # Kategorik kodlama
        if role in {SemanticRole.CATEGORICAL, SemanticRole.BOOLEAN}:
            if ColumnFlag.HIGH_CARDINALITY in cp.flags or cp.stats.n_unique > cfg.max_onehot_cardinality:
                ops.append(
                    ColumnOp(op="encode", column=col, params={"strategy": cfg.high_cardinality_encoding})
                )
            else:
                ops.append(
                    ColumnOp(op="encode", column=col, params={"strategy": cfg.low_cardinality_encoding})
                )
            continue

        # İmputasyon (eksik varsa)
        if cp.stats.missing_ratio > 0.0:
            numeric_roles = {SemanticRole.NUMERIC_CONTINUOUS, SemanticRole.NUMERIC_DISCRETE}
            strategy = "median" if role in numeric_roles else "most_frequent"
            ops.append(ColumnOp(op="impute", column=col, params={"strategy": strategy}))

    # time_col varsa: takvim özelliklerini exogenous olarak ekle
    if task.time_col and task.time_col in by_name:
        ops.append(ColumnOp(op="date_expand", column=task.time_col, params={"keep_original": False}))

    # Custom recipe'ler (isimle; registry doğrular)
    for recipe_name in cfg.recipes:
        ops.append(ColumnOp(op=f"recipe:{recipe_name}", column="*"))

    return ops, notes


def _candidate_ops(profile: DataProfile, task: TaskSpec, cfg: DynamicsConfig) -> list[CandidateOpGroup]:
    groups: list[CandidateOpGroup] = []
    by_name = {c.name: c for c in profile.columns}

    heavy_numeric = [
        col
        for col in _feature_columns(profile, task)
        if by_name[col].semantic_role is SemanticRole.NUMERIC_CONTINUOUS
        and (ColumnFlag.SKEWED in by_name[col].flags or ColumnFlag.HEAVY_TAILED in by_name[col].flags)
    ]
    if heavy_numeric:
        groups.append(
            CandidateOpGroup(
                group_name="heavy_tailed_numeric",
                columns=heavy_numeric,
                choices=cfg.numeric_transform_choices,
                default="none",
            )
        )

    if task.task in {Task.REGRESSION, Task.FORECASTING, Task.QUANTILE_REGRESSION}:
        tp = profile.target_profile
        choices: list[str] = []
        default = "none"
        if ColumnFlag.SKEWED in tp.flags or ColumnFlag.HEAVY_TAILED in tp.flags:
            positive = tp.stats.min is not None and tp.stats.min >= 0.0
            choices = [c for c in cfg.target_transform_choices if positive or c != "log1p"]
        # seasonal differencing (ADR 0026): forecasting + mevsim ≥ horizon + trend/mevsim gücü
        if _seasonal_diff_applicable(profile, task):
            choices = [*dict.fromkeys([*choices, "none", "seasonal_difference"])]
            default = "seasonal_difference"
        if choices:
            groups.append(
                CandidateOpGroup(
                    group_name="target",
                    columns=list(task.targets),
                    choices=choices or ["none"],
                    default=default,
                )
            )

    return groups


def _seasonal_diff_applicable(profile: DataProfile, task: TaskSpec) -> bool:
    """seasonal_difference tersine çevrilebilir mi + faydalı mı (ADR 0026)."""
    ts = profile.timeseries
    if task.task is not Task.FORECASTING or ts is None or not ts.seasonality:
        return False
    horizon = task.horizon or 1
    seasons = [s for s in ts.seasonality if 2 <= int(s.period) <= 400 and int(s.period) >= horizon]
    if not seasons:
        return False
    strong = any(s.strength >= 0.3 for s in seasons)
    trend = (ts.trend_strength or 0.0) >= 0.3
    return strong or trend


def _resolve_structure(
    profile: DataProfile, task: TaskSpec, cfg: DynamicsConfig
) -> Literal["pooled", "per_group_champion"]:
    if cfg.structure == "pooled":
        return "pooled"
    if cfg.structure == "per_group_champion":
        return "per_group_champion"
    if task.task is not Task.FORECASTING or not task.group_col or profile.timeseries is None:
        return "pooled"
    series = profile.timeseries.per_series
    n_series = len(series)
    if not (cfg.per_group_min_series <= n_series <= cfg.per_group_max_series):
        return "pooled"
    horizon = task.horizon or 1
    median_obs = statistics.median(sp.n_obs for sp in series) if series else 0
    if median_obs < cfg.per_group_min_history_multiplier * horizon:
        return "pooled"
    return "per_group_champion"


# SBC sınıfları ADI ekseninde sıralı — küçük segment en yakın komşuya birleşir (ADR 0028)
_SEGMENT_ORDER = (
    IntermittencyClass.SMOOTH,
    IntermittencyClass.ERRATIC,
    IntermittencyClass.INTERMITTENT,
    IntermittencyClass.LUMPY,
)


def _resolve_segments(profile: DataProfile, task: TaskSpec, cfg: DynamicsConfig) -> list[SegmentSpec]:
    """`per_group_champion` iken serileri SBC intermittency sınıfına göre segmentle (ADR 0028).

    Boş liste döndürürse engine pooled ilerler (tek anlamlı segment / yetersiz seri).
    """
    ts = profile.timeseries
    if ts is None or not task.group_col or not ts.per_series:
        return []

    by_class: dict[IntermittencyClass, list[str]] = {}
    for sp in ts.per_series:
        cls = sp.intermittency_class
        if cls is IntermittencyClass.INSUFFICIENT:
            cls = IntermittencyClass.INTERMITTENT  # kısa/az-aktif seriler intermittent'a
        by_class.setdefault(cls, []).append(sp.group)

    # sıralı eksende küçük grupları komşuya kaydır
    ordered = [(c, by_class.get(c, [])) for c in _SEGMENT_ORDER if by_class.get(c)]
    if not ordered:
        return []
    merged: list[tuple[str, list[str]]] = []
    for cls, ids in ordered:
        if merged and len(ids) < cfg.segment_min_series:
            prev_name, prev_ids = merged[-1]
            merged[-1] = (f"{prev_name}+{cls.value}", [*prev_ids, *ids])
        else:
            merged.append((cls.value, list(ids)))
    # baştaki küçük segment → sonrakine
    while len(merged) > 1 and len(merged[0][1]) < cfg.segment_min_series:
        (n0, i0), (n1, i1) = merged[0], merged[1]
        merged[1] = (f"{n0}+{n1}", [*i0, *i1])
        merged.pop(0)
    # sayı sınırı: en küçükleri sonrakine kat
    while len(merged) > cfg.segment_max_count:
        j = min(range(len(merged)), key=lambda k: len(merged[k][1]))
        tgt = j + 1 if j + 1 < len(merged) else j - 1
        merged[tgt] = (f"{merged[j][0]}+{merged[tgt][0]}", [*merged[j][1], *merged[tgt][1]])
        merged.pop(j)

    if len(merged) < 2:
        return []
    return [
        SegmentSpec(name=name, group_ids=sorted(ids), source="intermittency_class")
        for name, ids in merged
    ]


def _row_policies(profile: DataProfile, task: TaskSpec) -> list[str]:
    policies: list[str] = []
    ts = profile.timeseries
    if ts is None:
        return policies
    summary = ts.intermittency_summary
    for cls in (IntermittencyClass.INTERMITTENT, IntermittencyClass.LUMPY, IntermittencyClass.ERRATIC):
        if summary.get(cls.value, 0) > 0:
            policies.append(f"intermittent_augment:{cls.value}")
    if summary.get(IntermittencyClass.INSUFFICIENT.value, 0) > 0:
        policies.append("filter_low_activity")
    if task.horizon and any(sp.n_obs < 4 * task.horizon for sp in ts.per_series):
        policies.append("coldstart_split")
    return policies


_IRREGULAR = (IntermittencyClass.INTERMITTENT, IntermittencyClass.LUMPY, IntermittencyClass.ERRATIC)


def _model_hints(profile: DataProfile, task: TaskSpec) -> dict[str, dict[str, object]]:
    """Intermittency ipucundan model param ipuçları (ADR 0024). Yalnız reduction GBDT."""
    ts = profile.timeseries
    if task.task is not Task.FORECASTING or ts is None or not ts.intermittency_summary:
        return {}
    total = sum(ts.intermittency_summary.values())
    irregular = sum(ts.intermittency_summary.get(c.value, 0) for c in _IRREGULAR)
    if total == 0 or irregular / total < 0.5:
        return {}
    logger.info("[dynamics] panelin %.0f%%'i düzensiz talep — Tweedie/Poisson ipucu", irregular / total * 100)
    return {
        "lightgbm": {"objective": "tweedie", "tweedie_variance_power": 1.3},
        "xgboost": {"objective": "reg:tweedie", "tweedie_variance_power": 1.3},
        "hist_gbm": {"loss": "poisson"},
    }


def _regimes(config: RunConfig) -> list[RegimeDef]:
    if "scenario_2" not in config.scenarios:
        return []
    return [
        RegimeDef(name="trend_regime", kind="trend", params={"clusters": 3}),
        RegimeDef(name="volatility_regime", kind="volatility", params={"clusters": 3}),
        RegimeDef(name="joint_regime", kind="joint", params={}),
    ]


def build_plan(profile: DataProfile, task: TaskSpec, config: RunConfig) -> AdaptivePlan:
    """`DataProfile` + `TaskSpec` + `RunConfig` → `AdaptivePlan`."""
    cfg = config.dynamics
    validate_recipes(cfg.recipes)

    committed, notes = _committed_ops(profile, task, cfg)
    candidate = _candidate_ops(profile, task, cfg)
    structure = _resolve_structure(profile, task, cfg)
    segments = _resolve_segments(profile, task, cfg) if structure == "per_group_champion" else []
    row_policies = _row_policies(profile, task)
    regimes = _regimes(config)

    for note in notes:
        logger.info("[dynamics] %s", note)
    if segments:
        logger.info(
            "[dynamics] %d segment: %s",
            len(segments), ", ".join(f"{s.name}({len(s.group_ids)})" for s in segments),
        )

    return AdaptivePlan(
        committed_ops=committed,
        candidate_ops=candidate,
        row_policies=row_policies,
        structure=structure,
        segments=segments,
        regimes=regimes,
        family_policy=dict(_FAMILY_POLICY),
        recipes_used=list(cfg.recipes),
        model_hints=_model_hints(profile, task),
    )
