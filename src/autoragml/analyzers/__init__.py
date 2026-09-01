"""analyzers — deterministik perception (ADR 0010).

`analyze(dataset, config)`:
1. çalışma frame'i al (eager → handle; lazy → örneklem)
2. modality tespiti
3. kolon profilleri + target summary
4. task çıkarımı
5. timeseries tanısı (yalnız timeseries modalitesinde)
6. quality + leakage taraması (yumuşak)

Döner: `(DataProfile, TaskSpec)`. **Model eğitmez, fit etmez.**
"""

from __future__ import annotations

from autoragml.analyzers._frame import get_analysis_frame
from autoragml.analyzers.leakage import scan_leakage
from autoragml.analyzers.modality import detect_modality
from autoragml.analyzers.profiling import (
    build_column_profiles,
    build_target_summary,
)
from autoragml.analyzers.quality import scan_quality
from autoragml.analyzers.task_inference import infer_task
from autoragml.analyzers.timeseries import diagnose_timeseries
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.enums import Modality
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.exceptions import DataLoadError
from autoragml.logging import get_logger

logger = get_logger(__name__)

__all__ = ["analyze"]


def analyze(dataset: Dataset, config: RunConfig) -> tuple[DataProfile, TaskSpec]:
    """Bir `Dataset`'i çözümle → `DataProfile` + `TaskSpec`."""
    frame, sampled = get_analysis_frame(dataset, config)
    if config.target not in frame.columns:
        msg = f"Hedef kolon veride yok: {config.target!r}. Kolonlar: {list(frame.columns)[:20]}"
        raise DataLoadError(msg)

    warnings: list[str] = []
    thr = config.analyzers.thresholds

    modality, mod_warn = detect_modality(frame, config, layout=dataset.layout)
    warnings.extend(mod_warn)

    columns = build_column_profiles(frame, target=config.target, thr=thr, sampled=sampled)
    target_profile = next(c for c in columns if c.name == config.target)
    target_summary = build_target_summary(frame[config.target], config.task_hint)

    task = infer_task(frame, config, modality=modality, target_profile=target_profile)
    warnings.extend(task.inference_warnings)

    timeseries = None
    if modality is Modality.TIMESERIES:
        if config.time_col is None or config.time_col not in frame.columns:
            warnings.append(
                "timeseries modalitesi ama geçerli time_col yok — TS tanısı atlandı."
            )
        else:
            ts_profile, ts_warn = diagnose_timeseries(
                frame,
                target=config.target,
                time_col=config.time_col,
                group_col=config.group_col if config.group_col in frame.columns else None,
                config=config.analyzers.timeseries,
            )
            timeseries = ts_profile
            warnings.extend(ts_warn)

    leakage_suspects = scan_leakage(
        frame,
        columns=columns,
        target=config.target,
        time_col=config.time_col,
        thr=thr,
    )
    for suspect in leakage_suspects:
        warnings.append(f"Sızıntı şüphesi: {suspect.column} ({suspect.reason})")

    quality_flags = scan_quality(
        frame,
        columns=columns,
        target_summary=target_summary,
        target=config.target,
        thr=thr,
    )

    confidences = [c.confidence for c in columns] + [task.inference_confidence]
    overall = min(confidences) if confidences else 1.0
    if sampled:
        overall = min(overall, 0.75)

    profile = DataProfile(
        columns=columns,
        n_rows=len(frame) if not sampled else dataset.shape.n_rows,
        n_cols=frame.shape[1],
        target_profile=target_profile,
        target_summary=target_summary,
        timeseries=timeseries,
        quality_flags=quality_flags,
        leakage_suspects=leakage_suspects,
        confidence=overall,
    )

    for line in warnings:
        logger.warning("[analyzers] %s", line)

    return profile, task
