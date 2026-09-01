"""Opsiyonel grafikler — matplotlib (`[report]` extra, ADR 0019).

matplotlib import edilemezse `[]` döner (WARNING, hata değil). Her grafik ayrı
try/except — biri patlarsa diğerleri yazılır.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.validation import ValidationReport
from autoragml.logging import get_logger
from autoragml.persistence.paths import RunPaths

logger = get_logger(__name__)


def maybe_plots(
    result: EngineResult, paths: RunPaths, *, reports: list[ValidationReport] | None = None
) -> list[Path]:
    """`reports/plots/` içine PNG'ler; matplotlib yoksa boş liste."""
    if importlib.util.find_spec("matplotlib") is None:
        logger.warning("[reporters] matplotlib yok — grafikler atlandı (`pip install autoragml[report]`)")
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = paths.reports / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(fig: object, name: str) -> None:
        dest = out_dir / name
        fig.savefig(dest, dpi=110, bbox_inches="tight")  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]
        written.append(dest)

    # 1. Leaderboard — birincil metrik çubukları
    try:
        rows = sorted(result.scoreboard.rows, key=lambda r: r.oof_metric_mean)
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1))
        ax.barh([r.model_key for r in rows], [r.oof_metric_mean for r in rows],
                xerr=[r.oof_metric_se for r in rows])
        ax.set_xlabel(result.scoreboard.primary_metric)
        ax.invert_yaxis()
        _save(fig, "leaderboard.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[reporters] leaderboard grafiği atlandı: %s", exc)

    # 2. Fold metrikleri (reports verilmişse) — şampiyon
    if reports:
        try:
            champ_key = result.selection.champion.model_key
            rep = next((r for r in reports if r.candidate_key == champ_key), None)
            metric = result.scoreboard.primary_metric
            if rep and rep.folds:
                vals = [f.metrics.get(metric, np.nan) for f in rep.folds]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(range(1, len(vals) + 1), vals, marker="o")
                ax.set_xlabel("fold")
                ax.set_ylabel(metric)
                ax.set_title(f"{champ_key} — fold {metric}")
                _save(fig, "champion_folds.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[reporters] fold grafiği atlandı: %s", exc)

    # 3. Feature importance (estimator destekliyorsa)
    try:
        pipeline = result.champion.pipeline
        est = getattr(pipeline, "estimator", None)
        importances = getattr(est, "feature_importances_", None)
        cols = result.champion.metadata.feature_cols
        if importances is not None and len(importances) == len(cols):
            order = np.argsort(importances)[::-1][:25]
            fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1))
            ax.barh([cols[i] for i in order], [importances[i] for i in order])
            ax.invert_yaxis()
            ax.set_xlabel("importance")
            _save(fig, "feature_importance.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[reporters] importance grafiği atlandı: %s", exc)

    return written
