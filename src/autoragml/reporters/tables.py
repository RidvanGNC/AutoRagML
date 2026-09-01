"""Leaderboard tablosu — `ScoreBoard` → pandas (ADR 0019). Bağımlılıksız."""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.scoreboard import ScoreBoard
from autoragml.scoring.metrics import lower_is_better


def scoreboard_to_frame(scoreboard: ScoreBoard) -> pd.DataFrame:
    """Birincil metriğe göre sıralı leaderboard (deterministik)."""
    metric = scoreboard.primary_metric
    rows: list[dict[str, object]] = []
    for r in scoreboard.rows:
        rows.append(
            {
                "model_key": r.model_key,
                "family": r.family,
                "scenario": r.scenario,
                metric: r.oof_metric_mean,
                "se": r.oof_metric_se,
                "quarantined": r.is_quarantined,
                "eligible": r.selection_eligible,
                "guardrails": ";".join(r.guardrail_flags),
                "n_trials": r.n_trials,
                "seconds": round(r.realized_seconds, 3),
                **{f"m::{k}": v for k, v in sorted(r.all_metrics_mean.items())},
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=[metric, "model_key"], ascending=[lower_is_better(metric), True]
        ).reset_index(drop=True)
    return df
