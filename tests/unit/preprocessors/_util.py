"""Test yardımcıları."""

from __future__ import annotations

from autoragml.contracts.enums import Task
from autoragml.contracts.plan_context import PlanContext


def ctx(target: str = "y", **over: object) -> PlanContext:
    return PlanContext(target=target, task=Task.REGRESSION, **over)  # type: ignore[arg-type]
