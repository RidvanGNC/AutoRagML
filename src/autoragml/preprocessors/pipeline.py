"""Feature pipeline — `AdaptivePlan`'i yürüten leakage-safe dönüşüm zinciri (ADR 0011).

`fit`/`fit_transform` yalnız train frame'inde (fold içinde, `validators` çağırır).
`target` candidate grubu burada değil — hedef dönüşümü `preprocessors.target` işi.
"""

from __future__ import annotations

import pandas as pd

from autoragml.contracts.adaptive_plan import AdaptivePlan, ColumnOp
from autoragml.contracts.enums import Provenance
from autoragml.contracts.plan_context import PlanContext
from autoragml.preprocessors.catalog import build_numeric_transform, build_op
from autoragml.transform import FittedTransform, Transform

# committed op yürütme sırası (küçük = önce)
_OP_ORDER = {"drop": 0, "recipe": 1, "date_expand": 2, "impute": 3, "encode": 4, "scale": 5}


def _op_rank(op: ColumnOp) -> int:
    key = "recipe" if op.op.startswith("recipe:") else op.op
    return _OP_ORDER.get(key, 9)


class FittedFeaturePipeline:
    """Fitted dönüşüm zinciri — yalnız `apply`."""

    __slots__ = ("provenance_fitted_on", "steps")

    def __init__(self, steps: list[FittedTransform]) -> None:
        self.steps = steps
        self.provenance_fitted_on = Provenance.TRAIN

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame
        for step in self.steps:
            out = step.apply(out)
        return out

    def get_params(self) -> dict[str, object]:
        return {"steps": [s.get_params() for s in self.steps]}


class FeaturePipeline:
    """`AdaptivePlan` + seçilmiş `candidate_ops` → fit edilebilir dönüşüm zinciri."""

    def __init__(self, transforms: list[Transform]) -> None:
        self._transforms = transforms

    @classmethod
    def from_plan(
        cls, plan: AdaptivePlan, candidate_choices: dict[str, str] | None = None
    ) -> FeaturePipeline:
        choices = candidate_choices or {}
        transforms: list[Transform] = []

        for op in sorted(plan.committed_ops, key=_op_rank):
            columns = [] if op.column == "*" else [op.column]
            built = build_op(op.op, columns, dict(op.params))
            if built is not None:
                transforms.append(built)

        for group in plan.candidate_ops:
            if group.group_name == "target":
                continue
            choice = choices.get(group.group_name, group.default)
            built = build_numeric_transform(list(group.columns), choice)
            if built is not None:
                transforms.append(built)

        return cls(transforms)

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> FittedFeaturePipeline:
        fitted, _ = self.fit_transform(frame, ctx)
        return fitted

    def fit_transform(
        self, frame: pd.DataFrame, ctx: PlanContext
    ) -> tuple[FittedFeaturePipeline, pd.DataFrame]:
        steps: list[FittedTransform] = []
        out = frame
        for transform in self._transforms:
            if hasattr(transform, "fit_transform"):
                fitted, out = transform.fit_transform(out, ctx)
            else:  # pragma: no cover - tüm transform'lar BaseTransform'dan
                fitted = transform.fit(out, ctx)
                out = fitted.apply(out)
            steps.append(fitted)
        return FittedFeaturePipeline(steps), out
