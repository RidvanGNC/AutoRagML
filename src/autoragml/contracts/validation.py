"""ValidationReport — `validators` çıktısı. DONDU (ADR 0010/6 + 0011).

`validators` split sınırını yöneten tek yer. Nested CV: HPO + candidate_ops seçimi
iç resample'da; dış fold yalnız skorlar. Leakage 3 kategori → BLOCK.
"""

from __future__ import annotations

from pydantic import Field

from autoragml.contracts._base import Contract
from autoragml.contracts.enums import LeakageCategory, SplitKind


class FoldReport(Contract):
    """Tek bir dış fold'un sonucu."""

    fold_id: int = Field(ge=0)
    train_span: tuple[str, str] | None = None
    test_span: tuple[str, str] | None = None
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    predictions_ref: str | None = None
    best_iteration: int | None = None


class LeakageViolation(Contract):
    """Sert sızıntı ihlali (BLOCK)."""

    category: LeakageCategory
    detail: str
    fold_id: int | None = None


class LeakageReport(Contract):
    """Fold döngüsü sızıntı denetimi."""

    status: str = "PASS"  # "PASS" | "FAIL"
    violations: list[LeakageViolation] = Field(default_factory=list)


class ValidationReport(Contract):
    """Bir aday için tam doğrulama çıktısı."""

    candidate_key: str
    scenario: str = "scenario_1"
    split_kind: SplitKind
    folds: list[FoldReport] = Field(default_factory=list)
    oof_metrics: dict[str, float] = Field(default_factory=dict)
    oof_metric_se: dict[str, float] = Field(default_factory=dict)
    oof_predictions_ref: str | None = None
    leakage: LeakageReport = Field(default_factory=LeakageReport)
    nested: bool = True  # HPO/candidate_ops iç resample'da mı (ADR 0010/6)
    realized_seconds: float = Field(default=0.0, ge=0.0)
