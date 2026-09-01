"""postprocessors — fit-ayrımlı düzeltme zinciri (ADR 0017)."""

from __future__ import annotations

import numpy as np
import pytest

from autoragml.contracts.data_profile import (
    ColumnProfile,
    ColumnStats,
    DataProfile,
    TargetSummary,
)
from autoragml.contracts.enums import Modality, RawDtype, SemanticRole, Task
from autoragml.contracts.postprocess_config import (
    CalibrateConfig,
    ClipConfig,
    ConformalConfig,
    PostprocessConfig,
    RoundConfig,
)
from autoragml.contracts.task_spec import TaskSpec
from autoragml.postprocessors import build_postprocessor


def _profile(target_min: float) -> DataProfile:
    tgt = ColumnProfile(
        name="y",
        raw_dtype=RawDtype.FLOAT,
        semantic_role=SemanticRole.TARGET,
        stats=ColumnStats(n_unique=50, missing_ratio=0.0, min=target_min, max=target_min + 100.0),
    )
    return DataProfile(
        columns=[tgt],
        n_rows=100,
        n_cols=1,
        target_profile=tgt,
        target_summary=TargetSummary(),
    )


_REG = TaskSpec(task=Task.REGRESSION, modality=Modality.TABULAR, targets=["y"])
_CLS = TaskSpec(task=Task.BINARY_CLASSIFICATION, modality=Modality.TABULAR, targets=["y"])


# --- contract v1 guardları ---------------------------------------------------


def test_conformal_enabled_rejected_v1() -> None:
    with pytest.raises(ValueError, match="conformal"):
        PostprocessConfig(conformal=ConformalConfig(enabled=True))


def test_apply_in_validation_rejected_v1() -> None:
    with pytest.raises(ValueError, match="apply_in_validation"):
        PostprocessConfig(apply_in_validation=True)


def test_clip_bounds_ordering() -> None:
    with pytest.raises(ValueError, match="lower"):
        ClipConfig(lower=10.0, upper=1.0)


def test_calibrate_ratio_bounds_validated() -> None:
    with pytest.raises(ValueError, match="ratio_bounds"):
        CalibrateConfig(method="multiplicative", ratio_bounds=(5.0, 1.0))


# --- auto_nonneg -----------------------------------------------------------


def test_auto_nonneg_clips_when_target_nonnegative() -> None:
    post = build_postprocessor(PostprocessConfig(), _profile(target_min=0.0), _REG).fit()
    assert not post.is_noop
    np.testing.assert_allclose(post.apply([-5.0, 3.0, -0.1]), [0.0, 3.0, 0.0])


def test_auto_nonneg_noop_when_target_has_negatives() -> None:
    post = build_postprocessor(PostprocessConfig(), _profile(target_min=-4.0), _REG).fit()
    assert post.is_noop
    np.testing.assert_allclose(post.apply([-5.0, 3.0]), [-5.0, 3.0])


def test_auto_nonneg_skipped_for_classification() -> None:
    post = build_postprocessor(PostprocessConfig(), _profile(target_min=0.0), _CLS).fit()
    assert post.is_noop


def test_auto_nonneg_disabled_flag() -> None:
    cfg = PostprocessConfig(clip=ClipConfig(auto_nonneg=False))
    post = build_postprocessor(cfg, _profile(target_min=0.0), _REG).fit()
    assert post.is_noop


def test_explicit_lower_wins_over_auto() -> None:
    cfg = PostprocessConfig(clip=ClipConfig(lower=2.0))
    post = build_postprocessor(cfg, _profile(target_min=0.0), _REG).fit()
    np.testing.assert_allclose(post.apply([0.0, 5.0]), [2.0, 5.0])


# --- clip / round --------------------------------------------------------


def test_explicit_clip_both_ends() -> None:
    cfg = PostprocessConfig(clip=ClipConfig(lower=1.0, upper=10.0))
    post = build_postprocessor(cfg, _profile(-100.0), _REG).fit()
    np.testing.assert_allclose(post.apply([-3.0, 5.0, 42.0]), [1.0, 5.0, 10.0])


def test_auto_upper_from_oof_percentile() -> None:
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False, auto_upper_multiplier=3.0, auto_upper_percentile=50.0)
    )
    y_true = np.array([2.0, 2.0, 2.0, 2.0])  # p50 = 2 → upper = 6
    post = build_postprocessor(cfg, _profile(-100.0), _REG).fit(y_true, y_true)
    np.testing.assert_allclose(post.apply([5.0, 25.0]), [5.0, 6.0])


def test_round_nearest_and_threshold() -> None:
    near = build_postprocessor(
        PostprocessConfig(clip=ClipConfig(auto_nonneg=False), round=RoundConfig(mode="nearest")),
        _profile(-1.0),
        _REG,
    ).fit()
    np.testing.assert_allclose(near.apply([1.4, 1.6, 2.5]), [1.0, 2.0, 2.0])

    thr = build_postprocessor(
        PostprocessConfig(
            clip=ClipConfig(auto_nonneg=False), round=RoundConfig(mode="threshold", threshold=0.7)
        ),
        _profile(-1.0),
        _REG,
    ).fit()
    np.testing.assert_allclose(thr.apply([1.5, 1.8, 2.7]), [1.0, 2.0, 3.0])


# --- calibrate ---------------------------------------------------------


def test_calibrate_additive_bias() -> None:
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False), calibrate=CalibrateConfig(method="additive_bias")
    )
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([12.0, 12.0, 12.0])  # bias = +2
    post = build_postprocessor(cfg, _profile(-1.0), _REG).fit(y_true, y_pred)
    np.testing.assert_allclose(post.apply([12.0, 20.0]), [10.0, 18.0])


def test_calibrate_multiplicative_and_clamp() -> None:
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False),
        calibrate=CalibrateConfig(method="multiplicative", ratio_bounds=(0.2, 5.0)),
    )
    post = build_postprocessor(cfg, _profile(-1.0), _REG).fit(
        np.array([10.0, 10.0, 10.0]), np.array([20.0, 20.0, 20.0])
    )
    np.testing.assert_allclose(post.apply([20.0]), [10.0])  # ratio 0.5

    clamped = build_postprocessor(cfg, _profile(-1.0), _REG).fit(
        np.array([1000.0]), np.array([10.0])
    )
    np.testing.assert_allclose(clamped.apply([10.0]), [50.0])  # raw 100 → clamp 5.0


def test_calibrate_skipped_without_oof() -> None:
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False), calibrate=CalibrateConfig(method="additive_bias")
    )
    post = build_postprocessor(cfg, _profile(-1.0), _REG).fit(None, None)
    assert post.is_noop
    assert post.summary["calibrate"]["bias"] is None


def test_order_calibrate_then_clip_then_round() -> None:
    cfg = PostprocessConfig(
        clip=ClipConfig(lower=0.0),
        round=RoundConfig(mode="nearest"),
        calibrate=CalibrateConfig(method="additive_bias"),
    )
    # bias = mean(pred - true) = -0.5 → apply: y - (-0.5) = y + 0.5
    post = build_postprocessor(cfg, _profile(0.0), _REG).fit(
        np.array([10.0, 10.0]), np.array([9.5, 9.5])
    )
    # -1.2 + 0.5 = -0.7 → clip 0 → 0 → round 0
    np.testing.assert_allclose(post.apply([-1.2, 4.4]), [0.0, 5.0])


def test_business_rule_applied_last_and_immutable() -> None:
    post = build_postprocessor(
        PostprocessConfig(clip=ClipConfig(lower=0.0)), _profile(0.0), _REG
    ).fit()
    halved = post.with_business_rule(lambda a: np.asarray(a) / 2.0)
    assert post is not halved
    np.testing.assert_allclose(post.apply([-2.0, 8.0]), [0.0, 8.0])
    np.testing.assert_allclose(halved.apply([-2.0, 8.0]), [0.0, 4.0])
    assert halved.summary["business_rule"] is True
