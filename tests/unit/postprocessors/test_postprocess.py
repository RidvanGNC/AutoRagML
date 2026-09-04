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


# --- conformal (ADR 0044) -------------------------------------------------


def _oof_pair(n: int = 200, spread: float = 4.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(50.0, 10.0, n)
    y_pred = y_true + rng.normal(0.0, spread, n)
    return y_true, y_pred


def test_conformal_disabled_returns_point_as_both_bounds() -> None:
    post = build_postprocessor(PostprocessConfig(), _profile(-100.0), _REG).fit(*_oof_pair())
    assert not post.has_conformal
    lower, upper = post.interval([10.0, 20.0])
    np.testing.assert_allclose(lower, upper)


def test_conformal_global_interval_covers_residuals() -> None:
    y_true, y_pred = _oof_pair(n=500, spread=4.0)
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False), conformal=ConformalConfig(enabled=True, coverage=0.9)
    )
    post = build_postprocessor(cfg, _profile(-100.0), _REG).fit(y_true, y_pred)
    assert post.has_conformal
    lower, upper = post.interval([10.0, 20.0])
    assert upper[0] > lower[0]
    # ~normal(0,4) residual → %90 kantili ~6.6; makul aralıkta olmalı
    half_width = (upper[0] - lower[0]) / 2
    assert 4.0 < half_width < 10.0
    # gerçek residual dağılımının kapsanma oranı ~coverage'a yakın olmalı
    lo_oof, hi_oof = post.interval(y_pred)
    covered = np.mean((y_true >= lo_oof) & (y_true <= hi_oof))
    assert covered > 0.80  # finite-sample marj + tek-nokta test toleransı


def test_conformal_per_group_uses_group_specific_width() -> None:
    y_true1, y_pred1 = _oof_pair(n=100, spread=1.0, seed=1)  # dar residual
    y_true2, y_pred2 = _oof_pair(n=100, spread=10.0, seed=2)  # geniş residual
    y_true = np.concatenate([y_true1, y_true2])
    y_pred = np.concatenate([y_pred1, y_pred2])
    group = np.array(["tight"] * 100 + ["wide"] * 100)

    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False),
        conformal=ConformalConfig(enabled=True, coverage=0.9, per_group=True),
    )
    postproc = build_postprocessor(cfg, _profile(-1000.0), _REG)
    fitted = postproc.fit(y_true, y_pred, group=group)  # type: ignore[call-arg]

    lower, upper = fitted.interval(
        np.array([0.0, 0.0]), group=np.array(["tight", "wide"])
    )
    tight_width = upper[0] - lower[0]
    wide_width = upper[1] - lower[1]
    assert wide_width > tight_width * 2  # farklı grup → belirgin farklı genişlik


def test_conformal_small_group_falls_back_to_global() -> None:
    y_true1, y_pred1 = _oof_pair(n=100, spread=2.0, seed=3)  # dar, yeterli örneklem
    y_true2, y_pred2 = _oof_pair(n=5, spread=20.0, seed=4)  # geniş AMA min_group_oof(10) altında
    y_true = np.concatenate([y_true1, y_true2])
    y_pred = np.concatenate([y_pred1, y_pred2])
    group = np.array(["big"] * 100 + ["tiny"] * 5)

    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False),
        conformal=ConformalConfig(enabled=True, coverage=0.9, per_group=True),
    )
    fitted = build_postprocessor(cfg, _profile(-1000.0), _REG).fit(y_true, y_pred, group=group)  # type: ignore[call-arg]

    # tiny grup örneklemi (n=5) yetersiz → group_residuals'a girmez → global genişliğe düşer.
    # group=None her zaman global'i kullanır → tiny ile aynı sonucu vermeli (aynı kod yolu).
    lo_none, hi_none = fitted.interval(np.array([0.0]))
    lo_tiny, hi_tiny = fitted.interval(np.array([0.0]), group=np.array(["tiny"]))
    np.testing.assert_allclose(hi_none - lo_none, hi_tiny - lo_tiny)

    # big grup örneklemi yeterli (n=100) → KENDİ dar kantilini kullanır → tiny'nin global'inden farklı.
    lo_big, hi_big = fitted.interval(np.array([0.0]), group=np.array(["big"]))
    assert (hi_big - lo_big) < (hi_tiny - lo_tiny)


def test_conformal_coverage_override_at_call_time() -> None:
    y_true, y_pred = _oof_pair(n=500, spread=4.0)
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False), conformal=ConformalConfig(enabled=True, coverage=0.5)
    )
    fitted = build_postprocessor(cfg, _profile(-100.0), _REG).fit(y_true, y_pred)
    lo50, hi50 = fitted.interval([0.0])
    lo95, hi95 = fitted.interval([0.0], coverage=0.95)  # fit-zamanı coverage'ı geçersiz kılar
    assert (hi95[0] - lo95[0]) > (hi50[0] - lo50[0])


def test_conformal_insufficient_oof_skips_silently() -> None:
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False), conformal=ConformalConfig(enabled=True)
    )
    post = build_postprocessor(cfg, _profile(-100.0), _REG).fit(np.array([1.0]), np.array([1.0]))
    assert not post.has_conformal
    assert post.summary["conformal"]["fitted"] is False


def test_conformal_only_enabled_is_not_noop() -> None:
    """clip/round/calibrate hepsi kapalı ama conformal açık → is_noop=False (ADR 0044)."""
    cfg = PostprocessConfig(
        clip=ClipConfig(auto_nonneg=False), conformal=ConformalConfig(enabled=True)
    )
    post = build_postprocessor(cfg, _profile(-100.0), _REG).fit(*_oof_pair())
    assert not post.is_noop
    assert post.has_conformal
