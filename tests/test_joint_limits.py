from __future__ import annotations

import numpy as np
import pytest

from gmr_quality.io import validate_motion_payload
from gmr_quality.kinematic_metrics import (
    JointLimits,
    KinematicThresholds,
    compute_joint_feasibility_metrics,
    compute_kinematic_metrics,
)
from gmr_quality.types import MotionRecord, MotionValidationConfig


def make_motion(joint_pos: np.ndarray, fps: float = 10.0) -> MotionRecord:
    frames, dofs = joint_pos.shape
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    payload = {
        "fps": fps,
        "root_pos": np.zeros((frames, 3), dtype=np.float64),
        "root_rot": root_quat,
        "dof_pos": joint_pos,
    }
    return validate_motion_payload(
        payload,
        "limits_synthetic.pkl",
        MotionValidationConfig(expected_dof_count=dofs),
    ).require_record()


def two_joint_limits() -> JointLimits:
    return JointLimits(
        lower_rad=np.array([-1.0, -1.0]),
        upper_rad=np.array([1.0, 1.0]),
        joint_names=("left_joint", "right_joint"),
    )


def test_exact_limits_are_near_but_not_hard_violations() -> None:
    joint_pos = np.array(
        [
            [0.0, 0.0],
            [-1.0, 0.98],
            [-1.1, 1.2],
            [0.0, 0.0],
        ]
    )
    metrics = compute_joint_feasibility_metrics(
        make_motion(joint_pos),
        two_joint_limits(),
        KinematicThresholds(velocity_limit_rad_s=100.0),
    )
    summary = metrics.summary

    np.testing.assert_array_equal(
        metrics.position_violation_frame,
        np.array([False, False, True, False]),
    )
    assert summary.joint_limit_violation_frame_count == 1
    assert summary.joint_limit_violation_frame_rate == pytest.approx(0.25)
    assert summary.joint_limit_violation_element_count == 2
    assert summary.joint_limit_violation_element_rate == pytest.approx(0.25)
    assert summary.near_limit_0p05rad_frame_rate == pytest.approx(0.5)
    assert summary.near_limit_0p05rad_element_rate == pytest.approx(0.5)
    assert summary.near_limit_2pct_frame_rate == pytest.approx(0.5)
    assert summary.near_limit_2pct_element_rate == pytest.approx(0.5)
    assert summary.min_limit_margin_rad == pytest.approx(-0.2)
    assert summary.min_limit_margin_norm == pytest.approx(-0.1)
    assert summary.max_limit_excess_rad == pytest.approx(0.2)
    assert summary.worst_limit_joint == "right_joint"
    assert summary.worst_limit_frame == 2
    assert metrics.nearest_limit_joint_index[2] == 1


def test_velocity_rates_exclude_the_undefined_first_frame() -> None:
    joint_pos = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
        ]
    )
    metrics = compute_joint_feasibility_metrics(
        make_motion(joint_pos, fps=10.0),
        two_joint_limits(),
        KinematicThresholds(velocity_limit_rad_s=0.5),
    )
    summary = metrics.summary

    np.testing.assert_array_equal(
        metrics.velocity_violation_frame,
        np.array([False, True, True, True]),
    )
    assert summary.velocity_violation_frame_count == 3
    assert summary.velocity_violation_frame_rate == pytest.approx(1.0)
    assert summary.velocity_violation_element_count == 3
    assert summary.velocity_violation_element_rate == pytest.approx(0.5)
    assert summary.max_velocity_excess_rad_s == pytest.approx(0.5)
    assert summary.worst_velocity_limit_joint == "left_joint"
    assert summary.worst_velocity_limit_frame == 1
    assert np.isnan(metrics.velocity_excess_rad_s[0]).all()


def test_configured_limit_tolerance_suppresses_only_numerical_overshoot() -> None:
    joint_pos = np.zeros((4, 2), dtype=np.float64)
    joint_pos[1, 0] = 1.0 + 5e-7
    joint_pos[2, 0] = 1.0 + 2e-6
    metrics = compute_joint_feasibility_metrics(
        make_motion(joint_pos),
        two_joint_limits(),
        KinematicThresholds(
            velocity_limit_rad_s=100.0,
            joint_limit_tolerance_rad=1e-6,
        ),
    )

    np.testing.assert_array_equal(
        metrics.position_violation_frame,
        np.array([False, False, True, False]),
    )
    assert metrics.summary.joint_limit_tolerance_rad == pytest.approx(1e-6)
    assert metrics.summary.joint_limit_violation_frame_count == 1
    assert metrics.summary.max_limit_excess_rad == pytest.approx(2e-6)


def test_per_joint_velocity_thresholds_are_supported() -> None:
    joint_pos = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ]
    )
    metrics = compute_joint_feasibility_metrics(
        make_motion(joint_pos, fps=10.0),
        two_joint_limits(),
        KinematicThresholds(velocity_limit_rad_s=(2.0, 0.5)),
    )

    np.testing.assert_array_equal(
        metrics.velocity_violation[1:],
        np.array([[False, True], [False, True], [False, True]]),
    )
    np.testing.assert_array_equal(metrics.velocity_limit_rad_s, [2.0, 0.5])


def test_velocity_exactly_at_threshold_is_not_a_violation() -> None:
    joint_pos = np.arange(4, dtype=np.float64)[:, None]
    limits = JointLimits(np.array([-10.0]), np.array([10.0]))
    metrics = compute_joint_feasibility_metrics(
        make_motion(joint_pos, fps=1.0),
        limits,
        KinematicThresholds(velocity_limit_rad_s=1.0),
    )

    assert metrics.summary.velocity_violation_frame_count == 0
    assert metrics.summary.max_velocity_excess_rad_s == 0.0


def test_combined_kinematic_metrics_returns_both_dimensions() -> None:
    motion = make_motion(np.zeros((5, 2)))
    metrics = compute_kinematic_metrics(motion, two_joint_limits())

    assert metrics.smoothness.summary.joint_jump_count == 0
    assert metrics.joint_feasibility.summary.joint_limit_violation_frame_count == 0
    assert metrics.joint_feasibility.summary.velocity_violation_frame_count == 0


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ([-1.0], [1.0, 2.0], "same-length"),
        ([-1.0, 0.0], [1.0, 0.0], "greater"),
        ([-np.inf], [1.0], "finite"),
    ],
)
def test_invalid_joint_limits_are_rejected(lower, upper, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JointLimits(np.asarray(lower), np.asarray(upper))


def test_joint_limit_count_must_match_motion_dofs() -> None:
    motion = make_motion(np.zeros((4, 2)))
    one_limit = JointLimits(np.array([-1.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="motion contains 2"):
        compute_joint_feasibility_metrics(motion, one_limit)
