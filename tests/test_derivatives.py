from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gmr_quality.derivatives import compute_derivative_series
from gmr_quality.io import validate_motion_payload
from gmr_quality.kinematic_metrics import (
    KinematicThresholds,
    compute_smoothness_metrics,
)
from gmr_quality.types import MotionRecord, MotionValidationConfig


def make_motion(
    joint_pos: np.ndarray,
    *,
    fps: float = 10.0,
    root_pos: np.ndarray | None = None,
    root_quat_xyzw: np.ndarray | None = None,
    joint_names: tuple[str, ...] | None = None,
) -> MotionRecord:
    frames, dofs = joint_pos.shape
    if root_pos is None:
        root_pos = np.zeros((frames, 3), dtype=np.float64)
    if root_quat_xyzw is None:
        root_quat_xyzw = np.zeros((frames, 4), dtype=np.float64)
        root_quat_xyzw[:, 3] = 1.0
    payload: dict[str, object] = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_quat_xyzw,
        "dof_pos": joint_pos,
    }
    if joint_names is not None:
        payload["dof_names"] = joint_names
    result = validate_motion_payload(
        payload,
        "synthetic.pkl",
        MotionValidationConfig(expected_dof_count=dofs),
    )
    return result.require_record()


def test_constant_trajectory_derivatives_are_zero_with_documented_alignment() -> None:
    motion = make_motion(np.zeros((6, 2)))
    series = compute_derivative_series(motion)

    assert series.joint_vel_rad_s.shape == (6, 2)
    assert np.isnan(series.joint_step_abs_rad[0]).all()
    assert np.isnan(series.joint_vel_rad_s[0]).all()
    assert np.isnan(series.joint_acc_rad_s2[:2]).all()
    assert np.isnan(series.joint_jerk_rad_s3[:3]).all()
    assert np.isnan(series.root_ang_vel_rad_s[0]).all()
    np.testing.assert_array_equal(series.joint_step_abs_rad[1:], 0.0)
    np.testing.assert_array_equal(series.joint_vel_rad_s[1:], 0.0)
    np.testing.assert_array_equal(series.joint_acc_rad_s2[2:], 0.0)
    np.testing.assert_array_equal(series.joint_jerk_rad_s3[3:], 0.0)
    np.testing.assert_array_equal(series.root_vel_m_s[1:], 0.0)
    np.testing.assert_array_equal(series.root_acc_m_s2[2:], 0.0)
    np.testing.assert_array_equal(series.root_jerk_m_s3[3:], 0.0)
    np.testing.assert_array_equal(series.root_ang_vel_rad_s[1:], 0.0)


def test_linear_trajectory_uses_real_fps_and_has_zero_acceleration_and_jerk() -> None:
    fps = 20.0
    time = np.arange(8) / fps
    slopes = np.array([2.0, -3.0])
    joint_pos = time[:, None] * slopes
    root_velocity = np.array([1.0, -2.0, 0.5])
    root_pos = time[:, None] * root_velocity

    motion = make_motion(joint_pos, fps=fps, root_pos=root_pos)
    series = compute_derivative_series(motion)

    np.testing.assert_allclose(
        series.joint_vel_rad_s[1:],
        np.broadcast_to(slopes, (7, 2)),
        atol=1e-12,
    )
    np.testing.assert_allclose(series.joint_acc_rad_s2[2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(series.joint_jerk_rad_s3[3:], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        series.root_vel_m_s[1:],
        np.broadcast_to(root_velocity, (7, 3)),
        atol=1e-12,
    )
    np.testing.assert_allclose(series.root_acc_m_s2[2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(series.root_jerk_m_s3[3:], 0.0, atol=1e-10)
    summary = compute_smoothness_metrics(motion).summary
    assert summary.joint_vel_p95_rad_s == pytest.approx(3.0)
    assert summary.root_speed_p95_m_s == pytest.approx(np.linalg.norm(root_velocity))
    assert summary.root_acc_p95_m_s2 == pytest.approx(0.0, abs=1e-12)
    assert summary.root_jerk_p95_m_s3 == pytest.approx(0.0, abs=1e-10)


def test_quadratic_trajectory_has_constant_second_difference_and_zero_jerk() -> None:
    fps = 10.0
    time = np.arange(8) / fps
    acceleration = np.array([2.0, -4.0])
    joint_pos = 0.5 * (time[:, None] ** 2) * acceleration

    series = compute_derivative_series(make_motion(joint_pos, fps=fps))

    np.testing.assert_allclose(
        series.joint_acc_rad_s2[2:],
        np.broadcast_to(acceleration, (6, 2)),
        atol=1e-12,
    )
    np.testing.assert_allclose(series.joint_jerk_rad_s3[3:], 0.0, atol=1e-10)


def test_quaternion_sign_flips_do_not_create_angular_velocity_spikes() -> None:
    fps = 20.0
    omega = 2.0
    time = np.arange(8) / fps
    rotvec = np.zeros((8, 3), dtype=np.float64)
    rotvec[:, 2] = omega * time
    quaternions = Rotation.from_rotvec(rotvec).as_quat()
    quaternions[1::2] *= -1.0

    series = compute_derivative_series(
        make_motion(
            np.zeros((8, 1)),
            fps=fps,
            root_quat_xyzw=quaternions,
        )
    )

    np.testing.assert_allclose(series.root_ang_vel_rad_s[1:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(series.root_ang_vel_rad_s[1:, 2], omega, atol=1e-12)


def test_joint_jump_uses_unfiltered_step_and_reports_original_frame_and_joint() -> None:
    joint_pos = np.zeros((6, 2), dtype=np.float64)
    joint_pos[3:, 1] = 0.6
    motion = make_motion(joint_pos, joint_names=("hip", "knee"))

    metrics = compute_smoothness_metrics(motion)

    np.testing.assert_array_equal(
        metrics.joint_jump_frame,
        np.array([False, False, False, True, False, False]),
    )
    assert metrics.summary.joint_jump_count == 1
    assert metrics.summary.joint_jump_rate == pytest.approx(1 / 5)
    assert metrics.summary.joint_step_max_rad == pytest.approx(0.6)
    assert metrics.summary.worst_step_joint == "knee"
    assert metrics.summary.worst_step_frame == 3
    assert metrics.summary.worst_velocity_joint == "knee"
    assert metrics.summary.worst_velocity_frame == 3
    assert metrics.max_joint_step_joint_index[3] == 1


def test_same_frame_step_has_fps_dependent_velocity() -> None:
    joint_pos = np.zeros((4, 1), dtype=np.float64)
    joint_pos[1:] = 0.1
    slow = compute_derivative_series(make_motion(joint_pos, fps=30.0))
    fast = compute_derivative_series(make_motion(joint_pos, fps=60.0))

    assert slow.joint_vel_rad_s[1, 0] == pytest.approx(3.0)
    assert fast.joint_vel_rad_s[1, 0] == pytest.approx(6.0)


def test_threshold_configuration_accepts_scalar_or_per_joint_velocity_limits() -> None:
    scalar = KinematicThresholds(velocity_limit_rad_s=3.0)
    vector = KinematicThresholds(velocity_limit_rad_s=(1.0, 2.0))

    assert scalar.velocity_limit_rad_s == 3.0
    assert vector.velocity_limit_rad_s == (1.0, 2.0)
    with pytest.raises(ValueError, match="finite and positive"):
        KinematicThresholds(velocity_limit_rad_s=0.0)
