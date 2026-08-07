from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .types import MotionRecord


@dataclass(frozen=True)
class DerivativeSeries:
    """Derivatives aligned to original frame indices with leading NaN rows.

    A first-order value at frame ``t`` describes the interval ending at ``t``.
    First-, second-, and third-order series therefore have 1, 2, and 3 leading
    NaN rows respectively.
    """

    joint_step_abs_rad: NDArray[np.float64]
    joint_vel_rad_s: NDArray[np.float64]
    joint_acc_rad_s2: NDArray[np.float64]
    joint_jerk_rad_s3: NDArray[np.float64]
    root_vel_m_s: NDArray[np.float64]
    root_acc_m_s2: NDArray[np.float64]
    root_jerk_m_s3: NDArray[np.float64]
    root_ang_vel_rad_s: NDArray[np.float64]


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    array.setflags(write=False)
    return array


def _aligned_finite_difference(
    values: np.ndarray,
    order: int,
    dt: float,
) -> NDArray[np.float64]:
    values_float = np.asarray(values, dtype=np.float64)
    aligned = np.full(values_float.shape, np.nan, dtype=np.float64)
    aligned[order:] = np.diff(values_float, n=order, axis=0) / (dt**order)
    return _readonly(aligned)


def compute_derivative_series(motion: MotionRecord) -> DerivativeSeries:
    """Compute raw finite differences using the trajectory's actual fps."""
    if motion.num_frames < 4:
        raise ValueError("At least 4 frames are required to compute jerk")
    if not np.isfinite(motion.fps) or motion.fps <= 0:
        raise ValueError("Motion fps must be finite and positive")

    dt = 1.0 / motion.fps
    joint_delta = _aligned_finite_difference(motion.joint_pos, order=1, dt=1.0)
    joint_step = _readonly(np.abs(joint_delta))
    joint_vel = _aligned_finite_difference(motion.joint_pos, order=1, dt=dt)
    joint_acc = _aligned_finite_difference(motion.joint_pos, order=2, dt=dt)
    joint_jerk = _aligned_finite_difference(motion.joint_pos, order=3, dt=dt)

    root_vel = _aligned_finite_difference(motion.root_pos, order=1, dt=dt)
    root_acc = _aligned_finite_difference(motion.root_pos, order=2, dt=dt)
    root_jerk = _aligned_finite_difference(motion.root_pos, order=3, dt=dt)

    rotations = Rotation.from_quat(motion.root_quat_xyzw, scalar_first=False)
    relative = rotations[:-1].inv() * rotations[1:]
    angular_velocity = relative.as_rotvec() / dt
    aligned_angular_velocity = np.full(
        (motion.num_frames, 3), np.nan, dtype=np.float64
    )
    aligned_angular_velocity[1:] = angular_velocity

    return DerivativeSeries(
        joint_step_abs_rad=joint_step,
        joint_vel_rad_s=joint_vel,
        joint_acc_rad_s2=joint_acc,
        joint_jerk_rad_s3=joint_jerk,
        root_vel_m_s=root_vel,
        root_acc_m_s2=root_acc,
        root_jerk_m_s3=root_jerk,
        root_ang_vel_rad_s=_readonly(aligned_angular_velocity),
    )
