from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .derivatives import DerivativeSeries, compute_derivative_series
from .types import MotionRecord


@dataclass(frozen=True)
class KinematicThresholds:
    joint_jump_step_rad: float = 0.5
    velocity_limit_rad_s: float | tuple[float, ...] = 3.0 * np.pi
    joint_limit_tolerance_rad: float = 0.0
    near_limit_margin_rad: float = 0.05
    near_limit_fraction: float = 0.02

    def __post_init__(self) -> None:
        scalar_values = (
            ("joint_jump_step_rad", self.joint_jump_step_rad),
            ("joint_limit_tolerance_rad", self.joint_limit_tolerance_rad),
            ("near_limit_margin_rad", self.near_limit_margin_rad),
            ("near_limit_fraction", self.near_limit_fraction),
        )
        for name, value in scalar_values:
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.joint_jump_step_rad == 0:
            raise ValueError("joint_jump_step_rad must be positive")
        if not 0 <= self.near_limit_fraction < 0.5:
            raise ValueError("near_limit_fraction must be in [0, 0.5)")
        velocity = np.asarray(self.velocity_limit_rad_s, dtype=np.float64)
        if velocity.ndim > 1 or velocity.size == 0:
            raise ValueError("velocity_limit_rad_s must be a scalar or one-dimensional sequence")
        if not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise ValueError("velocity limits must be finite and positive")

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "KinematicThresholds":
        thresholds = config["thresholds"]
        raw_velocity = thresholds["velocity_limit_rad_s"]
        velocity: float | tuple[float, ...]
        if isinstance(raw_velocity, (list, tuple)):
            velocity = tuple(float(value) for value in raw_velocity)
        else:
            velocity = float(raw_velocity)
        return cls(
            joint_jump_step_rad=float(thresholds["joint_jump_step_rad"]),
            velocity_limit_rad_s=velocity,
            joint_limit_tolerance_rad=float(
                thresholds.get("joint_limit_tolerance_rad", 0.0)
            ),
            near_limit_margin_rad=float(thresholds["near_limit_margin_rad"]),
            near_limit_fraction=float(thresholds["near_limit_fraction"]),
        )


@dataclass(frozen=True)
class JointLimits:
    lower_rad: NDArray[np.float64]
    upper_rad: NDArray[np.float64]
    joint_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        lower = np.array(self.lower_rad, dtype=np.float64, copy=True)
        upper = np.array(self.upper_rad, dtype=np.float64, copy=True)
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
            raise ValueError("Joint lower and upper limits must be same-length 1D arrays")
        if lower.size == 0:
            raise ValueError("Joint limits cannot be empty")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Joint limits must be finite")
        if np.any(upper <= lower):
            raise ValueError("Every upper joint limit must be greater than its lower limit")

        names = self.joint_names
        if names is None:
            names = tuple(f"joint_{index}" for index in range(lower.size))
        if (
            len(names) != lower.size
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint_names must contain one unique non-empty name per joint")

        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower_rad", lower)
        object.__setattr__(self, "upper_rad", upper)
        object.__setattr__(self, "joint_names", tuple(names))

    @property
    def num_dofs(self) -> int:
        return int(self.lower_rad.size)


@dataclass(frozen=True)
class SmoothnessSummary:
    joint_jump_threshold_rad: float
    joint_jump_rate: float
    joint_jump_count: int
    joint_step_p95_rad: float
    joint_step_max_rad: float
    worst_step_joint: str
    worst_step_frame: int
    joint_vel_p95_rad_s: float
    joint_vel_max_rad_s: float
    joint_acc_p95_rad_s2: float
    joint_acc_max_rad_s2: float
    joint_jerk_p95_rad_s3: float
    joint_jerk_max_rad_s3: float
    worst_velocity_joint: str
    worst_velocity_frame: int
    worst_jerk_joint: str
    worst_jerk_frame: int
    root_speed_p95_m_s: float
    root_speed_max_m_s: float
    root_acc_p95_m_s2: float
    root_acc_max_m_s2: float
    root_jerk_p95_m_s3: float
    root_jerk_max_m_s3: float
    root_ang_vel_p95_rad_s: float
    root_ang_vel_max_rad_s: float


@dataclass(frozen=True)
class SmoothnessMetrics:
    derivatives: DerivativeSeries
    joint_jump_frame: NDArray[np.bool_]
    max_joint_step_rad: NDArray[np.float64]
    max_joint_step_joint_index: NDArray[np.int64]
    max_joint_vel_rad_s: NDArray[np.float64]
    max_joint_vel_joint_index: NDArray[np.int64]
    max_joint_acc_rad_s2: NDArray[np.float64]
    max_joint_acc_joint_index: NDArray[np.int64]
    max_joint_jerk_rad_s3: NDArray[np.float64]
    max_joint_jerk_joint_index: NDArray[np.int64]
    root_speed_m_s: NDArray[np.float64]
    root_acc_m_s2: NDArray[np.float64]
    root_jerk_m_s3: NDArray[np.float64]
    root_ang_vel_rad_s: NDArray[np.float64]
    summary: SmoothnessSummary


@dataclass(frozen=True)
class JointFeasibilitySummary:
    joint_limit_tolerance_rad: float
    near_limit_margin_threshold_rad: float
    near_limit_fraction_threshold: float
    joint_limit_violation_frame_rate: float
    joint_limit_violation_frame_count: int
    joint_limit_violation_element_rate: float
    joint_limit_violation_element_count: int
    max_limit_excess_rad: float
    worst_limit_joint: str | None
    worst_limit_frame: int | None
    near_limit_0p05rad_frame_rate: float
    near_limit_0p05rad_element_rate: float
    near_limit_2pct_frame_rate: float
    near_limit_2pct_element_rate: float
    min_limit_margin_rad: float
    min_limit_margin_norm: float
    velocity_violation_frame_rate: float
    velocity_violation_frame_count: int
    velocity_violation_element_rate: float
    velocity_violation_element_count: int
    max_velocity_excess_rad_s: float
    worst_velocity_limit_joint: str | None
    worst_velocity_limit_frame: int | None


@dataclass(frozen=True)
class JointFeasibilityMetrics:
    position_violation: NDArray[np.bool_]
    position_violation_frame: NDArray[np.bool_]
    limit_excess_rad: NDArray[np.float64]
    margin_rad: NDArray[np.float64]
    margin_norm: NDArray[np.float64]
    min_limit_margin_rad: NDArray[np.float64]
    min_limit_margin_norm: NDArray[np.float64]
    nearest_limit_joint_index: NDArray[np.int64]
    near_limit_0p05rad: NDArray[np.bool_]
    near_limit_0p05rad_frame: NDArray[np.bool_]
    near_limit_2pct: NDArray[np.bool_]
    near_limit_2pct_frame: NDArray[np.bool_]
    velocity_limit_rad_s: NDArray[np.float64]
    velocity_violation: NDArray[np.bool_]
    velocity_violation_frame: NDArray[np.bool_]
    velocity_excess_rad_s: NDArray[np.float64]
    summary: JointFeasibilitySummary


@dataclass(frozen=True)
class KinematicMetrics:
    smoothness: SmoothnessMetrics
    joint_feasibility: JointFeasibilityMetrics


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _finite_values(array: np.ndarray) -> NDArray[np.float64]:
    values = np.abs(np.asarray(array, dtype=np.float64))
    return values[np.isfinite(values)]


def _p95(array: np.ndarray) -> float:
    values = _finite_values(array)
    if values.size == 0:
        raise ValueError("Metric contains no finite samples")
    return float(np.percentile(values, 95))


def _max(array: np.ndarray) -> float:
    values = _finite_values(array)
    if values.size == 0:
        raise ValueError("Metric contains no finite samples")
    return float(np.max(values))


def _rowwise_abs_max(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(np.asarray(array, dtype=np.float64))
    valid_rows = np.isfinite(absolute).any(axis=1)
    safe = np.where(np.isfinite(absolute), absolute, -np.inf)
    values = np.full(absolute.shape[0], np.nan, dtype=np.float64)
    indices = np.full(absolute.shape[0], -1, dtype=np.int64)
    values[valid_rows] = np.max(safe[valid_rows], axis=1)
    indices[valid_rows] = np.argmax(safe[valid_rows], axis=1)
    return _readonly(values), _readonly(indices)


def _worst_2d(array: np.ndarray, joint_names: tuple[str, ...]) -> tuple[str, int]:
    absolute = np.abs(np.asarray(array, dtype=np.float64))
    safe = np.where(np.isfinite(absolute), absolute, -np.inf)
    flat_index = int(np.argmax(safe))
    frame, joint = np.unravel_index(flat_index, safe.shape)
    return joint_names[int(joint)], int(frame)


def _vector_norm(array: np.ndarray) -> NDArray[np.float64]:
    return _readonly(np.linalg.norm(np.asarray(array, dtype=np.float64), axis=1))


def _joint_names(motion: MotionRecord) -> tuple[str, ...]:
    if motion.joint_names is not None:
        return motion.joint_names
    return tuple(f"joint_{index}" for index in range(motion.num_dofs))


def _smoothness_from_derivatives(
    motion: MotionRecord,
    derivatives: DerivativeSeries,
    thresholds: KinematicThresholds,
) -> SmoothnessMetrics:
    names = _joint_names(motion)
    max_step, max_step_joint = _rowwise_abs_max(derivatives.joint_step_abs_rad)
    max_vel, max_vel_joint = _rowwise_abs_max(derivatives.joint_vel_rad_s)
    max_acc, max_acc_joint = _rowwise_abs_max(derivatives.joint_acc_rad_s2)
    max_jerk, max_jerk_joint = _rowwise_abs_max(derivatives.joint_jerk_rad_s3)

    jump_frame = _readonly(
        np.isfinite(max_step) & (max_step > thresholds.joint_jump_step_rad)
    )
    root_speed = _vector_norm(derivatives.root_vel_m_s)
    root_acc = _vector_norm(derivatives.root_acc_m_s2)
    root_jerk = _vector_norm(derivatives.root_jerk_m_s3)
    root_ang_vel = _vector_norm(derivatives.root_ang_vel_rad_s)

    worst_step_joint, worst_step_frame = _worst_2d(
        derivatives.joint_step_abs_rad, names
    )
    worst_velocity_joint, worst_velocity_frame = _worst_2d(
        derivatives.joint_vel_rad_s, names
    )
    worst_jerk_joint, worst_jerk_frame = _worst_2d(
        derivatives.joint_jerk_rad_s3, names
    )
    jump_count = int(np.count_nonzero(jump_frame))
    summary = SmoothnessSummary(
        joint_jump_threshold_rad=thresholds.joint_jump_step_rad,
        joint_jump_rate=jump_count / (motion.num_frames - 1),
        joint_jump_count=jump_count,
        joint_step_p95_rad=_p95(derivatives.joint_step_abs_rad),
        joint_step_max_rad=_max(derivatives.joint_step_abs_rad),
        worst_step_joint=worst_step_joint,
        worst_step_frame=worst_step_frame,
        joint_vel_p95_rad_s=_p95(derivatives.joint_vel_rad_s),
        joint_vel_max_rad_s=_max(derivatives.joint_vel_rad_s),
        joint_acc_p95_rad_s2=_p95(derivatives.joint_acc_rad_s2),
        joint_acc_max_rad_s2=_max(derivatives.joint_acc_rad_s2),
        joint_jerk_p95_rad_s3=_p95(derivatives.joint_jerk_rad_s3),
        joint_jerk_max_rad_s3=_max(derivatives.joint_jerk_rad_s3),
        worst_velocity_joint=worst_velocity_joint,
        worst_velocity_frame=worst_velocity_frame,
        worst_jerk_joint=worst_jerk_joint,
        worst_jerk_frame=worst_jerk_frame,
        root_speed_p95_m_s=_p95(root_speed),
        root_speed_max_m_s=_max(root_speed),
        root_acc_p95_m_s2=_p95(root_acc),
        root_acc_max_m_s2=_max(root_acc),
        root_jerk_p95_m_s3=_p95(root_jerk),
        root_jerk_max_m_s3=_max(root_jerk),
        root_ang_vel_p95_rad_s=_p95(root_ang_vel),
        root_ang_vel_max_rad_s=_max(root_ang_vel),
    )
    return SmoothnessMetrics(
        derivatives=derivatives,
        joint_jump_frame=jump_frame,
        max_joint_step_rad=max_step,
        max_joint_step_joint_index=max_step_joint,
        max_joint_vel_rad_s=max_vel,
        max_joint_vel_joint_index=max_vel_joint,
        max_joint_acc_rad_s2=max_acc,
        max_joint_acc_joint_index=max_acc_joint,
        max_joint_jerk_rad_s3=max_jerk,
        max_joint_jerk_joint_index=max_jerk_joint,
        root_speed_m_s=root_speed,
        root_acc_m_s2=root_acc,
        root_jerk_m_s3=root_jerk,
        root_ang_vel_rad_s=root_ang_vel,
        summary=summary,
    )


def compute_smoothness_metrics(
    motion: MotionRecord,
    thresholds: KinematicThresholds | None = None,
) -> SmoothnessMetrics:
    resolved = thresholds or KinematicThresholds()
    derivatives = compute_derivative_series(motion)
    return _smoothness_from_derivatives(motion, derivatives, resolved)


def _velocity_limits(
    thresholds: KinematicThresholds,
    num_dofs: int,
) -> NDArray[np.float64]:
    raw = np.asarray(thresholds.velocity_limit_rad_s, dtype=np.float64)
    if raw.ndim == 0:
        limits = np.full(num_dofs, float(raw), dtype=np.float64)
    elif raw.shape == (num_dofs,):
        limits = np.array(raw, copy=True)
    else:
        raise ValueError(
            f"Per-joint velocity limits must have shape ({num_dofs},), got {raw.shape}"
        )
    return _readonly(limits)


def compute_joint_feasibility_metrics(
    motion: MotionRecord,
    joint_limits: JointLimits,
    thresholds: KinematicThresholds | None = None,
    derivatives: DerivativeSeries | None = None,
) -> JointFeasibilityMetrics:
    resolved = thresholds or KinematicThresholds()
    if joint_limits.num_dofs != motion.num_dofs:
        raise ValueError(
            f"Joint limits contain {joint_limits.num_dofs} DoFs, motion contains {motion.num_dofs}"
        )
    series = derivatives or compute_derivative_series(motion)
    if series.joint_vel_rad_s.shape != motion.joint_pos.shape:
        raise ValueError("Derivative joint velocity shape does not match the motion")

    q = np.asarray(motion.joint_pos, dtype=np.float64)
    lower = joint_limits.lower_rad
    upper = joint_limits.upper_rad
    tolerance = resolved.joint_limit_tolerance_rad
    position_violation = _readonly(
        (q < lower - tolerance) | (q > upper + tolerance)
    )
    position_violation_frame = _readonly(np.any(position_violation, axis=1))
    limit_excess = _readonly(np.maximum(np.maximum(lower - q, q - upper), 0.0))

    margin_rad = _readonly(np.minimum(q - lower, upper - q))
    ranges = upper - lower
    margin_norm = _readonly(
        np.minimum((q - lower) / ranges, (upper - q) / ranges)
    )
    min_margin_rad = _readonly(np.min(margin_rad, axis=1))
    min_margin_norm = _readonly(np.min(margin_norm, axis=1))
    nearest_limit_joint = _readonly(np.argmin(margin_rad, axis=1).astype(np.int64))

    near_abs = _readonly(margin_rad < resolved.near_limit_margin_rad)
    near_abs_frame = _readonly(np.any(near_abs, axis=1))
    near_norm = _readonly(margin_norm < resolved.near_limit_fraction)
    near_norm_frame = _readonly(np.any(near_norm, axis=1))

    velocity_limits = _velocity_limits(resolved, motion.num_dofs)
    velocity = np.asarray(series.joint_vel_rad_s, dtype=np.float64)
    valid_velocity_rows = np.isfinite(velocity).all(axis=1)
    velocity_violation = _readonly(
        np.isfinite(velocity) & (np.abs(velocity) > velocity_limits)
    )
    velocity_violation_frame = _readonly(np.any(velocity_violation, axis=1))
    velocity_excess = _readonly(
        np.where(
            np.isfinite(velocity),
            np.maximum(np.abs(velocity) - velocity_limits, 0.0),
            np.nan,
        )
    )

    frame_count = motion.num_frames
    element_count = frame_count * motion.num_dofs
    position_frame_count = int(np.count_nonzero(position_violation_frame))
    position_element_count = int(np.count_nonzero(position_violation))
    valid_velocity_frame_count = int(np.count_nonzero(valid_velocity_rows))
    valid_velocity_element_count = valid_velocity_frame_count * motion.num_dofs
    velocity_frame_count = int(np.count_nonzero(velocity_violation_frame))
    velocity_element_count = int(np.count_nonzero(velocity_violation))

    max_limit_excess = float(np.max(limit_excess))
    worst_limit_joint: str | None = None
    worst_limit_frame: int | None = None
    if max_limit_excess > 0:
        flat = int(np.argmax(limit_excess))
        worst_limit_frame, worst_limit_index = np.unravel_index(flat, limit_excess.shape)
        worst_limit_frame = int(worst_limit_frame)
        worst_limit_joint = joint_limits.joint_names[int(worst_limit_index)]

    finite_velocity_excess = velocity_excess[np.isfinite(velocity_excess)]
    max_velocity_excess = float(np.max(finite_velocity_excess))
    worst_velocity_joint: str | None = None
    worst_velocity_frame: int | None = None
    if max_velocity_excess > 0:
        safe_velocity_excess = np.where(
            np.isfinite(velocity_excess), velocity_excess, -np.inf
        )
        flat = int(np.argmax(safe_velocity_excess))
        worst_velocity_frame, worst_velocity_index = np.unravel_index(
            flat, velocity_excess.shape
        )
        worst_velocity_frame = int(worst_velocity_frame)
        worst_velocity_joint = joint_limits.joint_names[int(worst_velocity_index)]

    summary = JointFeasibilitySummary(
        joint_limit_tolerance_rad=tolerance,
        near_limit_margin_threshold_rad=resolved.near_limit_margin_rad,
        near_limit_fraction_threshold=resolved.near_limit_fraction,
        joint_limit_violation_frame_rate=position_frame_count / frame_count,
        joint_limit_violation_frame_count=position_frame_count,
        joint_limit_violation_element_rate=position_element_count / element_count,
        joint_limit_violation_element_count=position_element_count,
        max_limit_excess_rad=max_limit_excess,
        worst_limit_joint=worst_limit_joint,
        worst_limit_frame=worst_limit_frame,
        near_limit_0p05rad_frame_rate=float(np.mean(near_abs_frame)),
        near_limit_0p05rad_element_rate=float(np.mean(near_abs)),
        near_limit_2pct_frame_rate=float(np.mean(near_norm_frame)),
        near_limit_2pct_element_rate=float(np.mean(near_norm)),
        min_limit_margin_rad=float(np.min(margin_rad)),
        min_limit_margin_norm=float(np.min(margin_norm)),
        velocity_violation_frame_rate=(
            velocity_frame_count / valid_velocity_frame_count
        ),
        velocity_violation_frame_count=velocity_frame_count,
        velocity_violation_element_rate=(
            velocity_element_count / valid_velocity_element_count
        ),
        velocity_violation_element_count=velocity_element_count,
        max_velocity_excess_rad_s=max_velocity_excess,
        worst_velocity_limit_joint=worst_velocity_joint,
        worst_velocity_limit_frame=worst_velocity_frame,
    )
    return JointFeasibilityMetrics(
        position_violation=position_violation,
        position_violation_frame=position_violation_frame,
        limit_excess_rad=limit_excess,
        margin_rad=margin_rad,
        margin_norm=margin_norm,
        min_limit_margin_rad=min_margin_rad,
        min_limit_margin_norm=min_margin_norm,
        nearest_limit_joint_index=nearest_limit_joint,
        near_limit_0p05rad=near_abs,
        near_limit_0p05rad_frame=near_abs_frame,
        near_limit_2pct=near_norm,
        near_limit_2pct_frame=near_norm_frame,
        velocity_limit_rad_s=velocity_limits,
        velocity_violation=velocity_violation,
        velocity_violation_frame=velocity_violation_frame,
        velocity_excess_rad_s=velocity_excess,
        summary=summary,
    )


def compute_kinematic_metrics(
    motion: MotionRecord,
    joint_limits: JointLimits,
    thresholds: KinematicThresholds | None = None,
) -> KinematicMetrics:
    resolved = thresholds or KinematicThresholds()
    derivatives = compute_derivative_series(motion)
    smoothness = _smoothness_from_derivatives(motion, derivatives, resolved)
    feasibility = compute_joint_feasibility_metrics(
        motion,
        joint_limits,
        thresholds=resolved,
        derivatives=derivatives,
    )
    return KinematicMetrics(smoothness=smoothness, joint_feasibility=feasibility)
