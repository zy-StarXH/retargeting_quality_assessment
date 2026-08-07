from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contact_metrics import BodyPair, ContactMetrics
from .kinematic_metrics import KinematicMetrics
from .types import MotionRecord


DetailValue = str | int | float | bool


@dataclass(frozen=True)
class QualityEvent:
    motion_name: str
    event_type: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    entities: tuple[str, ...]
    max_severity: float
    details: tuple[tuple[str, DetailValue], ...]

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def details_dict(self) -> dict[str, DetailValue]:
        return dict(self.details)


def _runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.bool_)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return zip(starts.tolist(), ends.tolist())


def _event(
    motion_name: str,
    event_type: str,
    start: int,
    end: int,
    fps: float,
    entities: Iterable[str],
    max_severity: float,
    severity_metric: str,
    severity_unit: str,
    *,
    threshold: float | None = None,
) -> QualityEvent:
    details: list[tuple[str, DetailValue]] = [
        ("frame_count", end - start + 1),
        ("severity_metric", severity_metric),
        ("severity_unit", severity_unit),
    ]
    if threshold is not None:
        details.append(("threshold", float(threshold)))
    return QualityEvent(
        motion_name=motion_name,
        event_type=event_type,
        start_frame=start,
        end_frame=end,
        start_time_s=start / fps,
        end_time_s=end / fps,
        duration_s=(end - start + 1) / fps,
        entities=tuple(sorted(set(entities))),
        max_severity=float(max_severity),
        details=tuple(details),
    )


def _joint_events(
    events: list[QualityEvent],
    motion_name: str,
    event_type: str,
    flags: np.ndarray,
    severity: np.ndarray,
    joint_names: tuple[str, ...],
    fps: float,
    severity_metric: str,
    severity_unit: str,
    threshold: float | np.ndarray,
) -> None:
    threshold_values = np.asarray(threshold, dtype=np.float64)
    if threshold_values.ndim == 0:
        thresholds = np.full(len(joint_names), float(threshold_values))
    elif threshold_values.shape == (len(joint_names),):
        thresholds = threshold_values
    else:
        raise ValueError("Joint event threshold shape does not match joint names")
    for joint_index, joint_name in enumerate(joint_names):
        for start, end in _runs(flags[:, joint_index]):
            values = severity[start : end + 1, joint_index]
            finite = values[np.isfinite(values)]
            max_severity = float(np.max(finite)) if finite.size else 0.0
            events.append(
                _event(
                    motion_name,
                    event_type,
                    start,
                    end,
                    fps,
                    (joint_name,),
                    max_severity,
                    severity_metric,
                    severity_unit,
                    threshold=float(thresholds[joint_index]),
                )
            )


def _pair_events(
    events: list[QualityEvent],
    motion_name: str,
    event_type: str,
    pairs_by_frame: tuple[tuple[BodyPair, ...], ...],
    pair_depths_by_frame: tuple[tuple[tuple[BodyPair, float], ...], ...],
    fps: float,
) -> None:
    unique_pairs = sorted({pair for frame in pairs_by_frame for pair in frame})
    depth_maps = tuple(dict(values) for values in pair_depths_by_frame)
    for pair in unique_pairs:
        mask = np.asarray([pair in frame for frame in pairs_by_frame])
        depths = np.asarray([values.get(pair, 0.0) for values in depth_maps])
        for start, end in _runs(mask):
            events.append(
                _event(
                    motion_name,
                    event_type,
                    start,
                    end,
                    fps,
                    pair,
                    float(np.max(depths[start : end + 1])),
                    "contact_penetration_depth",
                    "m",
                )
            )


def _mask_events(
    events: list[QualityEvent],
    motion_name: str,
    event_type: str,
    mask: np.ndarray,
    severity: np.ndarray,
    entities_by_frame: tuple[tuple[str, ...], ...],
    fps: float,
    severity_metric: str,
    severity_unit: str,
    threshold: float | None = None,
) -> None:
    for start, end in _runs(mask):
        entities = {
            entity
            for frame_entities in entities_by_frame[start : end + 1]
            for entity in frame_entities
        }
        values = severity[start : end + 1]
        finite = values[np.isfinite(values)]
        max_severity = float(np.max(finite)) if finite.size else 0.0
        events.append(
            _event(
                motion_name,
                event_type,
                start,
                end,
                fps,
                entities,
                max_severity,
                severity_metric,
                severity_unit,
                threshold=threshold,
            )
        )


def _body_events(
    events: list[QualityEvent],
    motion_name: str,
    event_type: str,
    bodies_by_frame: tuple[tuple[str, ...], ...],
    body_depths_by_frame: tuple[tuple[tuple[str, float], ...], ...],
    fps: float,
) -> None:
    unique_bodies = sorted({body for frame in bodies_by_frame for body in frame})
    depth_maps = tuple(dict(values) for values in body_depths_by_frame)
    for body in unique_bodies:
        mask = np.asarray([body in frame for frame in bodies_by_frame])
        depths = np.asarray([values.get(body, 0.0) for values in depth_maps])
        for start, end in _runs(mask):
            events.append(
                _event(
                    motion_name,
                    event_type,
                    start,
                    end,
                    fps,
                    (body,),
                    float(np.max(depths[start : end + 1])),
                    "contact_penetration_depth",
                    "m",
                )
            )


def extract_quality_events(
    motion_name: str,
    motion: MotionRecord,
    kinematic: KinematicMetrics,
    contact: ContactMetrics,
) -> tuple[QualityEvent, ...]:
    """Convert aligned metric flags into inclusive, entity-aware event intervals."""
    if not motion_name:
        raise ValueError("motion_name must be non-empty")
    frames = motion.num_frames
    arrays = (
        kinematic.smoothness.joint_jump_frame,
        kinematic.joint_feasibility.position_violation_frame,
        contact.self_collision_any_frame,
        contact.ground_penetration_frame,
    )
    if any(array.shape != (frames,) for array in arrays):
        raise ValueError("Metric frame arrays do not align with the motion")
    joint_names = motion.joint_names or tuple(
        f"joint_{index}" for index in range(motion.num_dofs)
    )
    events: list[QualityEvent] = []
    smooth = kinematic.smoothness
    feasibility = kinematic.joint_feasibility
    derivatives = smooth.derivatives

    jump_flags = np.isfinite(derivatives.joint_step_abs_rad) & (
        derivatives.joint_step_abs_rad > smooth.summary.joint_jump_threshold_rad
    )
    _joint_events(
        events,
        motion_name,
        "JOINT_JUMP",
        jump_flags,
        derivatives.joint_step_abs_rad,
        joint_names,
        motion.fps,
        "joint_step",
        "rad",
        smooth.summary.joint_jump_threshold_rad,
    )
    _joint_events(
        events,
        motion_name,
        "VELOCITY_LIMIT_EXCEEDED",
        feasibility.velocity_violation,
        np.abs(derivatives.joint_vel_rad_s),
        joint_names,
        motion.fps,
        "absolute_joint_velocity",
        "rad_s",
        feasibility.velocity_limit_rad_s,
    )
    _joint_events(
        events,
        motion_name,
        "JOINT_LIMIT_EXCEEDED",
        feasibility.position_violation,
        feasibility.limit_excess_rad,
        joint_names,
        motion.fps,
        "joint_limit_excess",
        "rad",
        0.0,
    )
    near_severity = np.maximum(
        feasibility.summary.near_limit_fraction_threshold - feasibility.margin_norm,
        0.0,
    )
    _joint_events(
        events,
        motion_name,
        "NEAR_JOINT_LIMIT_2PCT",
        feasibility.near_limit_2pct,
        near_severity,
        joint_names,
        motion.fps,
        "normalized_margin_shortfall",
        "fraction",
        feasibility.summary.near_limit_fraction_threshold,
    )

    _pair_events(
        events,
        motion_name,
        "SELF_COLLISION_ANY",
        contact.self_collision_pairs,
        contact.self_collision_pair_penetration_m,
        motion.fps,
    )
    _pair_events(
        events,
        motion_name,
        "SELF_COLLISION_NONHAND",
        contact.self_collision_nonhand_pairs,
        contact.self_collision_pair_penetration_m,
        motion.fps,
    )
    _mask_events(
        events,
        motion_name,
        "GROUND_PENETRATION",
        contact.ground_penetration_frame,
        contact.ground_penetration_depth_m,
        contact.ground_penetration_bodies,
        motion.fps,
        "ground_penetration_depth",
        "m",
        contact.summary.penetration_threshold_m,
    )
    _body_events(
        events,
        motion_name,
        "UNDESIRED_GROUND_CONTACT",
        contact.undesired_ground_contact_bodies,
        contact.ground_body_penetration_m,
        motion.fps,
    )

    slide_entities = tuple(
        tuple(
            side
            for side, active in (
                ("left_foot", bool(contact.left_foot_slide_frame[frame])),
                ("right_foot", bool(contact.right_foot_slide_frame[frame])),
            )
            if active
        )
        for frame in range(frames)
    )
    slide_speeds = np.column_stack(
        (
            contact.left_foot_horizontal_speed_m_s,
            contact.right_foot_horizontal_speed_m_s,
        )
    )
    slide_severity = np.max(
        np.where(np.isfinite(slide_speeds), slide_speeds, -np.inf), axis=1
    )
    slide_severity[~np.isfinite(slide_severity)] = np.nan
    _mask_events(
        events,
        motion_name,
        "FOOT_SLIDE",
        contact.foot_slide_frame,
        slide_severity,
        slide_entities,
        motion.fps,
        "foot_horizontal_speed",
        "m_s",
        contact.summary.foot_slide_speed_threshold_m_s,
    )

    for start, end in _runs(contact.airborne_frame):
        duration_s = (end - start + 1) / motion.fps
        events.append(
            _event(
                motion_name,
                "AIRBORNE_PROXY",
                start,
                end,
                motion.fps,
                (),
                duration_s,
                "duration",
                "s",
            )
        )
    return tuple(
        sorted(
            events,
            key=lambda item: (
                item.start_frame,
                item.event_type,
                item.entities,
                item.end_frame,
            ),
        )
    )
