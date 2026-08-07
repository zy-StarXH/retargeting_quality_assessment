from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class InteractionRecord:
    name: str
    task_label: str
    interaction_mode: str
    fps: float
    robot_qpos: NDArray[np.float64]
    left_hand_pos_m: NDArray[np.float64]
    right_hand_pos_m: NDArray[np.float64]
    left_hand_quat_xyzw: NDArray[np.float64]
    right_hand_quat_xyzw: NDArray[np.float64]
    object_pos_m: NDArray[np.float64]
    object_quat_xyzw: NDArray[np.float64]
    object_linvel_m_s: NDArray[np.float64]
    expected_contact: NDArray[np.bool_]
    observed_contact: NDArray[np.bool_]
    phase_id: NDArray[np.int64]
    phase_names: tuple[str, ...]
    target_pos_m: NDArray[np.float64]
    object_half_size_m: NDArray[np.float64]
    hand_proxy_radius_m: float
    source_path: Path

    @property
    def num_frames(self) -> int:
        return int(self.robot_qpos.shape[0])


@dataclass(frozen=True)
class InteractionFrameMetrics:
    """Frame-aligned hand/object metrics used by reports and event extraction."""

    signed_distance_m: NDArray[np.float64]
    separation_m: NDArray[np.float64]
    overlap_m: NDArray[np.float64]
    illegal_penetration_m: NDArray[np.float64]
    contact_miss: NDArray[np.bool_]
    unexpected_contact: NDArray[np.bool_]
    stable_bilateral_contact: NDArray[np.bool_]
    relative_position_error_m: NDArray[np.float64]
    relative_orientation_error_deg: NDArray[np.float64]
    relative_slip_speed_m_s: NDArray[np.float64]


@dataclass(frozen=True)
class InteractionEvent:
    motion_name: str
    event_type: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    entities: tuple[str, ...]
    max_severity: float
    details: str


def _scalar(data: Any, key: str) -> Any:
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be a scalar")
    return value.item()


def load_interaction_record(path: str | Path) -> InteractionRecord:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        required = {
            "schema_version", "name", "task_label", "interaction_mode", "fps",
            "robot_qpos", "left_hand_pos_m", "right_hand_pos_m",
            "left_hand_quat_xyzw", "right_hand_quat_xyzw", "object_pos_m",
            "object_quat_xyzw", "object_linvel_m_s", "expected_contact",
            "observed_contact", "phase_id", "phase_names", "target_pos_m",
            "object_half_size_m", "hand_proxy_radius_m",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Interaction record missing keys: {sorted(missing)}")
        if _scalar(data, "schema_version") not in {
            "gmr_interaction_v1",
            "gmr_interaction_v2",
        }:
            raise ValueError("Unsupported interaction schema")
        for key in (
            "robot_qpos", "left_hand_pos_m", "right_hand_pos_m",
            "left_hand_quat_xyzw", "right_hand_quat_xyzw", "object_pos_m",
            "object_quat_xyzw", "object_linvel_m_s", "target_pos_m",
            "object_half_size_m",
        ):
            if np.asarray(data[key]).dtype.kind not in "iuf":
                raise ValueError(f"{key} must use a real numeric dtype")
        for key in ("expected_contact", "observed_contact"):
            if np.asarray(data[key]).dtype.kind != "b":
                raise ValueError(f"{key} must use a boolean dtype")
        if np.asarray(data["phase_id"]).dtype.kind not in "iu":
            raise ValueError("phase_id must use an integer dtype")
        record = InteractionRecord(
            name=str(_scalar(data, "name")),
            task_label=str(_scalar(data, "task_label")),
            interaction_mode=str(_scalar(data, "interaction_mode")),
            fps=float(_scalar(data, "fps")),
            robot_qpos=np.asarray(data["robot_qpos"], dtype=np.float64),
            left_hand_pos_m=np.asarray(data["left_hand_pos_m"], dtype=np.float64),
            right_hand_pos_m=np.asarray(data["right_hand_pos_m"], dtype=np.float64),
            left_hand_quat_xyzw=np.asarray(data["left_hand_quat_xyzw"], dtype=np.float64),
            right_hand_quat_xyzw=np.asarray(data["right_hand_quat_xyzw"], dtype=np.float64),
            object_pos_m=np.asarray(data["object_pos_m"], dtype=np.float64),
            object_quat_xyzw=np.asarray(data["object_quat_xyzw"], dtype=np.float64),
            object_linvel_m_s=np.asarray(data["object_linvel_m_s"], dtype=np.float64),
            expected_contact=np.asarray(data["expected_contact"], dtype=np.bool_),
            observed_contact=np.asarray(data["observed_contact"], dtype=np.bool_),
            phase_id=np.asarray(data["phase_id"], dtype=np.int64),
            phase_names=tuple(str(value) for value in data["phase_names"].tolist()),
            target_pos_m=np.asarray(data["target_pos_m"], dtype=np.float64),
            object_half_size_m=np.asarray(data["object_half_size_m"], dtype=np.float64),
            hand_proxy_radius_m=float(_scalar(data, "hand_proxy_radius_m")),
            source_path=source,
        )
    validate_interaction_record(record)
    return record


def validate_interaction_record(record: InteractionRecord) -> None:
    if not np.isfinite(record.fps) or record.fps <= 0:
        raise ValueError("fps must be finite and positive")
    frames = record.num_frames
    expected_shapes = {
        "robot_qpos": (frames, 36),
        "left_hand_pos_m": (frames, 3),
        "right_hand_pos_m": (frames, 3),
        "left_hand_quat_xyzw": (frames, 4),
        "right_hand_quat_xyzw": (frames, 4),
        "object_pos_m": (frames, 3),
        "object_quat_xyzw": (frames, 4),
        "object_linvel_m_s": (frames, 3),
        "expected_contact": (frames, 2),
        "observed_contact": (frames, 2),
        "phase_id": (frames,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(record, name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype.kind not in "biu" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if frames < 4:
        raise ValueError("Interaction record requires at least four frames")
    for name in ("left_hand_quat_xyzw", "right_hand_quat_xyzw", "object_quat_xyzw"):
        norms = np.linalg.norm(getattr(record, name), axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-4:
            raise ValueError(f"{name} contains non-unit quaternions")
    root_quat_wxyz = record.robot_qpos[:, 3:7]
    root_norm_error = np.abs(np.linalg.norm(root_quat_wxyz, axis=1) - 1.0)
    if float(np.max(root_norm_error)) > 1e-4:
        raise ValueError("robot_qpos contains non-unit wxyz root quaternions")
    if record.target_pos_m.shape != (3,) or record.object_half_size_m.shape != (3,):
        raise ValueError("target_pos_m and object_half_size_m must have shape (3,)")
    if not np.isfinite(record.target_pos_m).all() or not np.isfinite(
        record.object_half_size_m
    ).all():
        raise ValueError("target_pos_m and object_half_size_m must be finite")
    if (
        np.any(record.object_half_size_m <= 0)
        or not np.isfinite(record.hand_proxy_radius_m)
        or record.hand_proxy_radius_m <= 0
    ):
        raise ValueError("Object half-size and hand radius must be positive")
    if not record.name or not record.task_label or not record.interaction_mode:
        raise ValueError("name, task_label, and interaction_mode must be non-empty")
    if not record.phase_names or any(not name for name in record.phase_names):
        raise ValueError("phase_names must contain non-empty names")
    if len(set(record.phase_names)) != len(record.phase_names):
        raise ValueError("phase_names must be unique")
    if np.any(record.phase_id < 0) or np.any(record.phase_id >= len(record.phase_names)):
        raise ValueError("phase_id references an unknown phase")


def _sphere_to_oriented_box_signed_distance(
    points_world: np.ndarray,
    box_pos_world: np.ndarray,
    box_quat_xyzw: np.ndarray,
    half_size: np.ndarray,
    sphere_radius: float,
) -> NDArray[np.float64]:
    rotation = Rotation.from_quat(box_quat_xyzw)
    local = rotation.inv().apply(points_world - box_pos_world)
    q = np.abs(local) - half_size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return np.asarray(outside + inside - sphere_radius, dtype=np.float64)


def _f1(expected: np.ndarray, observed: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(expected & observed))
    false_positive = int(np.count_nonzero(~expected & observed))
    false_negative = int(np.count_nonzero(expected & ~observed))
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def _orientation_std_deg(quat_xyzw: np.ndarray) -> float:
    if len(quat_xyzw) < 2:
        return 0.0
    reference = Rotation.from_quat(quat_xyzw[0])
    relative = reference.inv() * Rotation.from_quat(quat_xyzw)
    angles = np.linalg.norm(relative.as_rotvec(), axis=1)
    return float(np.degrees(np.std(angles)))


def _relative_hand_object_pose(
    hand_pos_m: np.ndarray,
    hand_quat_xyzw: np.ndarray,
    object_pos_m: np.ndarray,
    object_quat_xyzw: np.ndarray,
) -> tuple[NDArray[np.float64], Rotation]:
    hand_rotation = Rotation.from_quat(hand_quat_xyzw)
    object_rotation = Rotation.from_quat(object_quat_xyzw)
    relative_position = hand_rotation.inv().apply(object_pos_m - hand_pos_m)
    relative_rotation = hand_rotation.inv() * object_rotation
    return np.asarray(relative_position, dtype=np.float64), relative_rotation


def compute_interaction_frame_metrics(
    record: InteractionRecord,
    config: dict[str, Any],
) -> InteractionFrameMetrics:
    """Compute finite-or-NaN frame metrics without joining disjoint contact segments."""

    validate_interaction_record(record)
    metric_cfg = config["offline_metrics"]
    left_signed = _sphere_to_oriented_box_signed_distance(
        record.left_hand_pos_m,
        record.object_pos_m,
        record.object_quat_xyzw,
        record.object_half_size_m,
        record.hand_proxy_radius_m,
    )
    right_signed = _sphere_to_oriented_box_signed_distance(
        record.right_hand_pos_m,
        record.object_pos_m,
        record.object_quat_xyzw,
        record.object_half_size_m,
        record.hand_proxy_radius_m,
    )
    signed = np.column_stack([left_signed, right_signed])
    overlap = np.maximum(-signed, 0.0)
    allowance = float(metric_cfg.get("allowed_contact_overlap_m", 0.0))
    # Timing errors are scored separately.  A real observed contact still gets
    # the geometric proxy allowance even when it occurs outside the annotated
    # phase, otherwise the same event is double-counted as timing and penetration.
    contact_for_allowance = record.expected_contact | record.observed_contact
    allowed_overlap = allowance * contact_for_allowance.astype(np.float64)
    illegal_penetration = np.maximum(overlap - allowed_overlap, 0.0)
    separation = np.maximum(signed, 0.0)
    contact_miss = record.expected_contact & ~record.observed_contact
    unexpected_contact = ~record.expected_contact & record.observed_contact
    stable = record.expected_contact.all(axis=1) & record.observed_contact.all(axis=1)

    relative_positions = np.full((record.num_frames, 2, 3), np.nan, dtype=np.float64)
    relative_position_error = np.full((record.num_frames, 2), np.nan, dtype=np.float64)
    relative_orientation_error = np.full_like(relative_position_error, np.nan)
    relative_slip_speed = np.full_like(relative_position_error, np.nan)
    for side, (hand_pos, hand_quat) in enumerate(
        (
            (record.left_hand_pos_m, record.left_hand_quat_xyzw),
            (record.right_hand_pos_m, record.right_hand_quat_xyzw),
        )
    ):
        if bool(metric_cfg.get("hand_proxy_orientation_invariant", False)):
            # The bundled G1 rubber hands are spherical contact proxies. Their
            # link orientation does not define a grasp frame, so rotating the
            # proxy must not create a false object-pose drift.
            relative_pos = np.asarray(record.object_pos_m - hand_pos, dtype=np.float64)
            relative_rot = Rotation.from_quat(record.object_quat_xyzw)
        else:
            relative_pos, relative_rot = _relative_hand_object_pose(
                hand_pos,
                hand_quat,
                record.object_pos_m,
                record.object_quat_xyzw,
            )
        relative_positions[:, side] = relative_pos
        valid = record.expected_contact[:, side] & record.observed_contact[:, side]
        indices = np.flatnonzero(valid)
        if len(indices):
            reference_index = int(indices[0])
            relative_position_error[valid, side] = np.linalg.norm(
                relative_pos[valid] - relative_pos[reference_index], axis=1
            )
            reference_rotation = relative_rot[reference_index]
            orientation_delta = reference_rotation.inv() * relative_rot[valid]
            relative_orientation_error[valid, side] = np.degrees(
                np.linalg.norm(orientation_delta.as_rotvec(), axis=1)
            )
        consecutive = valid[1:] & valid[:-1]
        consecutive_indices = np.flatnonzero(consecutive) + 1
        if len(consecutive_indices):
            relative_slip_speed[consecutive_indices, side] = np.linalg.norm(
                relative_pos[consecutive_indices]
                - relative_pos[consecutive_indices - 1],
                axis=1,
            ) * record.fps

    return InteractionFrameMetrics(
        signed_distance_m=np.asarray(signed, dtype=np.float64),
        separation_m=np.asarray(separation, dtype=np.float64),
        overlap_m=np.asarray(overlap, dtype=np.float64),
        illegal_penetration_m=np.asarray(illegal_penetration, dtype=np.float64),
        contact_miss=np.asarray(contact_miss, dtype=np.bool_),
        unexpected_contact=np.asarray(unexpected_contact, dtype=np.bool_),
        stable_bilateral_contact=np.asarray(stable, dtype=np.bool_),
        relative_position_error_m=relative_position_error,
        relative_orientation_error_deg=relative_orientation_error,
        relative_slip_speed_m_s=relative_slip_speed,
    )


def build_interaction_frame_rows(
    record: InteractionRecord,
    config: dict[str, Any],
    frame_metrics: InteractionFrameMetrics | None = None,
) -> list[dict[str, Any]]:
    metrics = frame_metrics or compute_interaction_frame_metrics(record, config)
    thresholds = config["offline_metrics"]["event_thresholds"]
    rows: list[dict[str, Any]] = []
    for frame in range(record.num_frames):
        phase = record.phase_names[int(record.phase_id[frame])]
        row: dict[str, Any] = {
            "motion_name": record.name,
            "frame": frame,
            "time_s": frame / record.fps,
            "phase": phase,
            "object_x_m": record.object_pos_m[frame, 0],
            "object_y_m": record.object_pos_m[frame, 1],
            "object_z_m": record.object_pos_m[frame, 2],
            "object_speed_m_s": np.linalg.norm(record.object_linvel_m_s[frame]),
            "stable_bilateral_contact_flag": metrics.stable_bilateral_contact[frame],
        }
        for side_index, side in enumerate(("left", "right")):
            position_error = metrics.relative_position_error_m[frame, side_index]
            orientation_error = metrics.relative_orientation_error_deg[frame, side_index]
            slip = metrics.relative_slip_speed_m_s[frame, side_index]
            illegal = metrics.illegal_penetration_m[frame, side_index]
            row.update(
                {
                    f"{side}_expected_contact_flag": record.expected_contact[frame, side_index],
                    f"{side}_observed_contact_flag": record.observed_contact[frame, side_index],
                    f"{side}_contact_miss_flag": metrics.contact_miss[frame, side_index],
                    f"{side}_unexpected_contact_flag": metrics.unexpected_contact[frame, side_index],
                    f"{side}_signed_distance_m": metrics.signed_distance_m[frame, side_index],
                    f"{side}_separation_m": metrics.separation_m[frame, side_index],
                    f"{side}_overlap_m": metrics.overlap_m[frame, side_index],
                    f"{side}_illegal_penetration_m": illegal,
                    f"{side}_illegal_penetration_flag": illegal
                    > float(thresholds["illegal_penetration_m"]),
                    f"{side}_relative_position_error_m": position_error,
                    f"{side}_relative_position_drift_flag": np.isfinite(position_error)
                    and position_error > float(thresholds["relative_position_error_m"]),
                    f"{side}_relative_orientation_error_deg": orientation_error,
                    f"{side}_relative_orientation_drift_flag": np.isfinite(orientation_error)
                    and orientation_error
                    > float(thresholds["relative_orientation_error_deg"]),
                    f"{side}_relative_slip_speed_m_s": slip,
                    f"{side}_grasp_slip_flag": np.isfinite(slip)
                    and slip > float(thresholds["slip_speed_m_s"]),
                }
            )
        rows.append(row)
    return rows


def _segments(flags: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(np.asarray(flags, dtype=bool))
    if not len(indices):
        return []
    starts = [int(indices[0])]
    ends: list[int] = []
    for previous, current in zip(indices[:-1], indices[1:]):
        if int(current) != int(previous) + 1:
            ends.append(int(previous))
            starts.append(int(current))
    ends.append(int(indices[-1]))
    return list(zip(starts, ends))


def extract_interaction_events(
    record: InteractionRecord,
    config: dict[str, Any],
    frame_metrics: InteractionFrameMetrics | None = None,
    summary: dict[str, Any] | None = None,
) -> tuple[InteractionEvent, ...]:
    metrics = frame_metrics or compute_interaction_frame_metrics(record, config)
    thresholds = config["offline_metrics"]["event_thresholds"]
    definitions: list[tuple[str, int, np.ndarray, np.ndarray, str]] = []
    for side_index, side in enumerate(("left_hand", "right_hand")):
        definitions.extend(
            [
                (
                    "EXPECTED_CONTACT_MISSED",
                    side_index,
                    metrics.contact_miss[:, side_index],
                    metrics.separation_m[:, side_index],
                    side,
                ),
                (
                    "UNEXPECTED_CONTACT",
                    side_index,
                    metrics.unexpected_contact[:, side_index],
                    metrics.overlap_m[:, side_index],
                    side,
                ),
                (
                    "ILLEGAL_HAND_OBJECT_PENETRATION",
                    side_index,
                    metrics.illegal_penetration_m[:, side_index]
                    > float(thresholds["illegal_penetration_m"]),
                    metrics.illegal_penetration_m[:, side_index],
                    side,
                ),
                (
                    "RELATIVE_POSITION_DRIFT",
                    side_index,
                    metrics.relative_position_error_m[:, side_index]
                    > float(thresholds["relative_position_error_m"]),
                    metrics.relative_position_error_m[:, side_index],
                    side,
                ),
                (
                    "RELATIVE_ORIENTATION_DRIFT",
                    side_index,
                    metrics.relative_orientation_error_deg[:, side_index]
                    > float(thresholds["relative_orientation_error_deg"]),
                    metrics.relative_orientation_error_deg[:, side_index],
                    side,
                ),
                (
                    "GRASP_SLIP",
                    side_index,
                    metrics.relative_slip_speed_m_s[:, side_index]
                    > float(thresholds["slip_speed_m_s"]),
                    metrics.relative_slip_speed_m_s[:, side_index],
                    side,
                ),
            ]
        )
    events: list[InteractionEvent] = []
    for event_type, _, flags, severity, entity in definitions:
        for start, end in _segments(flags):
            scoped = np.asarray(severity[start : end + 1], dtype=float)
            finite = scoped[np.isfinite(scoped)]
            events.append(
                InteractionEvent(
                    motion_name=record.name,
                    event_type=event_type,
                    start_frame=start,
                    end_frame=end,
                    start_time_s=start / record.fps,
                    end_time_s=end / record.fps,
                    duration_s=(end - start + 1) / record.fps,
                    entities=(entity,),
                    max_severity=float(np.max(finite)) if len(finite) else 1.0,
                    details=f"{event_type} on {entity}",
                )
            )
    result_summary = summary or compute_interaction_metrics(record, config)
    if not bool(result_summary["nominal_task_success"]):
        frame = record.num_frames - 1
        events.append(
            InteractionEvent(
                motion_name=record.name,
                event_type="TASK_NOT_COMPLETED",
                start_frame=frame,
                end_frame=frame,
                start_time_s=frame / record.fps,
                end_time_s=frame / record.fps,
                duration_s=1.0 / record.fps,
                entities=("manipulated_object",),
                max_severity=float(result_summary["final_object_position_error_m"]),
                details="Final object state does not satisfy the frozen task-success criteria",
            )
        )
    return tuple(
        sorted(events, key=lambda event: (event.start_frame, event.event_type, event.entities))
    )


def compute_interaction_metrics(record: InteractionRecord, config: dict[str, Any]) -> dict[str, Any]:
    validate_interaction_record(record)
    metric_cfg = config["offline_metrics"]
    task_cfg = config["task"]
    frames = compute_interaction_frame_metrics(record, config)
    signed = frames.signed_distance_m
    expected = record.expected_contact
    observed = record.observed_contact
    expected_values = signed[expected]
    separation = np.maximum(expected_values, 0.0)
    mean_separation = float(np.mean(separation)) if len(separation) else float("nan")
    max_penetration = float(np.max(frames.overlap_m))
    max_illegal_penetration = float(np.max(frames.illegal_penetration_m))
    severe_penetration = frames.illegal_penetration_m > float(
        metric_cfg["severe_penetration_m"]
    )
    severe_penetration_rate = float(np.mean(severe_penetration))
    timing_f1 = float(np.mean([_f1(expected[:, side], observed[:, side]) for side in range(2)]))

    stable_mask = frames.stable_bilateral_contact
    if bool(metric_cfg.get("relative_pose_requires_bilateral_contact", False)):
        stable_samples = np.repeat(stable_mask[:, None], 2, axis=1)
    else:
        stable_samples = np.ones((record.num_frames, 2), dtype=bool)
    position_values = frames.relative_position_error_m[
        stable_samples & np.isfinite(frames.relative_position_error_m)
    ]
    orientation_values = frames.relative_orientation_error_deg[
        stable_samples & np.isfinite(frames.relative_orientation_error_deg)
    ]
    slip_values = frames.relative_slip_speed_m_s[
        stable_samples & np.isfinite(frames.relative_slip_speed_m_s)
    ]
    relative_position_std = (
        float(np.std(position_values)) if len(position_values) >= 2 else float("nan")
    )
    relative_orientation_std = (
        float(np.std(orientation_values))
        if len(orientation_values) >= 2
        else float("nan")
    )
    slip_p95 = float(np.percentile(slip_values, 95)) if len(slip_values) else float("nan")

    expected_both_count = max(int(np.count_nonzero(expected.all(axis=1))), 1)
    bilateral_contact_rate = float(np.count_nonzero(stable_mask) / expected_both_count)
    initial_z = float(record.object_pos_m[0, 2])
    max_lift = float(np.max(record.object_pos_m[:, 2]) - initial_z)
    final_error_axis = np.abs(record.object_pos_m[-1] - record.target_pos_m)
    final_error = float(np.linalg.norm(final_error_axis))
    final_speed = float(np.linalg.norm(record.object_linvel_m_s[-1]))
    tolerance = np.asarray(task_cfg["success_position_tolerance_m"], dtype=float)
    nominal_task_success = bool(
        np.all(final_error_axis <= tolerance)
        and final_speed <= float(task_cfg["success_max_linear_speed_m_s"])
        and max_lift >= float(task_cfg["success_min_lift_m"])
    )

    proximity_available = bool(len(expected_values))
    proximity_score = (
        100.0
        * np.clip(
            1.0
            - mean_separation / float(metric_cfg["contact_distance_threshold_m"]),
            0.0,
            1.0,
        )
        if proximity_available
        else float("nan")
    )
    timing_score = 100.0 * timing_f1
    position_term = (
        np.clip(
            1.0
            - relative_position_std
            / float(metric_cfg["stable_relative_position_std_m"]),
            0.0,
            1.0,
        )
        if np.isfinite(relative_position_std)
        else 0.0
    )
    orientation_term = (
        np.clip(
            1.0
            - relative_orientation_std
            / float(metric_cfg["stable_relative_orientation_std_deg"]),
            0.0,
            1.0,
        )
        if np.isfinite(relative_orientation_std)
        else 0.0
    )
    relative_pose_score = 100.0 * position_term * orientation_term
    penetration_score = 100.0 * np.clip(1.0 - severe_penetration_rate / 0.05, 0.0, 1.0)
    slip_term = (
        np.clip(
            1.0 - slip_p95 / float(metric_cfg["stable_slip_speed_m_s"]),
            0.0,
            1.0,
        )
        if np.isfinite(slip_p95)
        else 0.0
    )
    lift_term = np.clip(max_lift / float(task_cfg["success_min_lift_m"]), 0.0, 1.0)
    grasp_stability_score = 100.0 * bilateral_contact_rate * slip_term * lift_term
    goal_term = np.clip(1.0 - np.linalg.norm(final_error_axis / tolerance) / np.sqrt(3.0), 0.0, 1.0)
    speed_term = np.clip(1.0 - final_speed / float(task_cfg["success_max_linear_speed_m_s"]), 0.0, 1.0)
    task_effectiveness_score = 100.0 * goal_term * speed_term * lift_term
    dimensions = {
        "hand_object_proximity": float(proximity_score),
        "contact_timing": float(timing_score),
        "relative_pose": float(relative_pose_score),
        "penetration": float(penetration_score),
        "grasp_stability": float(grasp_stability_score),
        "task_effectiveness": float(task_effectiveness_score),
    }
    weights = metric_cfg["weights"]
    if set(weights) != set(dimensions) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Interaction score weights must match dimensions and sum to one")
    available_dimensions = tuple(
        name for name, value in dimensions.items() if np.isfinite(value)
    )
    available_weight = sum(float(weights[name]) for name in available_dimensions)
    if available_weight <= 0:
        raise ValueError("No interaction score dimension is available")
    normalized_weights = {
        name: float(weights[name]) / available_weight for name in available_dimensions
    }
    score = float(
        sum(normalized_weights[name] * dimensions[name] for name in available_dimensions)
    )
    failure_codes: list[str] = []
    if max_illegal_penetration > float(metric_cfg["severe_penetration_m"]):
        failure_codes.append("ILLEGAL_HAND_OBJECT_PENETRATION")
    if not nominal_task_success:
        failure_codes.append("TASK_NOT_COMPLETED")
    return {
        "motion_name": record.name,
        "task_label": record.task_label,
        "interaction_mode": record.interaction_mode,
        "num_frames": record.num_frames,
        "duration_s": record.num_frames / record.fps,
        "mean_expected_hand_object_separation_m": mean_separation,
        "contact_timing_f1": timing_f1,
        "bilateral_contact_rate": bilateral_contact_rate,
        "relative_position_std_m": relative_position_std,
        "relative_orientation_std_deg": relative_orientation_std,
        "relative_slip_speed_p95_m_s": slip_p95,
        "max_penetration_m": max_penetration,
        "max_illegal_penetration_m": max_illegal_penetration,
        "allowed_contact_overlap_m": float(metric_cfg.get("allowed_contact_overlap_m", 0.0)),
        "severe_penetration_frame_rate": severe_penetration_rate,
        "contact_miss_frame_rate": float(np.mean(np.any(frames.contact_miss, axis=1))),
        "unexpected_contact_frame_rate": float(
            np.mean(np.any(frames.unexpected_contact, axis=1))
        ),
        "max_object_lift_m": max_lift,
        "final_object_position_error_m": final_error,
        "final_object_speed_m_s": final_speed,
        "nominal_task_success": nominal_task_success,
        **{f"{name}_score": value for name, value in dimensions.items()},
        "interaction_quality_score": score,
        "score_available_dimensions": available_dimensions,
        "normalized_score_weights": normalized_weights,
        "proximity_metric_available": proximity_available,
        "relative_pose_metric_available": bool(len(position_values) >= 2),
        "grasp_slip_metric_available": bool(len(slip_values)),
        "score_version": "interaction_v2",
        "score_is_calibrated": False,
        "hard_fail": bool(failure_codes),
        "failure_reason_codes": tuple(failure_codes),
    }
