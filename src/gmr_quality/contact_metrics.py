from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .model_info import BoundMotion, RobotModelInfo
from .mujoco_kinematics import ForwardKinematicsFrame, MujocoKinematics
from .types import MotionRecord


BodyPair = tuple[str, str]
BodyPairDepth = tuple[BodyPair, float]


@dataclass(frozen=True)
class ContactThresholds:
    penetration_m: float = 0.01
    severe_penetration_m: float = 0.05
    foot_near_ground_m: float = 0.01
    foot_vertical_speed_contact_m_s: float = 0.2
    foot_slide_speed_m_s: float = 0.1

    def __post_init__(self) -> None:
        values = {
            "penetration_m": self.penetration_m,
            "severe_penetration_m": self.severe_penetration_m,
            "foot_near_ground_m": self.foot_near_ground_m,
            "foot_vertical_speed_contact_m_s": self.foot_vertical_speed_contact_m_s,
            "foot_slide_speed_m_s": self.foot_slide_speed_m_s,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.severe_penetration_m < self.penetration_m:
            raise ValueError("severe_penetration_m must be at least penetration_m")

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "ContactThresholds":
        thresholds = config["thresholds"]
        return cls(
            penetration_m=float(thresholds["penetration_m"]),
            severe_penetration_m=float(thresholds["severe_penetration_m"]),
            foot_near_ground_m=float(thresholds["foot_near_ground_m"]),
            foot_vertical_speed_contact_m_s=float(
                thresholds["foot_vertical_speed_contact_m_s"]
            ),
            foot_slide_speed_m_s=float(thresholds["foot_slide_speed_m_s"]),
        )


@dataclass(frozen=True)
class ContactPolicy:
    name: str
    undesired_ground_contact_is_failure: bool
    airborne_is_failure: bool

    @classmethod
    def from_project_config(
        cls,
        config: dict[str, Any],
        policy_name: str,
    ) -> "ContactPolicy":
        policies = config["contact_policies"]
        if policy_name not in policies:
            available = ", ".join(sorted(policies))
            raise ValueError(
                f"Unknown contact policy {policy_name!r}; available: {available}"
            )
        raw = policies[policy_name]
        undesired_is_failure = raw["undesired_ground_contact_is_failure"]
        airborne_is_failure = raw["airborne_is_failure"]
        if not isinstance(undesired_is_failure, bool) or not isinstance(
            airborne_is_failure, bool
        ):
            raise ValueError(f"Contact policy {policy_name!r} flags must be booleans")
        return cls(
            name=policy_name,
            undesired_ground_contact_is_failure=undesired_is_failure,
            airborne_is_failure=airborne_is_failure,
        )


@dataclass(frozen=True)
class ContactEvaluationConfig:
    thresholds: ContactThresholds
    policy: ContactPolicy
    allowed_hand_contact_patterns: tuple[str, ...]

    def __post_init__(self) -> None:
        patterns = self.allowed_hand_contact_patterns
        if (
            not patterns
            or not all(isinstance(pattern, str) and pattern for pattern in patterns)
            or len(set(patterns)) != len(patterns)
        ):
            raise ValueError(
                "allowed_hand_contact_patterns must contain unique non-empty strings"
            )

    @classmethod
    def from_project_config(
        cls,
        config: dict[str, Any],
        contact_policy: str,
    ) -> "ContactEvaluationConfig":
        return cls(
            thresholds=ContactThresholds.from_project_config(config),
            policy=ContactPolicy.from_project_config(config, contact_policy),
            allowed_hand_contact_patterns=tuple(
                str(value)
                for value in config["robot"]["allowed_hand_contact_patterns"]
            ),
        )


@dataclass(frozen=True)
class ContactMetricsSummary:
    motion_category: str
    contact_policy: str
    undesired_ground_contact_is_failure: bool
    airborne_is_failure: bool
    penetration_threshold_m: float
    severe_penetration_threshold_m: float
    foot_near_ground_threshold_m: float
    foot_vertical_speed_contact_threshold_m_s: float
    foot_slide_speed_threshold_m_s: float
    raw_contact_count: int
    self_collision_contact_count: int
    self_collision_any_frame_count: int
    self_collision_any_frame_rate: float
    self_collision_nonhand_frame_count: int
    self_collision_nonhand_frame_rate: float
    self_collision_event_count: int
    self_collision_nonhand_event_count: int
    max_self_contacts_per_frame: int
    first_self_collision_frame: int | None
    self_collision_body_pairs: tuple[BodyPair, ...]
    self_collision_nonhand_body_pairs: tuple[BodyPair, ...]
    max_self_collision_penetration_m: float
    worst_self_collision_body_pair: BodyPair | None
    worst_self_collision_frame: int | None
    ground_contact_sample_count: int
    ground_contact_frame_count: int
    ground_contact_frame_rate: float
    ground_penetration_frame_count: int
    ground_penetration_frame_rate: float
    ground_penetration_event_count: int
    severe_ground_penetration_frame_count: int
    severe_ground_penetration_frame_rate: float
    mean_ground_penetration_m: float
    max_ground_penetration_m: float
    worst_penetration_body: str | None
    worst_penetration_frame: int | None
    undesired_ground_contact_frame_count: int
    undesired_ground_contact_frame_rate: float
    undesired_ground_contact_event_count: int
    undesired_ground_contact_bodies: tuple[str, ...]
    left_foot_height_min_m: float
    left_foot_height_p05_m: float
    left_foot_height_p95_m: float
    left_foot_height_max_m: float
    right_foot_height_min_m: float
    right_foot_height_p05_m: float
    right_foot_height_p95_m: float
    right_foot_height_max_m: float
    left_contact_proxy_rate: float
    right_contact_proxy_rate: float
    left_mujoco_ground_contact_frame_rate: float
    right_mujoco_ground_contact_frame_rate: float
    left_contact_proxy_mismatch_frame_count: int
    left_contact_proxy_mismatch_frame_rate: float
    right_contact_proxy_mismatch_frame_count: int
    right_contact_proxy_mismatch_frame_rate: float
    foot_slide_frame_count: int
    foot_slide_frame_rate: float
    left_foot_slide_frame_count: int
    left_foot_slide_frame_rate: float
    right_foot_slide_frame_count: int
    right_foot_slide_frame_rate: float
    foot_slide_speed_p95_m_s: float
    foot_slide_speed_max_m_s: float
    foot_slide_event_count: int
    airborne_frame_count: int
    airborne_frame_rate: float
    max_airborne_duration_s: float


@dataclass(frozen=True)
class ContactMetrics:
    self_collision_any_frame: NDArray[np.bool_]
    self_collision_nonhand_frame: NDArray[np.bool_]
    self_contacts_per_frame: NDArray[np.int64]
    self_collision_max_penetration_m: NDArray[np.float64]
    self_collision_pairs: tuple[tuple[BodyPair, ...], ...]
    self_collision_nonhand_pairs: tuple[tuple[BodyPair, ...], ...]
    self_collision_pair_penetration_m: tuple[tuple[BodyPairDepth, ...], ...]
    ground_contact_frame: NDArray[np.bool_]
    ground_contact_bodies: tuple[tuple[str, ...], ...]
    ground_penetration_frame: NDArray[np.bool_]
    severe_ground_penetration_frame: NDArray[np.bool_]
    ground_penetration_depth_m: NDArray[np.float64]
    ground_penetration_bodies: tuple[tuple[str, ...], ...]
    ground_body_penetration_m: tuple[tuple[tuple[str, float], ...], ...]
    undesired_ground_contact_frame: NDArray[np.bool_]
    undesired_ground_contact_bodies: tuple[tuple[str, ...], ...]
    left_sole_points_world_m: NDArray[np.float64]
    right_sole_points_world_m: NDArray[np.float64]
    left_foot_center_world_m: NDArray[np.float64]
    right_foot_center_world_m: NDArray[np.float64]
    left_foot_height_m: NDArray[np.float64]
    right_foot_height_m: NDArray[np.float64]
    left_foot_horizontal_speed_m_s: NDArray[np.float64]
    right_foot_horizontal_speed_m_s: NDArray[np.float64]
    left_foot_vertical_speed_m_s: NDArray[np.float64]
    right_foot_vertical_speed_m_s: NDArray[np.float64]
    left_contact_proxy_frame: NDArray[np.bool_]
    right_contact_proxy_frame: NDArray[np.bool_]
    left_mujoco_ground_contact_frame: NDArray[np.bool_]
    right_mujoco_ground_contact_frame: NDArray[np.bool_]
    left_contact_proxy_mismatch_frame: NDArray[np.bool_]
    right_contact_proxy_mismatch_frame: NDArray[np.bool_]
    left_foot_slide_frame: NDArray[np.bool_]
    right_foot_slide_frame: NDArray[np.bool_]
    foot_slide_frame: NDArray[np.bool_]
    airborne_frame: NDArray[np.bool_]
    warnings: tuple[str, ...]
    summary: ContactMetricsSummary


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _pair(body1: str, body2: str) -> BodyPair:
    return (body1, body2) if body1 <= body2 else (body2, body1)


def _is_allowed_hand_pair(pair: BodyPair, patterns: tuple[str, ...]) -> bool:
    return any(pattern in body for body in pair for pattern in patterns)


def _count_runs(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=np.bool_)
    if values.size == 0:
        return 0
    return int(values[0]) + int(np.count_nonzero(values[1:] & ~values[:-1]))


def _max_run_length(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=np.bool_):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _entity_event_count(per_frame: list[set[Any]]) -> int:
    entities = set().union(*per_frame) if per_frame else set()
    return sum(
        _count_runs(np.asarray([entity in frame for frame in per_frame]))
        for entity in entities
    )


def _first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _foot_geometry(
    geom_positions: np.ndarray,
    geom_ids: tuple[int, ...],
    model_info: RobotModelInfo,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.array(geom_positions[np.asarray(geom_ids)], dtype=np.float64, copy=True)
    radii = np.asarray([model_info.geom_sizes[geom_id][0] for geom_id in geom_ids])
    points[:, 2] -= radii
    center = np.mean(points, axis=0)
    height = float(np.min(points[:, 2]))
    return points, center, height


def compute_contact_metrics_from_frames(
    motion: MotionRecord,
    model_info: RobotModelInfo,
    frames: Iterable[ForwardKinematicsFrame],
    evaluation_config: ContactEvaluationConfig,
    *,
    motion_category: str = "unknown",
) -> ContactMetrics:
    """Compute geometry/contact metrics from aligned, state-isolated FK frames."""
    if not isinstance(motion_category, str) or not motion_category:
        raise ValueError("motion_category must be a non-empty string")
    frame_count = motion.num_frames
    left_ids = model_info.left_sole_geom_ids
    right_ids = model_info.right_sole_geom_ids
    left_id_set = set(left_ids)
    right_id_set = set(right_ids)
    left_body_ids = {model_info.geom_body_ids[geom_id] for geom_id in left_ids}
    right_body_ids = {model_info.geom_body_ids[geom_id] for geom_id in right_ids}
    if (
        len(left_body_ids) != 1
        or len(right_body_ids) != 1
        or left_body_ids == right_body_ids
    ):
        raise ValueError("Configured sole geoms must belong to two distinct foot bodies")
    foot_body_ids = left_body_ids | right_body_ids

    self_any = np.zeros(frame_count, dtype=np.bool_)
    self_nonhand = np.zeros(frame_count, dtype=np.bool_)
    self_count = np.zeros(frame_count, dtype=np.int64)
    self_depth = np.zeros(frame_count, dtype=np.float64)
    ground_contact = np.zeros(frame_count, dtype=np.bool_)
    ground_depth = np.zeros(frame_count, dtype=np.float64)
    penetration = np.zeros(frame_count, dtype=np.bool_)
    severe_penetration = np.zeros(frame_count, dtype=np.bool_)
    undesired_ground = np.zeros(frame_count, dtype=np.bool_)
    left_mujoco = np.zeros(frame_count, dtype=np.bool_)
    right_mujoco = np.zeros(frame_count, dtype=np.bool_)

    left_points = np.empty((frame_count, len(left_ids), 3), dtype=np.float64)
    right_points = np.empty((frame_count, len(right_ids), 3), dtype=np.float64)
    left_center = np.empty((frame_count, 3), dtype=np.float64)
    right_center = np.empty((frame_count, 3), dtype=np.float64)
    left_height = np.empty(frame_count, dtype=np.float64)
    right_height = np.empty(frame_count, dtype=np.float64)

    self_pairs_by_frame: list[set[BodyPair]] = []
    nonhand_pairs_by_frame: list[set[BodyPair]] = []
    self_pair_depths_by_frame: list[dict[BodyPair, float]] = []
    ground_bodies_by_frame: list[set[str]] = []
    penetration_bodies_by_frame: list[set[str]] = []
    ground_body_depths_by_frame: list[dict[str, float]] = []
    undesired_bodies_by_frame: list[set[str]] = []
    ground_depth_samples: list[float] = []
    raw_contact_count = 0
    self_contact_count = 0
    worst_self_depth = 0.0
    worst_self_pair: BodyPair | None = None
    worst_self_frame: int | None = None
    worst_ground_depth = 0.0
    worst_ground_body: str | None = None
    worst_ground_frame: int | None = None

    seen_frames = 0
    thresholds = evaluation_config.thresholds
    patterns = evaluation_config.allowed_hand_contact_patterns
    for expected_frame, snapshot in enumerate(frames):
        if expected_frame >= frame_count:
            raise ValueError("Received more FK frames than the motion contains")
        if snapshot.frame != expected_frame:
            raise ValueError(
                f"FK frame order mismatch: got {snapshot.frame}, expected {expected_frame}"
            )
        if not np.isclose(snapshot.motion_time_s, expected_frame / motion.fps):
            raise ValueError(f"FK motion time is misaligned at frame {expected_frame}")
        if snapshot.geom_pos_world_m.shape != (model_info.ngeom, 3):
            raise ValueError(
                f"FK geom positions have shape {snapshot.geom_pos_world_m.shape}, "
                f"expected ({model_info.ngeom}, 3)"
            )
        if not np.isfinite(snapshot.geom_pos_world_m).all():
            raise ValueError(f"FK geom positions are non-finite at frame {expected_frame}")

        lp, lc, lh = _foot_geometry(
            snapshot.geom_pos_world_m, left_ids, model_info
        )
        rp, rc, rh = _foot_geometry(
            snapshot.geom_pos_world_m, right_ids, model_info
        )
        left_points[expected_frame] = lp
        right_points[expected_frame] = rp
        left_center[expected_frame] = lc
        right_center[expected_frame] = rc
        left_height[expected_frame] = lh
        right_height[expected_frame] = rh

        frame_self_pairs: set[BodyPair] = set()
        frame_nonhand_pairs: set[BodyPair] = set()
        frame_self_pair_depths: dict[BodyPair, float] = {}
        frame_ground_bodies: set[str] = set()
        frame_penetration_bodies: set[str] = set()
        frame_ground_body_depths: dict[str, float] = {}
        frame_undesired_bodies: set[str] = set()

        raw_contact_count += len(snapshot.contacts)
        for contact in snapshot.contacts:
            if not 0 <= contact.geom1 < model_info.ngeom or not 0 <= contact.geom2 < model_info.ngeom:
                raise ValueError("Contact contains a geom id outside the evaluated model")
            if not np.isfinite(contact.distance_m):
                raise ValueError(
                    f"Contact distance is non-finite at frame {expected_frame}"
                )
            geom1 = contact.geom1
            geom2 = contact.geom2
            depth = max(0.0, -contact.distance_m)
            is_ground = model_info.floor_geom_id in (geom1, geom2)
            if is_ground:
                other_geom = geom2 if geom1 == model_info.floor_geom_id else geom1
                if other_geom == model_info.floor_geom_id:
                    continue
                body_id = model_info.geom_body_ids[other_geom]
                body = model_info.body_names[body_id]
                frame_ground_bodies.add(body)
                frame_ground_body_depths[body] = max(
                    frame_ground_body_depths.get(body, 0.0), depth
                )
                ground_depth_samples.append(depth)
                ground_depth[expected_frame] = max(
                    ground_depth[expected_frame], depth
                )
                if other_geom in left_id_set:
                    left_mujoco[expected_frame] = True
                if other_geom in right_id_set:
                    right_mujoco[expected_frame] = True
                if body_id not in foot_body_ids:
                    frame_undesired_bodies.add(body)
                if depth > thresholds.penetration_m:
                    frame_penetration_bodies.add(body)
                if depth > worst_ground_depth:
                    worst_ground_depth = depth
                    worst_ground_body = body
                    worst_ground_frame = expected_frame
                continue

            body_id1 = model_info.geom_body_ids[geom1]
            body_id2 = model_info.geom_body_ids[geom2]
            if body_id1 == body_id2:
                continue
            body_pair = _pair(
                model_info.body_names[body_id1],
                model_info.body_names[body_id2],
            )
            frame_self_pairs.add(body_pair)
            frame_self_pair_depths[body_pair] = max(
                frame_self_pair_depths.get(body_pair, 0.0), depth
            )
            self_contact_count += 1
            self_count[expected_frame] += 1
            self_depth[expected_frame] = max(self_depth[expected_frame], depth)
            if not _is_allowed_hand_pair(body_pair, patterns):
                frame_nonhand_pairs.add(body_pair)
            if depth > worst_self_depth:
                worst_self_depth = depth
                worst_self_pair = body_pair
                worst_self_frame = expected_frame

        self_any[expected_frame] = bool(frame_self_pairs)
        self_nonhand[expected_frame] = bool(frame_nonhand_pairs)
        ground_contact[expected_frame] = bool(frame_ground_bodies)
        penetration[expected_frame] = (
            ground_depth[expected_frame] > thresholds.penetration_m
        )
        severe_penetration[expected_frame] = (
            ground_depth[expected_frame] > thresholds.severe_penetration_m
        )
        undesired_ground[expected_frame] = bool(frame_undesired_bodies)
        self_pairs_by_frame.append(frame_self_pairs)
        nonhand_pairs_by_frame.append(frame_nonhand_pairs)
        self_pair_depths_by_frame.append(frame_self_pair_depths)
        ground_bodies_by_frame.append(frame_ground_bodies)
        penetration_bodies_by_frame.append(frame_penetration_bodies)
        ground_body_depths_by_frame.append(frame_ground_body_depths)
        undesired_bodies_by_frame.append(frame_undesired_bodies)
        seen_frames += 1

    if seen_frames != frame_count:
        raise ValueError(
            f"Received {seen_frames} FK frames, expected {frame_count}"
        )

    left_horizontal_speed = np.full(frame_count, np.nan, dtype=np.float64)
    right_horizontal_speed = np.full(frame_count, np.nan, dtype=np.float64)
    left_vertical_speed = np.full(frame_count, np.nan, dtype=np.float64)
    right_vertical_speed = np.full(frame_count, np.nan, dtype=np.float64)
    left_horizontal_speed[1:] = (
        np.linalg.norm(np.diff(left_center[:, :2], axis=0), axis=1) * motion.fps
    )
    right_horizontal_speed[1:] = (
        np.linalg.norm(np.diff(right_center[:, :2], axis=0), axis=1) * motion.fps
    )
    left_vertical_speed[1:] = np.diff(left_center[:, 2]) * motion.fps
    right_vertical_speed[1:] = np.diff(right_center[:, 2]) * motion.fps

    left_proxy = (
        np.abs(left_height) <= thresholds.foot_near_ground_m
    ) & (
        np.abs(left_vertical_speed)
        <= thresholds.foot_vertical_speed_contact_m_s
    )
    right_proxy = (
        np.abs(right_height) <= thresholds.foot_near_ground_m
    ) & (
        np.abs(right_vertical_speed)
        <= thresholds.foot_vertical_speed_contact_m_s
    )
    left_mismatch = left_proxy ^ left_mujoco
    right_mismatch = right_proxy ^ right_mujoco
    left_slide = left_proxy & (
        left_horizontal_speed > thresholds.foot_slide_speed_m_s
    )
    right_slide = right_proxy & (
        right_horizontal_speed > thresholds.foot_slide_speed_m_s
    )
    slide = left_slide | right_slide
    airborne = ~(left_proxy | right_proxy)

    contact_speeds = np.concatenate(
        (
            left_horizontal_speed[left_proxy & np.isfinite(left_horizontal_speed)],
            right_horizontal_speed[right_proxy & np.isfinite(right_horizontal_speed)],
        )
    )
    warnings = ["CONTACT_PROXY_WITHOUT_SOURCE_LABELS"]
    if np.any(left_mismatch) or np.any(right_mismatch):
        warnings.append("MUJOCO_CONTACT_PROXY_MISMATCH_PRESENT")
    if np.any(left_mujoco & (left_height > thresholds.foot_near_ground_m)) or np.any(
        ~left_mujoco & (left_height < -thresholds.foot_near_ground_m)
    ):
        warnings.append("LEFT_FOOT_CONTACT_GEOMETRY_INCONSISTENT")
    if np.any(right_mujoco & (right_height > thresholds.foot_near_ground_m)) or np.any(
        ~right_mujoco & (right_height < -thresholds.foot_near_ground_m)
    ):
        warnings.append("RIGHT_FOOT_CONTACT_GEOMETRY_INCONSISTENT")

    self_pairs = tuple(
        sorted(set().union(*self_pairs_by_frame) if self_pairs_by_frame else set())
    )
    nonhand_pairs = tuple(
        sorted(
            set().union(*nonhand_pairs_by_frame)
            if nonhand_pairs_by_frame
            else set()
        )
    )
    undesired_bodies = tuple(
        sorted(
            set().union(*undesired_bodies_by_frame)
            if undesired_bodies_by_frame
            else set()
        )
    )
    ground_samples = np.asarray(ground_depth_samples, dtype=np.float64)
    policy = evaluation_config.policy
    summary = ContactMetricsSummary(
        motion_category=motion_category,
        contact_policy=policy.name,
        undesired_ground_contact_is_failure=policy.undesired_ground_contact_is_failure,
        airborne_is_failure=policy.airborne_is_failure,
        penetration_threshold_m=thresholds.penetration_m,
        severe_penetration_threshold_m=thresholds.severe_penetration_m,
        foot_near_ground_threshold_m=thresholds.foot_near_ground_m,
        foot_vertical_speed_contact_threshold_m_s=thresholds.foot_vertical_speed_contact_m_s,
        foot_slide_speed_threshold_m_s=thresholds.foot_slide_speed_m_s,
        raw_contact_count=raw_contact_count,
        self_collision_contact_count=self_contact_count,
        self_collision_any_frame_count=int(np.count_nonzero(self_any)),
        self_collision_any_frame_rate=float(np.mean(self_any)),
        self_collision_nonhand_frame_count=int(np.count_nonzero(self_nonhand)),
        self_collision_nonhand_frame_rate=float(np.mean(self_nonhand)),
        self_collision_event_count=_entity_event_count(self_pairs_by_frame),
        self_collision_nonhand_event_count=_entity_event_count(
            nonhand_pairs_by_frame
        ),
        max_self_contacts_per_frame=int(np.max(self_count)),
        first_self_collision_frame=_first_true(self_any),
        self_collision_body_pairs=self_pairs,
        self_collision_nonhand_body_pairs=nonhand_pairs,
        max_self_collision_penetration_m=worst_self_depth,
        worst_self_collision_body_pair=worst_self_pair,
        worst_self_collision_frame=worst_self_frame,
        ground_contact_sample_count=len(ground_depth_samples),
        ground_contact_frame_count=int(np.count_nonzero(ground_contact)),
        ground_contact_frame_rate=float(np.mean(ground_contact)),
        ground_penetration_frame_count=int(np.count_nonzero(penetration)),
        ground_penetration_frame_rate=float(np.mean(penetration)),
        ground_penetration_event_count=_count_runs(penetration),
        severe_ground_penetration_frame_count=int(
            np.count_nonzero(severe_penetration)
        ),
        severe_ground_penetration_frame_rate=float(np.mean(severe_penetration)),
        mean_ground_penetration_m=(
            float(np.mean(ground_samples)) if ground_samples.size else 0.0
        ),
        max_ground_penetration_m=worst_ground_depth,
        worst_penetration_body=worst_ground_body,
        worst_penetration_frame=worst_ground_frame,
        undesired_ground_contact_frame_count=int(
            np.count_nonzero(undesired_ground)
        ),
        undesired_ground_contact_frame_rate=float(np.mean(undesired_ground)),
        undesired_ground_contact_event_count=_entity_event_count(
            undesired_bodies_by_frame
        ),
        undesired_ground_contact_bodies=undesired_bodies,
        left_foot_height_min_m=float(np.min(left_height)),
        left_foot_height_p05_m=float(np.percentile(left_height, 5)),
        left_foot_height_p95_m=float(np.percentile(left_height, 95)),
        left_foot_height_max_m=float(np.max(left_height)),
        right_foot_height_min_m=float(np.min(right_height)),
        right_foot_height_p05_m=float(np.percentile(right_height, 5)),
        right_foot_height_p95_m=float(np.percentile(right_height, 95)),
        right_foot_height_max_m=float(np.max(right_height)),
        left_contact_proxy_rate=float(np.mean(left_proxy)),
        right_contact_proxy_rate=float(np.mean(right_proxy)),
        left_mujoco_ground_contact_frame_rate=float(np.mean(left_mujoco)),
        right_mujoco_ground_contact_frame_rate=float(np.mean(right_mujoco)),
        left_contact_proxy_mismatch_frame_count=int(np.count_nonzero(left_mismatch)),
        left_contact_proxy_mismatch_frame_rate=float(np.mean(left_mismatch)),
        right_contact_proxy_mismatch_frame_count=int(
            np.count_nonzero(right_mismatch)
        ),
        right_contact_proxy_mismatch_frame_rate=float(np.mean(right_mismatch)),
        foot_slide_frame_count=int(np.count_nonzero(slide)),
        foot_slide_frame_rate=float(np.mean(slide)),
        left_foot_slide_frame_count=int(np.count_nonzero(left_slide)),
        left_foot_slide_frame_rate=float(np.mean(left_slide)),
        right_foot_slide_frame_count=int(np.count_nonzero(right_slide)),
        right_foot_slide_frame_rate=float(np.mean(right_slide)),
        foot_slide_speed_p95_m_s=_percentile(contact_speeds, 95),
        foot_slide_speed_max_m_s=(
            float(np.max(contact_speeds)) if contact_speeds.size else float("nan")
        ),
        foot_slide_event_count=_count_runs(slide),
        airborne_frame_count=int(np.count_nonzero(airborne)),
        airborne_frame_rate=float(np.mean(airborne)),
        max_airborne_duration_s=_max_run_length(airborne) / motion.fps,
    )

    pairs_tuple = tuple(tuple(sorted(values)) for values in self_pairs_by_frame)
    nonhand_pairs_tuple = tuple(
        tuple(sorted(values)) for values in nonhand_pairs_by_frame
    )
    self_pair_depths_tuple = tuple(
        tuple(sorted(values.items())) for values in self_pair_depths_by_frame
    )
    ground_bodies_tuple = tuple(
        tuple(sorted(values)) for values in ground_bodies_by_frame
    )
    penetration_bodies_tuple = tuple(
        tuple(sorted(values)) for values in penetration_bodies_by_frame
    )
    ground_body_depths_tuple = tuple(
        tuple(sorted(values.items())) for values in ground_body_depths_by_frame
    )
    undesired_bodies_tuple = tuple(
        tuple(sorted(values)) for values in undesired_bodies_by_frame
    )
    return ContactMetrics(
        self_collision_any_frame=_readonly(self_any),
        self_collision_nonhand_frame=_readonly(self_nonhand),
        self_contacts_per_frame=_readonly(self_count),
        self_collision_max_penetration_m=_readonly(self_depth),
        self_collision_pairs=pairs_tuple,
        self_collision_nonhand_pairs=nonhand_pairs_tuple,
        self_collision_pair_penetration_m=self_pair_depths_tuple,
        ground_contact_frame=_readonly(ground_contact),
        ground_contact_bodies=ground_bodies_tuple,
        ground_penetration_frame=_readonly(penetration),
        severe_ground_penetration_frame=_readonly(severe_penetration),
        ground_penetration_depth_m=_readonly(ground_depth),
        ground_penetration_bodies=penetration_bodies_tuple,
        ground_body_penetration_m=ground_body_depths_tuple,
        undesired_ground_contact_frame=_readonly(undesired_ground),
        undesired_ground_contact_bodies=undesired_bodies_tuple,
        left_sole_points_world_m=_readonly(left_points),
        right_sole_points_world_m=_readonly(right_points),
        left_foot_center_world_m=_readonly(left_center),
        right_foot_center_world_m=_readonly(right_center),
        left_foot_height_m=_readonly(left_height),
        right_foot_height_m=_readonly(right_height),
        left_foot_horizontal_speed_m_s=_readonly(left_horizontal_speed),
        right_foot_horizontal_speed_m_s=_readonly(right_horizontal_speed),
        left_foot_vertical_speed_m_s=_readonly(left_vertical_speed),
        right_foot_vertical_speed_m_s=_readonly(right_vertical_speed),
        left_contact_proxy_frame=_readonly(left_proxy),
        right_contact_proxy_frame=_readonly(right_proxy),
        left_mujoco_ground_contact_frame=_readonly(left_mujoco),
        right_mujoco_ground_contact_frame=_readonly(right_mujoco),
        left_contact_proxy_mismatch_frame=_readonly(left_mismatch),
        right_contact_proxy_mismatch_frame=_readonly(right_mismatch),
        left_foot_slide_frame=_readonly(left_slide),
        right_foot_slide_frame=_readonly(right_slide),
        foot_slide_frame=_readonly(slide),
        airborne_frame=_readonly(airborne),
        warnings=tuple(warnings),
        summary=summary,
    )


def compute_contact_metrics(
    bound_motion: BoundMotion,
    kinematics: MujocoKinematics,
    evaluation_config: ContactEvaluationConfig,
    *,
    motion_category: str = "unknown",
) -> ContactMetrics:
    """Run all FK frames and compute step-5 collision/contact metrics."""
    return compute_contact_metrics_from_frames(
        bound_motion.record,
        kinematics.info,
        kinematics.iter_forward(bound_motion),
        evaluation_config,
        motion_category=motion_category,
    )
