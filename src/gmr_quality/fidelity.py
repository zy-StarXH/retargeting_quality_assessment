from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .manifest import sha256_file
from .model_info import BoundMotion
from .mujoco_kinematics import MujocoKinematics


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FidelityConfig:
    source_format: str
    verified_pair_status: str
    supported_frame_alignments: tuple[str, ...]
    ik_config_path: Path
    expected_ik_config_sha256: str
    gmr_lafan1_loader_path: Path
    end_effector_robot_bodies: tuple[str, ...]
    fps_tolerance_hz: float
    position_threshold_m: float
    orientation_threshold_deg: float

    def __post_init__(self) -> None:
        if self.source_format != "bvh_lafan1":
            raise ValueError("Only source_format='bvh_lafan1' is supported")
        if not self.verified_pair_status:
            raise ValueError("verified_pair_status must be non-empty")
        if (
            not self.supported_frame_alignments
            or len(set(self.supported_frame_alignments))
            != len(self.supported_frame_alignments)
        ):
            raise ValueError("supported_frame_alignments must be unique and non-empty")
        if not _SHA256_PATTERN.fullmatch(self.expected_ik_config_sha256):
            raise ValueError("expected_ik_config_sha256 must be lowercase SHA-256")
        if (
            not self.end_effector_robot_bodies
            or len(set(self.end_effector_robot_bodies))
            != len(self.end_effector_robot_bodies)
        ):
            raise ValueError("end_effector_robot_bodies must be unique and non-empty")
        for name, value in (
            ("fps_tolerance_hz", self.fps_tolerance_hz),
            ("position_threshold_m", self.position_threshold_m),
            ("orientation_threshold_deg", self.orientation_threshold_deg),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> FidelityConfig:
        fidelity = config["fidelity"]
        thresholds = config["thresholds"]
        return cls(
            source_format=str(fidelity["source_format"]),
            verified_pair_status=str(fidelity["verified_pair_status"]),
            supported_frame_alignments=tuple(
                str(value) for value in fidelity["supported_frame_alignments"]
            ),
            ik_config_path=Path(
                config["paths"]["gmr_lafan1_g1_ik_config"]
            ).resolve(),
            expected_ik_config_sha256=str(
                fidelity["expected_ik_config_sha256"]
            ),
            gmr_lafan1_loader_path=Path(
                config["paths"]["gmr_lafan1_loader"]
            ).resolve(),
            end_effector_robot_bodies=tuple(
                str(value) for value in fidelity["end_effector_robot_bodies"]
            ),
            fps_tolerance_hz=float(fidelity["fps_tolerance_hz"]),
            position_threshold_m=float(thresholds["fidelity_position_m"]),
            orientation_threshold_deg=float(
                thresholds["fidelity_orientation_deg"]
            ),
        )


@dataclass(frozen=True)
class FidelityBodyMapping:
    robot_body: str
    human_body: str
    position_enabled: bool
    orientation_enabled: bool
    position_offset_local_m: NDArray[np.float64]
    rotation_offset_wxyz: NDArray[np.float64]


@dataclass(frozen=True)
class FidelityMetricsSummary:
    fidelity_source_format: str
    fidelity_pair_status: str
    fidelity_frame_alignment: str
    fidelity_source_frame_count: int
    fidelity_robot_frame_count: int
    fidelity_aligned_frame_count: int
    fidelity_source_fps: float
    fidelity_robot_fps: float
    fidelity_actual_human_height_m: float
    fidelity_human_height_assumption_m: float
    fidelity_height_ratio: float
    fidelity_position_threshold_m: float
    fidelity_orientation_threshold_deg: float
    fidelity_position_body_count: int
    fidelity_orientation_body_count: int
    fidelity_end_effector_body_count: int
    fidelity_position_robot_bodies: tuple[str, ...]
    fidelity_orientation_robot_bodies: tuple[str, ...]
    fidelity_end_effector_robot_bodies: tuple[str, ...]
    keybody_pos_error_mean_m: float
    keybody_pos_error_p95_m: float
    keybody_pos_error_max_m: float
    end_effector_pos_error_mean_m: float
    end_effector_pos_error_p95_m: float
    end_effector_pos_error_max_m: float
    link_orientation_error_mean_deg: float
    link_orientation_error_p95_deg: float
    link_orientation_error_max_deg: float
    fidelity_pass_frame_count: int
    fidelity_pass_frame_rate: float
    worst_fidelity_position_body: str
    worst_fidelity_position_frame: int
    worst_fidelity_orientation_body: str
    worst_fidelity_orientation_frame: int
    fidelity_source_bvh_sha256: str
    fidelity_ik_config_sha256: str
    fidelity_gmr_loader_sha256: str


@dataclass(frozen=True)
class FidelityMetrics:
    mappings: tuple[FidelityBodyMapping, ...]
    keybody_position_error_m: NDArray[np.float64]
    end_effector_position_error_m: NDArray[np.float64]
    link_orientation_error_deg: NDArray[np.float64]
    keybody_pos_error_frame_mean_m: NDArray[np.float64]
    keybody_pos_error_frame_max_m: NDArray[np.float64]
    end_effector_pos_error_frame_mean_m: NDArray[np.float64]
    link_orientation_error_frame_mean_deg: NDArray[np.float64]
    link_orientation_error_frame_max_deg: NDArray[np.float64]
    fidelity_valid_frame: NDArray[np.bool_]
    worst_position_body_by_frame: tuple[str, ...]
    worst_orientation_body_by_frame: tuple[str, ...]
    summary: FidelityMetricsSummary


def _readonly(value: np.ndarray) -> NDArray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _read_bvh_fps(path: Path) -> float:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = re.match(r"\s*Frame Time:\s+([\d.eE+-]+)", line)
            if match:
                frame_time = float(match.group(1))
                if not np.isfinite(frame_time) or frame_time <= 0:
                    break
                return 1.0 / frame_time
    raise ValueError(f"Could not read a positive Frame Time from {path}")


def _load_gmr_lafan1_loader(
    expected_source_path: Path,
) -> Callable[[str, str], tuple[list[dict[str, Any]], float]]:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with (
        contextlib.redirect_stdout(captured_stdout),
        contextlib.redirect_stderr(captured_stderr),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module("general_motion_retargeting.utils.lafan1")
    function = getattr(module, "load_bvh_file", None)
    actual_source = inspect.getsourcefile(function) if function else None
    if actual_source is None:
        raise ValueError("GMR load_bvh_file could not be resolved")
    actual_path = Path(actual_source).resolve()
    if actual_path != expected_source_path:
        raise ValueError(
            "Imported GMR LAFAN1 loader does not match the configured source: "
            f"{actual_path} != {expected_source_path}"
        )
    return function


def _entry(
    robot_body: str,
    raw: Any,
    ground: NDArray[np.float64],
) -> tuple[str, float, float, NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(raw, list) or len(raw) != 5:
        raise ValueError(f"Invalid IK mapping entry for {robot_body}")
    human_body = raw[0]
    if not isinstance(human_body, str) or not human_body:
        raise ValueError(f"Invalid human body for {robot_body}")
    position_weight = float(raw[1])
    orientation_weight = float(raw[2])
    position_offset = np.asarray(raw[3], dtype=np.float64)
    rotation_offset = np.asarray(raw[4], dtype=np.float64)
    if position_offset.shape != (3,) or rotation_offset.shape != (4,):
        raise ValueError(f"Invalid IK offsets for {robot_body}")
    if not np.isfinite(position_offset).all() or not np.isfinite(
        rotation_offset
    ).all():
        raise ValueError(f"Non-finite IK offsets for {robot_body}")
    if not np.isclose(np.linalg.norm(rotation_offset), 1.0, atol=1e-6):
        raise ValueError(f"Rotation offset is not unit length for {robot_body}")
    return (
        human_body,
        position_weight,
        orientation_weight,
        position_offset - ground,
        rotation_offset,
    )


def _same_rotation(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> bool:
    relative = Rotation.from_quat(
        left_wxyz, scalar_first=True
    ).inv() * Rotation.from_quat(right_wxyz, scalar_first=True)
    return bool(relative.magnitude() < 1e-10)


def _parse_ik_config(
    path: Path,
) -> tuple[
    dict[str, Any],
    tuple[FidelityBodyMapping, ...],
    dict[str, float],
]:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    required = {
        "human_root_name",
        "human_height_assumption",
        "human_scale_table",
        "ground_height",
        "use_ik_match_table1",
        "use_ik_match_table2",
        "ik_match_table1",
        "ik_match_table2",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"GMR IK config is missing keys: {', '.join(missing)}")
    ground = float(raw["ground_height"]) * np.array(
        [0.0, 0.0, 1.0], dtype=np.float64
    )
    combined: dict[
        str,
        tuple[str, bool, bool, NDArray[np.float64], NDArray[np.float64]],
    ] = {}
    for stage in (1, 2):
        if not bool(raw[f"use_ik_match_table{stage}"]):
            continue
        table = raw[f"ik_match_table{stage}"]
        if not isinstance(table, dict):
            raise ValueError(f"ik_match_table{stage} must be a mapping")
        for robot_body, value in table.items():
            human, pos_weight, rot_weight, pos_offset, rot_offset = _entry(
                robot_body, value, ground
            )
            if pos_weight == 0 and rot_weight == 0:
                continue
            previous = combined.get(robot_body)
            if previous is None:
                combined[robot_body] = (
                    human,
                    pos_weight != 0,
                    rot_weight != 0,
                    pos_offset,
                    rot_offset,
                )
                continue
            if (
                previous[0] != human
                or not np.allclose(previous[3], pos_offset, atol=1e-12)
                or not _same_rotation(previous[4], rot_offset)
            ):
                raise ValueError(
                    f"IK stages disagree on target identity or offset for {robot_body}"
                )
            combined[robot_body] = (
                human,
                previous[1] or pos_weight != 0,
                previous[2] or rot_weight != 0,
                previous[3],
                previous[4],
            )
    mappings = tuple(
        FidelityBodyMapping(
            robot_body=robot,
            human_body=value[0],
            position_enabled=value[1],
            orientation_enabled=value[2],
            position_offset_local_m=_readonly(value[3]),
            rotation_offset_wxyz=_readonly(value[4]),
        )
        for robot, value in combined.items()
    )
    if not mappings:
        raise ValueError("GMR IK config has no active body mappings")
    scale_table = {
        str(name): float(scale)
        for name, scale in raw["human_scale_table"].items()
    }
    if any(not np.isfinite(value) or value <= 0 for value in scale_table.values()):
        raise ValueError("human_scale_table values must be finite and positive")
    used_human_bodies = {mapping.human_body for mapping in mappings}
    missing_scales = sorted(used_human_bodies - set(scale_table))
    if missing_scales:
        raise ValueError(
            "GMR IK config lacks scale entries for: " + ", ".join(missing_scales)
        )
    return raw, mappings, scale_table


def _aligned_source_frames(
    frames: list[dict[str, Any]],
    robot_frame_count: int,
    alignment: str,
) -> list[dict[str, Any]]:
    if alignment == "exact":
        if len(frames) != robot_frame_count:
            raise ValueError(
                f"Exact alignment requires {robot_frame_count} source frames, "
                f"found {len(frames)}"
            )
        return frames
    if alignment == "source_prefix_drop_last":
        if len(frames) != robot_frame_count + 1:
            raise ValueError(
                "source_prefix_drop_last requires exactly one additional source "
                f"frame; source={len(frames)}, robot={robot_frame_count}"
            )
        return frames[:robot_frame_count]
    raise ValueError(f"Unsupported frame alignment: {alignment}")


def transform_lafan1_frame_targets(
    human_frame: dict[str, Any],
    mappings: tuple[FidelityBodyMapping, ...],
    adjusted_scale_table: dict[str, float],
    human_root_name: str,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Apply the same local scaling and target offsets as GMR update_targets()."""
    required_human_bodies = {mapping.human_body for mapping in mappings}
    missing_human_bodies = sorted(required_human_bodies - set(human_frame))
    if missing_human_bodies:
        raise ValueError(
            "Source frame lacks mapped bodies: " + ", ".join(missing_human_bodies)
        )
    missing_scales = sorted(required_human_bodies - set(adjusted_scale_table))
    if missing_scales:
        raise ValueError(
            "Adjusted scale table lacks bodies: " + ", ".join(missing_scales)
        )
    if human_root_name not in human_frame or human_root_name not in adjusted_scale_table:
        raise ValueError(f"Human root {human_root_name!r} is unavailable")
    root_position = np.asarray(
        human_frame[human_root_name][0], dtype=np.float64
    )
    if root_position.shape != (3,) or not np.isfinite(root_position).all():
        raise ValueError("Human root position must be finite shape (3,)")
    scaled_root_position = adjusted_scale_table[human_root_name] * root_position
    targets: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for mapping in mappings:
        human_position = np.asarray(
            human_frame[mapping.human_body][0], dtype=np.float64
        )
        human_quaternion = np.asarray(
            human_frame[mapping.human_body][1], dtype=np.float64
        )
        if (
            human_position.shape != (3,)
            or human_quaternion.shape != (4,)
            or not np.isfinite(human_position).all()
            or not np.isfinite(human_quaternion).all()
        ):
            raise ValueError(
                f"Human pose for {mapping.human_body!r} must be finite (3,)/(4,)"
            )
        if mapping.human_body == human_root_name:
            target_position = scaled_root_position.copy()
        else:
            target_position = (
                (human_position - root_position)
                * adjusted_scale_table[mapping.human_body]
                + scaled_root_position
            )
        target_rotation = Rotation.from_quat(
            human_quaternion, scalar_first=True
        ) * Rotation.from_quat(
            mapping.rotation_offset_wxyz, scalar_first=True
        )
        target_position = target_position + target_rotation.apply(
            mapping.position_offset_local_m
        )
        targets[mapping.robot_body] = (
            _readonly(target_position),
            _readonly(target_rotation.as_quat(scalar_first=True)),
        )
    return targets


def compute_lafan1_fidelity(
    bound_motion: BoundMotion,
    kinematics: MujocoKinematics,
    source_bvh: str | Path,
    fidelity_config: FidelityConfig,
    *,
    pair_status: str,
    expected_pkl_sha256: str,
    expected_source_bvh_sha256: str,
    frame_alignment: str,
) -> FidelityMetrics:
    """Compare verified GMR LAFAN1 task targets with robot body FK poses."""
    if pair_status != fidelity_config.verified_pair_status:
        raise ValueError(
            f"Fidelity requires pair_status={fidelity_config.verified_pair_status!r}, "
            f"got {pair_status!r}"
        )
    for label, value in (
        ("expected_pkl_sha256", expected_pkl_sha256),
        ("expected_source_bvh_sha256", expected_source_bvh_sha256),
    ):
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"{label} must be lowercase SHA-256")
    if frame_alignment not in fidelity_config.supported_frame_alignments:
        raise ValueError(
            f"Frame alignment {frame_alignment!r} is not enabled by configuration"
        )
    motion = bound_motion.record
    actual_pkl_hash = sha256_file(motion.source_path)
    if actual_pkl_hash != expected_pkl_sha256:
        raise ValueError(
            "Verified-pair PKL SHA-256 mismatch: "
            f"expected {expected_pkl_sha256}, got {actual_pkl_hash}"
        )
    source_path = Path(source_bvh).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"Source BVH does not exist: {source_path}")
    actual_source_hash = sha256_file(source_path)
    if actual_source_hash != expected_source_bvh_sha256:
        raise ValueError(
            "Verified-pair BVH SHA-256 mismatch: "
            f"expected {expected_source_bvh_sha256}, got {actual_source_hash}"
        )
    ik_hash = sha256_file(fidelity_config.ik_config_path)
    if ik_hash != fidelity_config.expected_ik_config_sha256:
        raise ValueError(
            "GMR IK config SHA-256 mismatch: "
            f"expected {fidelity_config.expected_ik_config_sha256}, got {ik_hash}"
        )
    loader = _load_gmr_lafan1_loader(
        fidelity_config.gmr_lafan1_loader_path
    )
    frames, actual_human_height = loader(str(source_path), format="lafan1")
    source_frame_count = len(frames)
    frames = _aligned_source_frames(
        frames, motion.num_frames, frame_alignment
    )
    source_fps = _read_bvh_fps(source_path)
    if not np.isclose(
        source_fps,
        motion.fps,
        rtol=0.0,
        atol=fidelity_config.fps_tolerance_hz,
    ):
        raise ValueError(
            f"Source/robot FPS mismatch: source={source_fps}, robot={motion.fps}"
        )

    raw_ik, mappings, base_scales = _parse_ik_config(
        fidelity_config.ik_config_path
    )
    human_height_assumption = float(raw_ik["human_height_assumption"])
    if (
        not np.isfinite(actual_human_height)
        or actual_human_height <= 0
        or not np.isfinite(human_height_assumption)
        or human_height_assumption <= 0
    ):
        raise ValueError("GMR human heights must be finite and positive")
    height_ratio = float(actual_human_height / human_height_assumption)
    scales = {name: value * height_ratio for name, value in base_scales.items()}
    root_name = str(raw_ik["human_root_name"])
    if root_name not in scales:
        raise ValueError(f"Human root {root_name!r} is missing from scale table")

    body_ids = {name: index for index, name in enumerate(kinematics.info.body_names)}
    missing_robot_bodies = sorted(
        {mapping.robot_body for mapping in mappings} - set(body_ids)
    )
    if missing_robot_bodies:
        raise ValueError(
            "Mapped robot bodies are absent from the evaluated XML: "
            + ", ".join(missing_robot_bodies)
        )
    position_mappings = tuple(
        mapping for mapping in mappings if mapping.position_enabled
    )
    orientation_mappings = tuple(
        mapping for mapping in mappings if mapping.orientation_enabled
    )
    end_effectors = tuple(
        mapping
        for mapping in position_mappings
        if mapping.robot_body in fidelity_config.end_effector_robot_bodies
    )
    if set(fidelity_config.end_effector_robot_bodies) != {
        mapping.robot_body for mapping in end_effectors
    }:
        raise ValueError(
            "Configured end effectors must all be position-enabled GMR targets"
        )

    frame_count = motion.num_frames
    position_error = np.empty(
        (frame_count, len(position_mappings)), dtype=np.float64
    )
    orientation_error = np.empty(
        (frame_count, len(orientation_mappings)), dtype=np.float64
    )
    position_index = {
        mapping.robot_body: index
        for index, mapping in enumerate(position_mappings)
    }
    end_effector_indices = tuple(
        position_index[mapping.robot_body] for mapping in end_effectors
    )

    for frame_index, snapshot in enumerate(kinematics.iter_forward(bound_motion)):
        raw_frame = frames[frame_index]
        targets = transform_lafan1_frame_targets(
            raw_frame,
            mappings,
            scales,
            root_name,
        )

        for body_index, mapping in enumerate(position_mappings):
            robot_position = snapshot.body_pos_world_m[
                body_ids[mapping.robot_body]
            ]
            position_error[frame_index, body_index] = np.linalg.norm(
                targets[mapping.robot_body][0] - robot_position
            )
        for body_index, mapping in enumerate(orientation_mappings):
            robot_rotation = Rotation.from_quat(
                snapshot.body_quat_wxyz[body_ids[mapping.robot_body]],
                scalar_first=True,
            )
            target_rotation = Rotation.from_quat(
                targets[mapping.robot_body][1], scalar_first=True
            )
            orientation_error[frame_index, body_index] = np.degrees(
                (target_rotation.inv() * robot_rotation).magnitude()
            )

    end_effector_error = position_error[:, end_effector_indices]
    position_frame_mean = np.mean(position_error, axis=1)
    position_frame_max = np.max(position_error, axis=1)
    end_effector_frame_mean = np.mean(end_effector_error, axis=1)
    orientation_frame_mean = np.mean(orientation_error, axis=1)
    orientation_frame_max = np.max(orientation_error, axis=1)
    valid = (position_frame_mean < fidelity_config.position_threshold_m) & (
        orientation_frame_mean < fidelity_config.orientation_threshold_deg
    )
    worst_position_flat = int(np.argmax(position_error))
    worst_position_frame, worst_position_body_index = np.unravel_index(
        worst_position_flat, position_error.shape
    )
    worst_orientation_flat = int(np.argmax(orientation_error))
    worst_orientation_frame, worst_orientation_body_index = np.unravel_index(
        worst_orientation_flat, orientation_error.shape
    )
    position_names = tuple(mapping.robot_body for mapping in position_mappings)
    orientation_names = tuple(
        mapping.robot_body for mapping in orientation_mappings
    )
    end_effector_names = tuple(mapping.robot_body for mapping in end_effectors)
    worst_position_by_frame = tuple(
        position_names[index] for index in np.argmax(position_error, axis=1)
    )
    worst_orientation_by_frame = tuple(
        orientation_names[index]
        for index in np.argmax(orientation_error, axis=1)
    )
    summary = FidelityMetricsSummary(
        fidelity_source_format=fidelity_config.source_format,
        fidelity_pair_status=pair_status,
        fidelity_frame_alignment=frame_alignment,
        fidelity_source_frame_count=source_frame_count,
        fidelity_robot_frame_count=frame_count,
        fidelity_aligned_frame_count=len(frames),
        fidelity_source_fps=source_fps,
        fidelity_robot_fps=motion.fps,
        fidelity_actual_human_height_m=float(actual_human_height),
        fidelity_human_height_assumption_m=human_height_assumption,
        fidelity_height_ratio=height_ratio,
        fidelity_position_threshold_m=fidelity_config.position_threshold_m,
        fidelity_orientation_threshold_deg=(
            fidelity_config.orientation_threshold_deg
        ),
        fidelity_position_body_count=len(position_mappings),
        fidelity_orientation_body_count=len(orientation_mappings),
        fidelity_end_effector_body_count=len(end_effectors),
        fidelity_position_robot_bodies=position_names,
        fidelity_orientation_robot_bodies=orientation_names,
        fidelity_end_effector_robot_bodies=end_effector_names,
        keybody_pos_error_mean_m=float(np.mean(position_error)),
        keybody_pos_error_p95_m=float(np.percentile(position_error, 95)),
        keybody_pos_error_max_m=float(np.max(position_error)),
        end_effector_pos_error_mean_m=float(np.mean(end_effector_error)),
        end_effector_pos_error_p95_m=float(
            np.percentile(end_effector_error, 95)
        ),
        end_effector_pos_error_max_m=float(np.max(end_effector_error)),
        link_orientation_error_mean_deg=float(np.mean(orientation_error)),
        link_orientation_error_p95_deg=float(
            np.percentile(orientation_error, 95)
        ),
        link_orientation_error_max_deg=float(np.max(orientation_error)),
        fidelity_pass_frame_count=int(np.count_nonzero(valid)),
        fidelity_pass_frame_rate=float(np.mean(valid)),
        worst_fidelity_position_body=position_names[worst_position_body_index],
        worst_fidelity_position_frame=int(worst_position_frame),
        worst_fidelity_orientation_body=(
            orientation_names[worst_orientation_body_index]
        ),
        worst_fidelity_orientation_frame=int(worst_orientation_frame),
        fidelity_source_bvh_sha256=actual_source_hash,
        fidelity_ik_config_sha256=ik_hash,
        fidelity_gmr_loader_sha256=sha256_file(
            fidelity_config.gmr_lafan1_loader_path
        ),
    )
    return FidelityMetrics(
        mappings=mappings,
        keybody_position_error_m=_readonly(position_error),
        end_effector_position_error_m=_readonly(end_effector_error),
        link_orientation_error_deg=_readonly(orientation_error),
        keybody_pos_error_frame_mean_m=_readonly(position_frame_mean),
        keybody_pos_error_frame_max_m=_readonly(position_frame_max),
        end_effector_pos_error_frame_mean_m=_readonly(
            end_effector_frame_mean
        ),
        link_orientation_error_frame_mean_deg=_readonly(
            orientation_frame_mean
        ),
        link_orientation_error_frame_max_deg=_readonly(
            orientation_frame_max
        ),
        fidelity_valid_frame=_readonly(valid),
        worst_position_body_by_frame=worst_position_by_frame,
        worst_orientation_body_by_frame=worst_orientation_by_frame,
        summary=summary,
    )
