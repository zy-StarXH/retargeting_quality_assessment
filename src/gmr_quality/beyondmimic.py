from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .io import load_motion_from_project_config
from .manifest import sha256_file
from .model_info import load_robot_model_from_project_config
from .mujoco_kinematics import xyzw_to_wxyz


# Isaac Lab's URDF importer orders G1 rigid bodies breadth-first.  This exact
# order is also used by the upstream whole_body_tracking csv_to_npz output.
BMIMIC_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)

# For each Isaac-Lab/BeyondMimic DoF index, the corresponding GMR XML DoF.
ISAAC_TO_GMR_DOF = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
        4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)

REQUIRED_BMIMIC_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class ConvertedMotion:
    motion_name: str
    source_pkl: Path
    output_npz: Path
    input_frames: int
    selected_start_frame: int
    selected_frames: int
    output_frames: int
    input_fps: float
    output_fps: float
    sha256: str


def _resample(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = (len(root_pos) - 1) / input_fps
    source_times = np.arange(len(root_pos), dtype=np.float64) / input_fps
    # Match upstream BeyondMimic's arange(0, duration, 1/output_fps).
    target_times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)
    positions = np.column_stack(
        [np.interp(target_times, source_times, root_pos[:, axis]) for axis in range(3)]
    )
    joints = np.column_stack(
        [np.interp(target_times, source_times, joint_pos[:, axis]) for axis in range(joint_pos.shape[1])]
    )
    quaternions = Slerp(
        source_times, Rotation.from_quat(root_quat_xyzw)
    )(target_times).as_quat()
    return positions, quaternions, joints


def _so3_derivative_wxyz(quaternions_wxyz: np.ndarray, dt: float) -> np.ndarray:
    count, bodies, _ = quaternions_wxyz.shape
    angular_velocity = np.zeros((count, bodies, 3), dtype=np.float64)
    if count < 2:
        return angular_velocity
    xyzw = quaternions_wxyz[..., [1, 2, 3, 0]]
    if count == 2:
        relative = Rotation.from_quat(xyzw[1]) * Rotation.from_quat(xyzw[0]).inv()
        angular_velocity[:] = relative.as_rotvec()[None, ...] / dt
        return angular_velocity
    for body in range(bodies):
        rotations = Rotation.from_quat(xyzw[:, body])
        relative = rotations[2:] * rotations[:-2].inv()
        angular_velocity[1:-1, body] = relative.as_rotvec() / (2.0 * dt)
        angular_velocity[0, body] = angular_velocity[1, body]
        angular_velocity[-1, body] = angular_velocity[-2, body]
    return angular_velocity


def validate_bmimic_payload(data: np.lib.npyio.NpzFile | dict[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED_BMIMIC_KEYS if key not in data]
    if missing:
        raise ValueError(f"BeyondMimic payload is missing keys: {missing}")
    fps = np.asarray(data["fps"])
    if fps.shape not in ((), (1,)) or float(fps.reshape(-1)[0]) <= 0:
        raise ValueError(f"Invalid BeyondMimic fps: shape={fps.shape}, value={fps}")
    frames = int(np.asarray(data["joint_pos"]).shape[0])
    expected = {
        "joint_pos": (frames, 29),
        "joint_vel": (frames, 29),
        "body_pos_w": (frames, 30, 3),
        "body_quat_w": (frames, 30, 4),
        "body_lin_vel_w": (frames, 30, 3),
        "body_ang_vel_w": (frames, 30, 3),
    }
    for key, shape in expected.items():
        array = np.asarray(data[key])
        if array.shape != shape:
            raise ValueError(f"{key} has shape {array.shape}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf")
    quat_error = np.abs(np.linalg.norm(np.asarray(data["body_quat_w"]), axis=-1) - 1.0)
    if float(quat_error.max()) > 2e-4:
        raise ValueError(f"body_quat_w is not unit length (max error {quat_error.max()})")
    if "body_names" in data and tuple(np.asarray(data["body_names"]).tolist()) != BMIMIC_BODY_NAMES:
        raise ValueError("body_names does not match the frozen Isaac Lab G1 body order")
    if "motion_starts" in data or "motion_ends" in data:
        starts = np.asarray(data["motion_starts"], dtype=np.int64)
        ends = np.asarray(data["motion_ends"], dtype=np.int64)
        if starts.ndim != 1 or ends.shape != starts.shape or len(starts) == 0:
            raise ValueError("motion_starts/motion_ends must be non-empty matching vectors")
        if starts[0] != 0 or ends[-1] != frames or np.any(ends <= starts):
            raise ValueError("Invalid motion boundaries")
        if len(starts) > 1 and not np.array_equal(starts[1:], ends[:-1]):
            raise ValueError("Motion boundaries must be contiguous")


def convert_gmr_motion(
    project_config: dict,
    motion_name: str,
    source_pkl: str | Path,
    output_npz: str | Path,
    *,
    selected_frames: int,
    output_fps: float = 50.0,
    normalize_xy_origin: bool = True,
) -> ConvertedMotion:
    source = Path(source_pkl).expanduser().resolve()
    destination = Path(output_npz).expanduser().resolve()
    loaded = load_motion_from_project_config(source, project_config)
    record = loaded.require_record()
    if record.num_frames < selected_frames:
        raise ValueError(
            f"{motion_name} has {record.num_frames} frames, fewer than required {selected_frames}"
        )
    start = (record.num_frames - selected_frames) // 2
    stop = start + selected_frames
    root_pos, root_quat, joint_gmr = _resample(
        np.asarray(record.root_pos[start:stop], dtype=np.float64),
        np.asarray(record.root_quat_xyzw[start:stop], dtype=np.float64),
        np.asarray(record.joint_pos[start:stop], dtype=np.float64),
        record.fps,
        output_fps,
    )
    if normalize_xy_origin:
        root_pos[:, :2] -= root_pos[0, :2]

    loaded_model = load_robot_model_from_project_config(project_config)
    model = loaded_model.model
    info = loaded_model.info
    missing_bodies = [name for name in BMIMIC_BODY_NAMES if name not in info.body_names]
    if missing_bodies:
        raise ValueError(f"GMR model is missing BeyondMimic bodies: {missing_bodies}")
    body_ids = np.asarray([info.body_names.index(name) for name in BMIMIC_BODY_NAMES])
    data = mujoco.MjData(model)
    frame_count = len(root_pos)
    body_pos = np.empty((frame_count, 30, 3), dtype=np.float64)
    body_quat = np.empty((frame_count, 30, 4), dtype=np.float64)
    qpos = np.zeros(model.nq, dtype=np.float64)
    for frame in range(frame_count):
        qpos[info.root_qpos_address : info.root_qpos_address + 3] = root_pos[frame]
        qpos[info.root_qpos_address + 3 : info.root_qpos_address + 7] = xyzw_to_wxyz(root_quat[frame])
        qpos[np.asarray(info.joint_qpos_addresses)] = joint_gmr[frame]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]

    dt = 1.0 / output_fps
    joint_isaac = joint_gmr[:, ISAAC_TO_GMR_DOF]
    payload = {
        "fps": np.asarray([int(round(output_fps))], dtype=np.int64),
        "joint_pos": joint_isaac.astype(np.float32),
        "joint_vel": np.gradient(joint_isaac, dt, axis=0).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": np.gradient(body_pos, dt, axis=0).astype(np.float32),
        "body_ang_vel_w": _so3_derivative_wxyz(body_quat, dt).astype(np.float32),
        "body_names": np.asarray(BMIMIC_BODY_NAMES),
        "motion_names": np.asarray([motion_name]),
        "motion_starts": np.asarray([0], dtype=np.int64),
        "motion_ends": np.asarray([frame_count], dtype=np.int64),
        "source_pkl_sha256": np.asarray([sha256_file(source)]),
        "selected_source_start_frame": np.asarray([start], dtype=np.int64),
        "selected_source_frames": np.asarray([selected_frames], dtype=np.int64),
    }
    validate_bmimic_payload(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    with np.load(destination, allow_pickle=False) as saved:
        validate_bmimic_payload(saved)
    return ConvertedMotion(
        motion_name=motion_name,
        source_pkl=source,
        output_npz=destination,
        input_frames=record.num_frames,
        selected_start_frame=start,
        selected_frames=selected_frames,
        output_frames=frame_count,
        input_fps=record.fps,
        output_fps=output_fps,
        sha256=sha256_file(destination),
    )


def merge_bmimic_motions(
    motion_files: Iterable[str | Path],
    output_npz: str | Path,
) -> Path:
    sources = [Path(path).expanduser().resolve() for path in motion_files]
    if not sources:
        raise ValueError("At least one motion file is required")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in REQUIRED_BMIMIC_KEYS if key != "fps"}
    names: list[str] = []
    hashes: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    fps: int | None = None
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            validate_bmimic_payload(data)
            current_fps = int(np.asarray(data["fps"]).reshape(-1)[0])
            if fps is None:
                fps = current_fps
            elif current_fps != fps:
                raise ValueError("Cannot merge motions with different fps")
            for key in arrays:
                arrays[key].append(np.asarray(data[key]))
            motion_names = np.asarray(data.get("motion_names", [source.stem])).tolist()
            if len(motion_names) != 1:
                raise ValueError(f"Expected a single motion in {source}")
            names.append(str(motion_names[0]))
            count = int(np.asarray(data["joint_pos"]).shape[0])
            starts.append(cursor)
            cursor += count
            ends.append(cursor)
            hashes.append(sha256_file(source))
    payload = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    payload.update(
        {
            "fps": np.asarray([fps], dtype=np.int64),
            "body_names": np.asarray(BMIMIC_BODY_NAMES),
            "motion_names": np.asarray(names),
            "motion_starts": np.asarray(starts, dtype=np.int64),
            "motion_ends": np.asarray(ends, dtype=np.int64),
            "component_npz_sha256": np.asarray(hashes),
        }
    )
    validate_bmimic_payload(payload)
    destination = Path(output_npz).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    with np.load(destination, allow_pickle=False) as saved:
        validate_bmimic_payload(saved)
    return destination


def inspect_bmimic_file(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        validate_bmimic_payload(data)
        return {
            "path": str(source),
            "sha256": sha256_file(source),
            "fps": int(np.asarray(data["fps"]).reshape(-1)[0]),
            "frames": int(data["joint_pos"].shape[0]),
            "motion_names": np.asarray(data.get("motion_names", [source.stem])).tolist(),
            "motion_starts": np.asarray(data.get("motion_starts", [0])).astype(int).tolist(),
            "motion_ends": np.asarray(data.get("motion_ends", [len(data["joint_pos"])] )).astype(int).tolist(),
            "body_names": np.asarray(data.get("body_names", BMIMIC_BODY_NAMES)).tolist(),
        }


def write_bmimic_index(entries: Iterable[dict[str, object]], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(entries), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
