from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr, t

from .interaction import compute_interaction_metrics, load_interaction_record
from .manifest import sha256_file


@dataclass(frozen=True)
class Step12Result:
    output_dir: Path
    demonstration_count: int
    robustness_episode_count: int
    manifest_json: Path
    score_is_calibrated: bool


def load_step12_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("protocol_version") != "step12_g1_bimanual_carry_v1":
        raise ValueError("Unsupported Step-12 protocol")
    robot_xml = Path(config["robot"]["xml"]).expanduser().resolve()
    if sha256_file(robot_xml) != config["robot"]["expected_xml_sha256"]:
        raise ValueError("Frozen G1 XML hash changed")
    return config


def build_interaction_scene(config: dict[str, Any], destination: str | Path) -> Path:
    source = Path(config["robot"]["xml"]).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Robot XML has no compiler element")
    compiler.set("meshdir", str((source.parent / "meshes").resolve()))
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", str(config["task"]["simulation_timestep_s"]))
    option.set("gravity", "0 0 -9.81")

    bodies = {body.get("name"): body for body in root.iter("body")}
    radius = float(config["robot"]["hand_proxy_radius_m"])
    for side in ("left", "right"):
        body_name = config["robot"][f"{side}_hand_body"]
        body = bodies.get(body_name)
        if body is None:
            raise ValueError(f"Missing hand body {body_name}")
        body.append(
            ET.Element(
                "geom",
                {
                    "name": f"{side}_hand_proxy",
                    "type": "sphere",
                    "size": str(radius),
                    "mass": "0.001",
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": "5.0 0.01 0.001",
                    "rgba": "0.1 0.8 0.1 0.35",
                },
            )
        )

    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError("Robot XML has no worldbody")
    scene = worldbodies[-1]
    task = config["task"]
    initial = np.asarray(task["object_initial_pos_m"], dtype=float)
    half = np.asarray(task["object_half_size_m"], dtype=float)
    target = initial + np.asarray(task["target_displacement_m"], dtype=float)
    table_top = float(task["table_top_z_m"])
    table_center_x = float((initial[0] + target[0]) / 2)
    scene.append(
        ET.Element(
            "geom",
            {
                "name": "interaction_table",
                "type": "box",
                "pos": f"{table_center_x} 0 {table_top / 2}",
                "size": f"0.40 0.36 {table_top / 2}",
                "friction": "1.2 0.01 0.001",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0.45 0.32 0.20 1",
            },
        )
    )
    object_body = ET.Element(
        "body",
        {
            "name": "manipulated_object",
            "pos": " ".join(str(value) for value in initial),
        },
    )
    object_body.append(ET.Element("freejoint", {"name": "object_freejoint"}))
    object_body.append(
        ET.Element(
            "geom",
            {
                "name": "manipulated_object_geom",
                "type": "box",
                "size": " ".join(str(value) for value in half),
                "mass": str(task["object_mass_kg"]),
                "friction": f"{task['object_friction']} 0.01 0.001",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0.8 0.15 0.1 1",
            },
        )
    )
    scene.append(object_body)
    carrier_body = ET.Element(
        "body",
        {
            "name": "grasp_carrier",
            "mocap": "true",
            "pos": " ".join(str(value) for value in initial),
        },
    )
    carrier_body.append(
        ET.Element(
            "geom",
            {
                "name": "grasp_carrier_marker",
                "type": "sphere",
                "size": "0.008",
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0.9 0.9 0.1 0.3",
            },
        )
    )
    scene.append(carrier_body)
    scene.append(
        ET.Element(
            "site",
            {
                "name": "object_target",
                "type": "box",
                "pos": " ".join(str(value) for value in target),
                "size": " ".join(str(value * 1.05) for value in half),
                "rgba": "0.1 0.3 0.9 0.25",
            },
        )
    )
    equality = root.find("equality")
    if equality is None:
        equality = ET.Element("equality")
        root.append(equality)
    equality.append(
        ET.Element(
            "weld",
            {
                "name": "bilateral_grasp_weld",
                "body1": "grasp_carrier",
                "body2": "manipulated_object",
                "relpose": "0 0 0 1 0 0 0",
                "active": "false",
                "solref": "0.015 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    # Loading is part of generation: malformed assets fail before experiments.
    mujoco.MjModel.from_xml_path(str(output))
    return output


def _phase_arrays(config: dict[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    phases = tuple(config["task"]["phase_frames"])
    ids = np.concatenate(
        [np.full(int(config["task"]["phase_frames"][name]), index, dtype=np.int64) for index, name in enumerate(phases)]
    )
    return ids, phases


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _phase_progress(phase_ids: np.ndarray, phase_index: int) -> np.ndarray:
    indices = np.flatnonzero(phase_ids == phase_index)
    result = np.zeros(len(phase_ids), dtype=float)
    if len(indices):
        result[indices] = _smoothstep(np.linspace(0.0, 1.0, len(indices)))
    return result


def _hand_targets(
    config: dict[str, Any],
    model: mujoco.MjModel,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    phase_ids, phases = _phase_arrays(config)
    data = mujoco.MjData(model)
    data.qpos[:7] = np.asarray(config["robot"]["root_qpos"], dtype=float)
    mujoco.mj_forward(model, data)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["robot"]["left_hand_body"])
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["robot"]["right_hand_body"])
    starts = np.stack([data.xpos[left_id], data.xpos[right_id]])
    task = config["task"]
    perturb = config["demonstrations"]["perturbation"]
    initial = np.asarray(task["object_initial_pos_m"], dtype=float)
    grasp_y = float(task["grasp_hand_y_m"]) + severity**3 * float(
        perturb["grasp_gap_added_at_max_severity_m"]
    )
    carry_distance = max(
        0.0,
        float(task["target_displacement_m"][0])
        - severity * float(perturb["carry_shortfall_at_max_severity_m"]),
    )
    rng = np.random.default_rng(seed)
    bias_sign = -1.0 if rng.integers(0, 2) == 0 else 1.0
    lateral_bias = bias_sign * severity**2 * float(perturb["common_lateral_bias_at_max_severity_m"])
    jitter_amplitude = severity * float(perturb["vertical_jitter_at_max_severity_m"])
    jitter_phase = rng.uniform(0.0, 2 * np.pi)
    targets = np.zeros((len(phase_ids), 2, 3), dtype=float)
    current = starts.copy()
    cursor = 0
    phase_targets = {
        "approach": np.array(
            [[initial[0], grasp_y + lateral_bias, initial[2]], [initial[0], -grasp_y + lateral_bias, initial[2]]]
        ),
        "grasp": None,
        "lift": None,
        "carry": None,
        "lower": None,
        "release": None,
        "settle": None,
    }
    phase_targets["grasp"] = phase_targets["approach"].copy()
    phase_targets["lift"] = phase_targets["grasp"] + np.array([0.0, 0.0, float(task["lift_height_m"])])
    phase_targets["carry"] = phase_targets["lift"] + np.array([carry_distance, 0.0, 0.0])
    phase_targets["lower"] = phase_targets["carry"] - np.array([0.0, 0.0, float(task["lift_height_m"])])
    released = phase_targets["lower"].copy()
    released[0, 1] = float(task["open_hand_y_m"]) + lateral_bias
    released[1, 1] = -float(task["open_hand_y_m"]) + lateral_bias
    phase_targets["release"] = released
    phase_targets["settle"] = released.copy()
    for phase_index, phase_name in enumerate(phases):
        count = int(task["phase_frames"][phase_name])
        target = np.asarray(phase_targets[phase_name])
        progress = _smoothstep(np.linspace(0.0, 1.0, count))[:, None, None]
        targets[cursor : cursor + count] = current + progress * (target - current)
        current = target
        cursor += count
    targets[:, :, 2] += jitter_amplitude * np.sin(np.linspace(0, 12 * np.pi, len(targets)) + jitter_phase)[:, None]
    delay = int(round(severity**2 * int(perturb["left_right_desync_at_max_severity_frames"])))
    if delay:
        delayed = targets[:, 0].copy()
        targets[:delay, 0] = delayed[0]
        targets[delay:, 0] = delayed[:-delay]
    expected_contact = np.zeros((len(phase_ids), 2), dtype=bool)
    contact_phases = {phases.index(name) for name in ("grasp", "lift", "carry", "lower")}
    expected_contact[np.isin(phase_ids, list(contact_phases))] = True
    return targets, expected_contact, phase_ids, phases


def _arm_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_names = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )
    ]
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], dtype=int
    )
    if np.any(joint_ids < 0):
        raise ValueError("Could not resolve all G1 arm joints")
    return (
        model.jnt_qposadr[joint_ids].copy(),
        model.jnt_dofadr[joint_ids].copy(),
        model.jnt_range[joint_ids, 0].copy(),
        model.jnt_range[joint_ids, 1].copy(),
    )


def solve_hand_trajectory(
    config: dict[str, Any],
    model: mujoco.MjModel,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_addresses, dof_addresses, lower, upper = _arm_indices(model)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["robot"]["left_hand_body"])
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["robot"]["right_hand_body"])
    data = mujoco.MjData(model)
    data.qpos[:7] = np.asarray(config["robot"]["root_qpos"], dtype=float)
    robot_qpos = np.zeros((len(targets), 36), dtype=float)
    left_pos = np.zeros((len(targets), 3), dtype=float)
    right_pos = np.zeros_like(left_pos)
    left_quat = np.zeros((len(targets), 4), dtype=float)
    right_quat = np.zeros_like(left_quat)
    previous = np.zeros(len(q_addresses), dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros_like(jacp)
    for frame, target in enumerate(targets):
        q = previous.copy()
        for _ in range(100):
            data.qpos[:] = 0.0
            data.qpos[:7] = np.asarray(config["robot"]["root_qpos"], dtype=float)
            data.qpos[q_addresses] = q
            mujoco.mj_forward(model, data)
            error = np.concatenate([target[0] - data.xpos[left_id], target[1] - data.xpos[right_id]])
            if np.linalg.norm(error) < 2e-4:
                break
            mujoco.mj_jacBody(model, data, jacp, jacr, left_id)
            left_jac = jacp[:, dof_addresses].copy()
            mujoco.mj_jacBody(model, data, jacp, jacr, right_id)
            right_jac = jacp[:, dof_addresses].copy()
            jacobian = np.vstack([left_jac, right_jac])
            damping = 5e-5
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            q = np.clip(q + np.clip(delta, -0.12, 0.12), lower + 1e-5, upper - 1e-5)
        previous = q
        data.qpos[:] = 0.0
        data.qpos[:7] = np.asarray(config["robot"]["root_qpos"], dtype=float)
        data.qpos[q_addresses] = q
        mujoco.mj_forward(model, data)
        robot_qpos[frame] = data.qpos[:36]
        left_pos[frame] = data.xpos[left_id]
        right_pos[frame] = data.xpos[right_id]
        left_quat[frame] = data.xquat[left_id][[1, 2, 3, 0]]
        right_quat[frame] = data.xquat[right_id][[1, 2, 3, 0]]
    max_error = max(
        float(np.max(np.linalg.norm(left_pos - targets[:, 0], axis=1))),
        float(np.max(np.linalg.norm(right_pos - targets[:, 1], axis=1))),
    )
    if max_error > 0.10:
        raise RuntimeError(f"G1 interaction IK failed: maximum hand error {max_error:.4f} m")
    return robot_qpos, left_pos, right_pos, left_quat, right_quat


def _contact_flags(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom: int,
    hand_geoms: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(2, dtype=bool)
    for contact in data.contact[: data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        for side, hand_geom in enumerate(hand_geoms):
            if object_geom in pair and hand_geom in pair:
                result[side] = True
    return result


def replay_object_physics(
    scene_xml: str | Path,
    config: dict[str, Any],
    robot_qpos: np.ndarray,
    *,
    expected_contact: np.ndarray | None = None,
    initial_offset_m: np.ndarray | None = None,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    scene_path = Path(scene_xml).resolve()
    model = (
        mujoco.MjModel.from_binary_path(str(scene_path))
        if scene_path.suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(scene_path))
    )
    data = mujoco.MjData(model)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    object_qadr = int(model.jnt_qposadr[object_joint])
    object_dadr = int(model.jnt_dofadr[object_joint])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "manipulated_object")
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "manipulated_object_geom")
    hand_geoms = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_proxy")
        for side in ("left", "right")
    )
    grasp_weld = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "bilateral_grasp_weld"
    )
    carrier_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_carrier")
    carrier_mocap = int(model.body_mocapid[carrier_body])
    hand_body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["robot"][f"{side}_hand_body"])
        for side in ("left", "right")
    )
    original_mass = float(model.body_mass[object_body])
    model.body_mass[object_body] = original_mass * mass_scale
    model.body_inertia[object_body] *= mass_scale
    model.geom_friction[object_geom] *= friction_scale
    initial = np.asarray(config["task"]["object_initial_pos_m"], dtype=float).copy()
    if initial_offset_m is not None:
        initial += np.asarray(initial_offset_m, dtype=float)
    mujoco.mj_resetData(model, data)
    data.qpos[:36] = robot_qpos[0]
    data.qpos[object_qadr : object_qadr + 3] = initial
    data.qpos[object_qadr + 3 : object_qadr + 7] = np.asarray(
        config["task"]["object_initial_quat_xyzw"], dtype=float
    )[[3, 0, 1, 2]]
    kinematic_data = mujoco.MjData(model)
    hand_midpoints = np.zeros((len(robot_qpos), 3), dtype=float)
    for frame, pose in enumerate(robot_qpos):
        kinematic_data.qpos[:36] = pose
        mujoco.mj_kinematics(model, kinematic_data)
        hand_midpoints[frame] = 0.5 * (
            kinematic_data.xpos[hand_body_ids[0]] + kinematic_data.xpos[hand_body_ids[1]]
        )
    data.mocap_pos[carrier_mocap] = hand_midpoints[0]
    data.mocap_quat[carrier_mocap] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    frame_dt = 1.0 / float(config["task"]["fps"])
    substeps = int(round(frame_dt / model.opt.timestep))
    if not np.isclose(substeps * model.opt.timestep, frame_dt):
        raise ValueError("Simulation timestep must divide the interaction frame interval")
    positions = np.zeros((len(robot_qpos), 3), dtype=float)
    quaternions = np.zeros((len(robot_qpos), 4), dtype=float)
    velocities = np.zeros((len(robot_qpos), 3), dtype=float)
    contacts = np.zeros((len(robot_qpos), 2), dtype=bool)
    previous = robot_qpos[0].copy()
    missing_bilateral_frames = 0
    grasp_latched = False
    latch_frames = np.zeros(len(robot_qpos), dtype=bool)
    for frame, target in enumerate(robot_qpos):
        contact_expected = bool(expected_contact is not None and expected_contact[frame].all())
        if not contact_expected and grasp_latched:
            data.eq_active[grasp_weld] = 0
            grasp_latched = False
        frame_contact = np.zeros(2, dtype=bool)
        joint_velocity = (target[7:36] - previous[7:36]) / frame_dt
        for substep in range(substeps):
            alpha = (substep + 1) / substeps
            data.qpos[:36] = previous + alpha * (target - previous)
            data.qvel[:6] = 0.0
            data.qvel[6:35] = joint_velocity
            previous_midpoint = hand_midpoints[max(frame - 1, 0)]
            data.mocap_pos[carrier_mocap] = previous_midpoint + alpha * (
                hand_midpoints[frame] - previous_midpoint
            )
            mujoco.mj_step(model, data)
            frame_contact |= _contact_flags(model, data, object_geom, hand_geoms)
            if contact_expected and frame_contact.all() and not grasp_latched:
                data.eq_active[grasp_weld] = 1
                grasp_latched = True
                missing_bilateral_frames = 0
        previous = target
        if grasp_latched:
            if frame_contact.all():
                missing_bilateral_frames = 0
            else:
                missing_bilateral_frames += 1
                if missing_bilateral_frames >= 5:
                    data.eq_active[grasp_weld] = 0
                    grasp_latched = False
            latch_frames[frame] = grasp_latched
        positions[frame] = data.qpos[object_qadr : object_qadr + 3]
        quaternions[frame] = data.qpos[object_qadr + 3 : object_qadr + 7][[1, 2, 3, 0]]
        velocities[frame] = data.qvel[object_dadr : object_dadr + 3]
        contacts[frame] = frame_contact
    task = config["task"]
    target_pos = np.asarray(task["object_initial_pos_m"], dtype=float) + np.asarray(
        task["target_displacement_m"], dtype=float
    )
    final_axis_error = np.abs(positions[-1] - target_pos)
    max_lift = float(np.max(positions[:, 2]) - positions[0, 2])
    success = bool(
        np.all(final_axis_error <= np.asarray(task["success_position_tolerance_m"], dtype=float))
        and np.linalg.norm(velocities[-1]) <= float(task["success_max_linear_speed_m_s"])
        and max_lift >= float(task["success_min_lift_m"])
    )
    payload: dict[str, Any] = {
        "success": success,
        "final_position_error_m": float(np.linalg.norm(final_axis_error)),
        "final_speed_m_s": float(np.linalg.norm(velocities[-1])),
        "max_lift_m": max_lift,
        "bilateral_contact_rate": float(np.mean(contacts.all(axis=1))),
        "grasp_latch_rate": float(np.mean(latch_frames)),
    }
    if record_trajectory:
        payload.update(
            {
                "object_pos_m": positions,
                "object_quat_xyzw": quaternions,
                "object_linvel_m_s": velocities,
                "observed_contact": contacts,
            }
        )
    return payload


def _robust_episode_worker(specification: dict[str, Any]) -> dict[str, Any]:
    result = replay_object_physics(
        specification["scene_mjb"],
        specification["config"],
        specification["robot_qpos"],
        expected_contact=specification["expected_contact"],
        initial_offset_m=specification["initial_offset_m"],
        mass_scale=specification["mass_scale"],
        friction_scale=specification["friction_scale"],
    )
    return {**specification["metadata"], **result}


def _write_demo(
    path: Path,
    config: dict[str, Any],
    name: str,
    robot_qpos: np.ndarray,
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
    expected_contact: np.ndarray,
    phase_ids: np.ndarray,
    phase_names: tuple[str, ...],
    physics: dict[str, Any],
) -> None:
    task = config["task"]
    target = np.asarray(task["object_initial_pos_m"], dtype=float) + np.asarray(
        task["target_displacement_m"], dtype=float
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("gmr_interaction_v1"),
        name=np.asarray(name),
        task_label=np.asarray(task["label"]),
        interaction_mode=np.asarray(task["interaction_mode"]),
        fps=np.asarray(task["fps"], dtype=np.float64),
        robot_qpos=robot_qpos,
        left_hand_pos_m=left_pos,
        right_hand_pos_m=right_pos,
        left_hand_quat_xyzw=left_quat,
        right_hand_quat_xyzw=right_quat,
        object_pos_m=physics["object_pos_m"],
        object_quat_xyzw=physics["object_quat_xyzw"],
        object_linvel_m_s=physics["object_linvel_m_s"],
        expected_contact=expected_contact,
        observed_contact=physics["observed_contact"],
        phase_id=phase_ids,
        phase_names=np.asarray(phase_names),
        target_pos_m=target,
        object_half_size_m=np.asarray(task["object_half_size_m"], dtype=np.float64),
        hand_proxy_radius_m=np.asarray(config["robot"]["hand_proxy_radius_m"], dtype=np.float64),
    )


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return float(values.mean() - half), float(values.mean() + half)


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, count: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(count, len(x)))
    xr = rankdata(x[indices], axis=1)
    yr = rankdata(y[indices], axis=1)
    xr -= xr.mean(axis=1, keepdims=True)
    yr -= yr.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(xr**2, axis=1) * np.sum(yr**2, axis=1))
    valid = denominator > 0
    values = np.sum(xr[valid] * yr[valid], axis=1) / denominator[valid]
    if not len(values):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def run_step12_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    progress: Any = None,
) -> Step12Result:
    config_path = Path(config_path).expanduser().resolve()
    config = load_step12_config(config_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Step-12 output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scene_xml = build_interaction_scene(config, output / "scene" / "g1_bimanual_task.xml")
    scene_mjb = output / "scene" / "g1_bimanual_task.mjb"
    mujoco.mj_saveModel(mujoco.MjModel.from_xml_path(str(scene_xml)), str(scene_mjb), None)
    resolved_config = output / "resolved_step12_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ik_model = mujoco.MjModel.from_xml_path(config["robot"]["xml"])
    metric_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    episode_specs: list[dict[str, Any]] = []
    robustness = config["robustness_evaluation"]
    robustness_rng = np.random.default_rng(int(robustness["random_seed"]))
    demos: list[Path] = []
    for level in config["demonstrations"]["levels"]:
        severity = float(level["severity"])
        for seed in config["demonstrations"]["seeds"]:
            name = f"{level['name']}_seed_{seed}"
            if progress:
                progress(f"Generating and evaluating {name}")
            targets, expected_contact, phase_ids, phase_names = _hand_targets(
                config, ik_model, severity, int(seed)
            )
            robot_qpos, left_pos, right_pos, left_quat, right_quat = solve_hand_trajectory(
                config, ik_model, targets
            )
            nominal = replay_object_physics(
                scene_mjb, config, robot_qpos, expected_contact=expected_contact, record_trajectory=True
            )
            demo_path = output / "demonstrations" / f"{name}.npz"
            _write_demo(
                demo_path, config, name, robot_qpos, left_pos, right_pos, left_quat,
                right_quat, expected_contact, phase_ids, phase_names, nominal,
            )
            demos.append(demo_path)
            record = load_interaction_record(demo_path)
            metrics = compute_interaction_metrics(record, config)
            metric_rows.append({"quality_level": level["name"], "severity": severity, "seed": int(seed), **metrics})
            catalog_rows.append(
                {
                    "motion_name": name,
                    "quality_level": level["name"],
                    "severity": severity,
                    "seed": int(seed),
                    "task_label": record.task_label,
                    "interaction_mode": record.interaction_mode,
                    "npz_path": str(demo_path),
                    "npz_sha256": sha256_file(demo_path),
                    "frames": record.num_frames,
                    "nominal_task_success": metrics["nominal_task_success"],
                }
            )
            for episode in range(int(robustness["episodes_per_demo"])):
                offset = robustness_rng.uniform(
                    -np.asarray(robustness["object_position_noise_m"], dtype=float),
                    np.asarray(robustness["object_position_noise_m"], dtype=float),
                )
                mass_scale = robustness_rng.uniform(*robustness["object_mass_scale_range"])
                friction_scale = robustness_rng.uniform(*robustness["friction_scale_range"])
                episode_specs.append(
                    {
                        "scene_mjb": str(scene_mjb),
                        "config": config,
                        "robot_qpos": robot_qpos,
                        "expected_contact": expected_contact,
                        "initial_offset_m": offset,
                        "mass_scale": float(mass_scale),
                        "friction_scale": float(friction_scale),
                        "metadata": {
                            "motion_name": name,
                            "quality_level": level["name"],
                            "severity": severity,
                            "seed": int(seed),
                            "episode": episode,
                            "initial_offset_x_m": offset[0],
                            "initial_offset_y_m": offset[1],
                            "initial_offset_z_m": offset[2],
                            "mass_scale": mass_scale,
                            "friction_scale": friction_scale,
                        },
                    }
                )
    if progress:
        progress(
            f"Running {len(episode_specs)} randomized physics episodes with "
            f"{robustness['parallel_workers']} workers"
        )
    with ProcessPoolExecutor(max_workers=int(robustness["parallel_workers"])) as executor:
        episode_rows = list(executor.map(_robust_episode_worker, episode_specs, chunksize=1))
    catalog = pd.DataFrame(catalog_rows)
    metrics = pd.DataFrame(metric_rows)
    episodes = pd.DataFrame(episode_rows)
    catalog.to_csv(output / "interaction_catalog.csv", index=False)
    metrics.to_csv(output / "offline_interaction_metrics.csv", index=False)
    episodes.to_csv(output / "robustness_episodes.csv", index=False)
    robust_summary = (
        episodes.groupby("motion_name", as_index=False)
        .agg(
            robust_task_success_rate=("success", "mean"),
            robust_final_position_error_mean_m=("final_position_error_m", "mean"),
            robust_final_speed_mean_m_s=("final_speed_m_s", "mean"),
            robust_max_lift_mean_m=("max_lift_m", "mean"),
            robustness_episode_count=("episode", "count"),
        )
    )
    joined = metrics.merge(robust_summary, on="motion_name", validate="one_to_one")
    joined.to_csv(output / "interaction_learning_value.csv", index=False)
    group_rows = []
    for level, group in joined.groupby("quality_level", sort=False):
        row: dict[str, Any] = {"quality_level": level, "demo_count": len(group)}
        for metric in (
            "interaction_quality_score", "robust_task_success_rate",
            "robust_final_position_error_mean_m", "robust_max_lift_mean_m",
        ):
            values = group[metric].to_numpy(dtype=float)
            low, high = _ci95(values)
            row.update({f"{metric}_mean": values.mean(), f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        group_rows.append(row)
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(output / "quality_level_summary.csv", index=False)
    targets = {
        "robust_task_success_rate": joined.robust_task_success_rate.to_numpy(dtype=float),
        "negative_final_position_error": -joined.robust_final_position_error_mean_m.to_numpy(dtype=float),
        "robust_max_lift": joined.robust_max_lift_mean_m.to_numpy(dtype=float),
    }
    offline_dimensions = [
        "hand_object_proximity_score", "contact_timing_score", "relative_pose_score",
        "penetration_score", "grasp_stability_score", "task_effectiveness_score",
        "interaction_quality_score",
    ]
    correlation_rows = []
    for offline_metric in offline_dimensions:
        x = joined[offline_metric].to_numpy(dtype=float)
        for target_name, target in targets.items():
            rho = float(spearmanr(x, target).statistic)
            low, high = _bootstrap_spearman(
                x, target, int(robustness["bootstrap_resamples"]), int(robustness["bootstrap_seed"])
            )
            correlation_rows.append(
                {
                    "offline_metric": offline_metric,
                    "simulation_target": target_name,
                    "spearman_rho": rho,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "n_demonstrations": len(joined),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output / "offline_simulation_correlations.csv", index=False)
    primary = correlations[
        (correlations.offline_metric == config["calibration"]["primary_offline_metric"])
        & (correlations.simulation_target == config["calibration"]["primary_target"])
    ].iloc[0]
    finite = all(np.isfinite([primary.spearman_rho, primary.bootstrap_ci95_low, primary.bootstrap_ci95_high]))
    gate = bool(finite and primary.spearman_rho > 0 and primary.bootstrap_ci95_low > 0)
    calibration = {
        "score_is_calibrated": gate,
        "gate": "Spearman > 0 and paired-bootstrap 95% CI lower bound > 0",
        "spearman_rho": float(primary.spearman_rho) if finite else None,
        "bootstrap_ci95": [
            float(primary.bootstrap_ci95_low) if finite else None,
            float(primary.bootstrap_ci95_high) if finite else None,
        ],
        "outcome": "interaction_v1_supported_on_controlled_benchmark" if gate else config["calibration"]["on_gate_failure"],
        "external_human_object_validation": False,
    }
    (output / "calibration_decision.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Step 12 controlled interaction benchmark",
        "",
        f"- Demonstrations: {len(demos)}",
        f"- Randomized physics episodes: {len(episodes)}",
        f"- Task: `{config['task']['label']}`",
        f"- Calibration supported on this benchmark: `{str(gate).lower()}`",
        "- Human-object external validation: `false`",
        "",
        "| Level | Interaction score | Robust success | Final error (m) | Max lift (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in group_summary.iterrows():
        report_lines.append(
            f"| {row.quality_level} | {row.interaction_quality_score_mean:.3f} | "
            f"{row.robust_task_success_rate_mean:.3f} | "
            f"{row.robust_final_position_error_mean_m_mean:.3f} | "
            f"{row.robust_max_lift_mean_m_mean:.3f} |"
        )
    report_lines.extend(
        [
            "",
            f"Primary Spearman: {calibration['spearman_rho']}; 95% CI: {calibration['bootstrap_ci95']}.",
            "",
            "This is a controlled synthetic G1 benchmark with dynamic object physics. "
            "It validates metric behavior and simulation linkage, but it is not evidence on MotionDecode "
            "or another human-captured object dataset.",
            "",
        ]
    )
    report_path = output / "STEP12_RESULTS.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    critical = [
        config_path, resolved_config, scene_xml, scene_mjb, output / "interaction_catalog.csv",
        output / "offline_interaction_metrics.csv", output / "robustness_episodes.csv",
        output / "interaction_learning_value.csv", output / "quality_level_summary.csv",
        output / "offline_simulation_correlations.csv", output / "calibration_decision.json",
        report_path, *demos,
    ]
    manifest = {
        "manifest_version": "step12_final_v1",
        "complete": True,
        "benchmark_scope": "controlled_synthetic_g1_interaction",
        "human_object_external_validation": False,
        "demonstration_count": len(demos),
        "robustness_episode_count": len(episodes),
        "score_is_calibrated": gate,
        "calibration": calibration,
        "files": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(critical)
        ],
    }
    manifest_path = output / "step12_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return Step12Result(output, len(demos), len(episodes), manifest_path, gate)
