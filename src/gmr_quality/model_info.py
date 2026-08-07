from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .kinematic_metrics import JointLimits
from .types import MotionRecord


class ModelValidationError(ValueError):
    """Raised when the configured MuJoCo model violates the expected GMR layout."""


class JointOrderError(ValueError):
    """Raised when a motion cannot be safely bound to the model joint order."""


@dataclass(frozen=True)
class RobotModelInfo:
    xml_path: Path
    xml_sha256: str
    nq: int
    nv: int
    nbody: int
    ngeom: int
    root_joint_id: int
    root_joint_name: str
    root_qpos_address: int
    root_dof_address: int
    joint_ids: tuple[int, ...]
    joint_names: tuple[str, ...]
    joint_qpos_addresses: tuple[int, ...]
    joint_dof_addresses: tuple[int, ...]
    joint_limits: JointLimits
    body_names: tuple[str, ...]
    body_parent_ids: tuple[int, ...]
    geom_names: tuple[str | None, ...]
    geom_body_ids: tuple[int, ...]
    geom_types: tuple[int, ...]
    geom_sizes: tuple[tuple[float, float, float], ...]
    geom_contype: tuple[int, ...]
    geom_conaffinity: tuple[int, ...]
    floor_geom_id: int
    left_sole_geom_ids: tuple[int, ...]
    right_sole_geom_ids: tuple[int, ...]

    @property
    def num_dofs(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class LoadedRobotModel:
    model: mujoco.MjModel
    info: RobotModelInfo


@dataclass(frozen=True)
class MotionModelBinding:
    joint_order_verified: bool
    joint_order_method: str
    explicit_joint_names_verified: bool
    evaluation_model_identity_verified: bool
    source_model_provenance_verified: bool
    model_xml_sha256: str
    joint_names: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BoundMotion:
    record: MotionRecord
    binding: MotionModelBinding
    joint_limits: JointLimits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> str | None:
    return mujoco.mj_id2name(model, object_type, object_id)


def _require_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
    label: str,
) -> str:
    name = _name(model, object_type, object_id)
    if not name:
        raise ModelValidationError(f"{label} {object_id} has no name")
    return name


def _find_sole_geoms(
    model: mujoco.MjModel,
    body_name: str,
    max_radius_m: float,
    expected_count: int,
) -> tuple[int, ...]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ModelValidationError(f"Foot body does not exist: {body_name}")
    geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        and float(model.geom_size[geom_id, 0]) <= max_radius_m
    )
    if len(geom_ids) != expected_count:
        raise ModelValidationError(
            f"Expected {expected_count} sole geoms under {body_name}, found {len(geom_ids)}"
        )
    return geom_ids


def load_robot_model(
    xml_path: str | Path,
    *,
    expected_xml_sha256: str,
    expected_dof_count: int,
    expected_nq: int,
    expected_nv: int,
    root_qpos_width: int,
    root_joint_name: str,
    floor_geom_name: str,
    left_foot_body: str,
    right_foot_body: str,
    sole_sphere_max_radius_m: float,
    expected_sole_geom_count_per_foot: int,
) -> LoadedRobotModel:
    """Load and strictly validate the G1 free-root plus hinge-joint layout."""
    path = Path(xml_path).expanduser().resolve()
    if not path.is_file():
        raise ModelValidationError(f"Robot XML does not exist: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_xml_sha256:
        raise ModelValidationError(
            "Robot XML SHA-256 mismatch: "
            f"expected {expected_xml_sha256}, got {actual_hash}"
        )

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise ModelValidationError(
            f"MuJoCo could not load {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if model.nq != expected_nq or model.nv != expected_nv:
        raise ModelValidationError(
            f"Model dimensions mismatch: nq={model.nq}, nv={model.nv}; "
            f"expected nq={expected_nq}, nv={expected_nv}"
        )

    free_joint_ids = tuple(
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if len(free_joint_ids) != 1:
        raise ModelValidationError(
            f"Expected exactly one free root joint, found {len(free_joint_ids)}"
        )
    root_joint_id = free_joint_ids[0]
    actual_root_name = _require_name(
        model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_id, "Root joint"
    )
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])
    root_dof_address = int(model.jnt_dofadr[root_joint_id])
    if actual_root_name != root_joint_name:
        raise ModelValidationError(
            f"Root joint is {actual_root_name}, expected {root_joint_name}"
        )
    if root_qpos_address != 0 or root_dof_address != 0:
        raise ModelValidationError(
            "Free root must start at qpos=0 and dof=0, got "
            f"qpos={root_qpos_address}, dof={root_dof_address}"
        )

    unsupported = tuple(
        joint_id
        for joint_id in range(model.njnt)
        if joint_id != root_joint_id
        and int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE)
    )
    if unsupported:
        raise ModelValidationError(
            f"Non-root joints must all be hinge joints; unsupported ids={unsupported}"
        )
    joint_ids = tuple(
        sorted(
            (
                joint_id
                for joint_id in range(model.njnt)
                if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
            ),
            key=lambda joint_id: int(model.jnt_qposadr[joint_id]),
        )
    )
    if len(joint_ids) != expected_dof_count:
        raise ModelValidationError(
            f"Expected {expected_dof_count} hinge DoFs, found {len(joint_ids)}"
        )
    if any(not bool(model.jnt_limited[joint_id]) for joint_id in joint_ids):
        raise ModelValidationError("Every evaluated hinge joint must have a finite range")

    qpos_addresses = tuple(int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids)
    expected_qpos_addresses = tuple(
        range(root_qpos_width, root_qpos_width + expected_dof_count)
    )
    if qpos_addresses != expected_qpos_addresses:
        raise ModelValidationError(
            f"Hinge qpos addresses are {qpos_addresses}, expected {expected_qpos_addresses}"
        )
    dof_addresses = tuple(int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids)
    expected_dof_addresses = tuple(range(6, 6 + expected_dof_count))
    if dof_addresses != expected_dof_addresses:
        raise ModelValidationError(
            f"Hinge DoF addresses are {dof_addresses}, expected {expected_dof_addresses}"
        )

    joint_names = tuple(
        _require_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, "Hinge joint")
        for joint_id in joint_ids
    )
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    joint_limits = JointLimits(lower, upper, joint_names)

    floor_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name
    )
    if floor_geom_id < 0:
        raise ModelValidationError(f"Floor geom does not exist: {floor_geom_name}")
    left_sole = _find_sole_geoms(
        model,
        left_foot_body,
        sole_sphere_max_radius_m,
        expected_sole_geom_count_per_foot,
    )
    right_sole = _find_sole_geoms(
        model,
        right_foot_body,
        sole_sphere_max_radius_m,
        expected_sole_geom_count_per_foot,
    )

    body_names = tuple(
        _require_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "Body")
        for body_id in range(model.nbody)
    )
    info = RobotModelInfo(
        xml_path=path,
        xml_sha256=actual_hash,
        nq=int(model.nq),
        nv=int(model.nv),
        nbody=int(model.nbody),
        ngeom=int(model.ngeom),
        root_joint_id=root_joint_id,
        root_joint_name=actual_root_name,
        root_qpos_address=root_qpos_address,
        root_dof_address=root_dof_address,
        joint_ids=joint_ids,
        joint_names=joint_names,
        joint_qpos_addresses=qpos_addresses,
        joint_dof_addresses=dof_addresses,
        joint_limits=joint_limits,
        body_names=body_names,
        body_parent_ids=tuple(int(value) for value in model.body_parentid),
        geom_names=tuple(
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(model.ngeom)
        ),
        geom_body_ids=tuple(int(value) for value in model.geom_bodyid),
        geom_types=tuple(int(value) for value in model.geom_type),
        geom_sizes=tuple(tuple(float(value) for value in row) for row in model.geom_size),
        geom_contype=tuple(int(value) for value in model.geom_contype),
        geom_conaffinity=tuple(int(value) for value in model.geom_conaffinity),
        floor_geom_id=int(floor_geom_id),
        left_sole_geom_ids=left_sole,
        right_sole_geom_ids=right_sole,
    )
    return LoadedRobotModel(model=model, info=info)


def load_robot_model_from_project_config(config: dict[str, Any]) -> LoadedRobotModel:
    doctor = config["doctor"]
    robot = config["robot"]
    return load_robot_model(
        config["paths"]["robot_xml"],
        expected_xml_sha256=str(robot["expected_xml_sha256"]),
        expected_dof_count=int(doctor["expected_dof_count"]),
        expected_nq=int(doctor["expected_nq"]),
        expected_nv=int(doctor["expected_nv"]),
        root_qpos_width=int(doctor["root_qpos_width"]),
        root_joint_name=str(robot["root_joint_name"]),
        floor_geom_name=str(robot["floor_geom_name"]),
        left_foot_body=str(robot["left_foot_body"]),
        right_foot_body=str(robot["right_foot_body"]),
        sole_sphere_max_radius_m=float(doctor["sole_sphere_max_radius_m"]),
        expected_sole_geom_count_per_foot=int(
            doctor["expected_sole_geom_count_per_foot"]
        ),
    )


def bind_motion_to_model(
    motion: MotionRecord,
    model_info: RobotModelInfo,
) -> BoundMotion:
    """Bind a validated motion to the exact qpos order of an evaluated model."""
    if motion.num_dofs != model_info.num_dofs:
        raise JointOrderError(
            f"Motion has {motion.num_dofs} DoFs, model has {model_info.num_dofs}"
        )

    explicit = motion.joint_names is not None
    warnings: tuple[str, ...]
    if explicit:
        if motion.joint_names != model_info.joint_names:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(motion.joint_names, model_info.joint_names)
                    )
                    if actual != expected
                ),
                None,
            )
            raise JointOrderError(
                "Explicit dof_names do not match model qpos order"
                + (f" at index {mismatch}" if mismatch is not None else "")
            )
        method = "explicit_dof_names"
        warnings = ("SOURCE_MODEL_PROVENANCE_UNAVAILABLE",)
    else:
        if (
            motion.format_name != "gmr_pickle_v1"
            or motion.joint_order_source != "implicit_gmr_qpos_order"
        ):
            raise JointOrderError(
                "Unnamed joints can only be bound under the documented GMR qpos[7:] convention"
            )
        method = "gmr_qpos_after_free_root"
        warnings = (
            "JOINT_ORDER_INFERRED_FROM_GMR_QPOS_LAYOUT",
            "SOURCE_MODEL_PROVENANCE_UNAVAILABLE",
        )

    binding = MotionModelBinding(
        joint_order_verified=True,
        joint_order_method=method,
        explicit_joint_names_verified=explicit,
        evaluation_model_identity_verified=True,
        source_model_provenance_verified=False,
        model_xml_sha256=model_info.xml_sha256,
        joint_names=model_info.joint_names,
        warnings=warnings,
    )
    bound_record = replace(
        motion,
        joint_names=model_info.joint_names,
        joint_order_source=f"verified:{method}",
    )
    return BoundMotion(
        record=bound_record,
        binding=binding,
        joint_limits=model_info.joint_limits,
    )
