from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import mujoco
import numpy as np
from numpy.typing import NDArray

from .model_info import (
    BoundMotion,
    LoadedRobotModel,
    RobotModelInfo,
    bind_motion_to_model,
    load_robot_model_from_project_config,
)
from .types import MotionRecord


@dataclass(frozen=True)
class RawContact:
    geom1: int
    geom2: int
    distance_m: float
    position_world_m: NDArray[np.float64]
    contact_frame_world: NDArray[np.float64]


@dataclass(frozen=True)
class ForwardKinematicsFrame:
    frame: int
    motion_time_s: float
    mujoco_time_s: float
    qpos: NDArray[np.float64]
    body_pos_world_m: NDArray[np.float64]
    body_quat_wxyz: NDArray[np.float64]
    geom_pos_world_m: NDArray[np.float64]
    geom_mat_world: NDArray[np.float64]
    contacts: tuple[RawContact, ...]


def _readonly_copy(value: np.ndarray) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def xyzw_to_wxyz(quaternion: np.ndarray) -> NDArray[np.float64]:
    """Return a non-mutating explicit quaternion-order conversion."""
    source = np.asarray(quaternion)
    if source.shape[-1:] != (4,):
        raise ValueError(f"Quaternion last dimension must be 4, got {source.shape}")
    converted = np.asarray(source[..., [3, 0, 1, 2]], dtype=np.float64)
    return _readonly_copy(converted)


class MujocoKinematics:
    """State-isolated MuJoCo forward kinematics for validated reference poses."""

    def __init__(self, loaded_model: LoadedRobotModel) -> None:
        self.model = loaded_model.model
        self.info: RobotModelInfo = loaded_model.info
        self.data = mujoco.MjData(self.model)

    @classmethod
    def from_project_config(cls, config: dict) -> "MujocoKinematics":
        return cls(load_robot_model_from_project_config(config))

    def bind_motion(self, motion: MotionRecord) -> BoundMotion:
        return bind_motion_to_model(motion, self.info)

    def qpos_for_frame(
        self,
        bound_motion: BoundMotion,
        frame: int,
    ) -> NDArray[np.float64]:
        self._validate_binding(bound_motion)
        motion = bound_motion.record
        if frame < 0 or frame >= motion.num_frames:
            raise IndexError(
                f"Frame {frame} is outside valid range [0, {motion.num_frames})"
            )
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        root_address = self.info.root_qpos_address
        qpos[root_address : root_address + 3] = motion.root_pos[frame]
        qpos[root_address + 3 : root_address + 7] = xyzw_to_wxyz(
            motion.root_quat_xyzw[frame]
        )
        qpos[np.asarray(self.info.joint_qpos_addresses, dtype=np.int64)] = (
            motion.joint_pos[frame]
        )
        qpos.setflags(write=False)
        return qpos

    def forward_frame(
        self,
        bound_motion: BoundMotion,
        frame: int,
    ) -> ForwardKinematicsFrame:
        qpos = self.qpos_for_frame(bound_motion, frame)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        contacts = tuple(
            RawContact(
                geom1=int(contact.geom1),
                geom2=int(contact.geom2),
                distance_m=float(contact.dist),
                position_world_m=_readonly_copy(contact.pos),
                contact_frame_world=_readonly_copy(contact.frame.reshape(3, 3)),
            )
            for contact in self.data.contact[: self.data.ncon]
        )
        motion = bound_motion.record
        return ForwardKinematicsFrame(
            frame=frame,
            motion_time_s=frame / motion.fps,
            mujoco_time_s=float(self.data.time),
            qpos=_readonly_copy(self.data.qpos),
            body_pos_world_m=_readonly_copy(self.data.xpos),
            body_quat_wxyz=_readonly_copy(self.data.xquat),
            geom_pos_world_m=_readonly_copy(self.data.geom_xpos),
            geom_mat_world=_readonly_copy(
                self.data.geom_xmat.reshape(self.model.ngeom, 3, 3)
            ),
            contacts=contacts,
        )

    def iter_forward(
        self,
        bound_motion: BoundMotion,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[ForwardKinematicsFrame]:
        stop = bound_motion.record.num_frames if end is None else end
        if start < 0 or stop < start or stop > bound_motion.record.num_frames:
            raise ValueError(
                f"Invalid frame interval [{start}, {stop}) for "
                f"{bound_motion.record.num_frames} frames"
            )
        for frame in range(start, stop):
            yield self.forward_frame(bound_motion, frame)

    def _validate_binding(self, bound_motion: BoundMotion) -> None:
        if not bound_motion.binding.joint_order_verified:
            raise ValueError("Motion joint order has not been verified")
        if bound_motion.binding.model_xml_sha256 != self.info.xml_sha256:
            raise ValueError("Bound motion belongs to a different robot XML")
