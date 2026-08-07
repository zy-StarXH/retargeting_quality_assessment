from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gmr_quality.config import load_config
from gmr_quality.io import load_motion_from_project_config, validate_motion_payload
from gmr_quality.mujoco_kinematics import MujocoKinematics, xyzw_to_wxyz
from gmr_quality.types import MotionRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "unitree_g1.yaml")
KNOWN_SAMPLE = Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "g1_dance1_subject2.pkl"


@pytest.fixture(scope="module")
def adapter() -> MujocoKinematics:
    return MujocoKinematics.from_project_config(CONFIG)


def make_pose_motion(joint_names: tuple[str, ...]) -> MotionRecord:
    frames = 4
    root_pos = np.array(
        [
            [0.10, -0.20, 1.00],
            [0.11, -0.20, 1.00],
            [0.12, -0.20, 1.00],
            [0.13, -0.20, 1.00],
        ],
        dtype=np.float64,
    )
    root_quat = np.broadcast_to(
        Rotation.from_euler("z", 90.0, degrees=True).as_quat(),
        (frames, 4),
    ).copy()
    joint_pos = np.zeros((frames, len(joint_names)), dtype=np.float64)
    joint_pos[:, 0] = np.arange(frames, dtype=np.float64) * 0.1
    return validate_motion_payload(
        {
            "fps": 20.0,
            "root_pos": root_pos,
            "root_rot": root_quat,
            "dof_pos": joint_pos,
            "dof_names": joint_names,
        },
        "pose_synthetic.pkl",
    ).require_record()


def test_xyzw_to_wxyz_is_explicit_readonly_and_non_mutating() -> None:
    source = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    before = source.copy()

    converted = xyzw_to_wxyz(source)

    np.testing.assert_array_equal(converted, [[4.0, 1.0, 2.0, 3.0], [-4.0, -1.0, -2.0, -3.0]])
    np.testing.assert_array_equal(source, before)
    assert converted.flags.writeable is False
    with pytest.raises(ValueError, match="last dimension must be 4"):
        xyzw_to_wxyz(np.zeros((2, 3)))


def test_qpos_mapping_and_forward_pose_are_correct(adapter: MujocoKinematics) -> None:
    motion = make_pose_motion(adapter.info.joint_names)
    bound = adapter.bind_motion(motion)
    qpos = adapter.qpos_for_frame(bound, 2)

    np.testing.assert_allclose(qpos[:3], motion.root_pos[2], atol=0.0)
    np.testing.assert_allclose(qpos[3:7], xyzw_to_wxyz(motion.root_quat_xyzw[2]), atol=1e-15)
    np.testing.assert_allclose(
        qpos[np.asarray(adapter.info.joint_qpos_addresses)],
        motion.joint_pos[2],
        atol=0.0,
    )
    assert qpos.flags.writeable is False

    frame = adapter.forward_frame(bound, 2)
    pelvis_body_id = adapter.info.body_names.index("pelvis")
    np.testing.assert_allclose(frame.body_pos_world_m[pelvis_body_id], motion.root_pos[2], atol=1e-12)
    actual_rotation = Rotation.from_quat(
        frame.body_quat_wxyz[pelvis_body_id], scalar_first=True
    ).as_matrix()
    expected_rotation = Rotation.from_quat(motion.root_quat_xyzw[2]).as_matrix()
    np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-12)
    assert frame.frame == 2
    assert frame.motion_time_s == pytest.approx(0.1)
    assert frame.mujoco_time_s == 0.0
    assert frame.body_pos_world_m.shape == (adapter.info.nbody, 3)
    assert frame.geom_mat_world.shape == (adapter.info.ngeom, 3, 3)
    assert frame.qpos.flags.writeable is False
    assert frame.body_pos_world_m.flags.writeable is False
    assert frame.geom_pos_world_m.flags.writeable is False


def test_frame_bounds_and_iterator_interval_validation(adapter: MujocoKinematics) -> None:
    bound = adapter.bind_motion(make_pose_motion(adapter.info.joint_names))

    assert [frame.frame for frame in adapter.iter_forward(bound, start=1, end=3)] == [1, 2]
    with pytest.raises(IndexError, match="outside valid range"):
        adapter.forward_frame(bound, 4)
    with pytest.raises(ValueError, match="Invalid frame interval"):
        tuple(adapter.iter_forward(bound, start=-1))
    with pytest.raises(ValueError, match="Invalid frame interval"):
        tuple(adapter.iter_forward(bound, start=3, end=2))

    wrong_model_binding = replace(
        bound,
        binding=replace(bound.binding, model_xml_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="different robot XML"):
        adapter.forward_frame(wrong_model_binding, 0)


def test_all_real_sample_frames_run_forward_kinematics_with_meter_scale_feet(
    adapter: MujocoKinematics,
) -> None:
    motion = load_motion_from_project_config(KNOWN_SAMPLE, CONFIG).require_record()
    bound = adapter.bind_motion(motion)
    sole_ids = np.asarray(
        adapter.info.left_sole_geom_ids + adapter.info.right_sole_geom_ids,
        dtype=np.int64,
    )
    frame_count = 0
    min_sole_z_m = np.inf
    max_sole_z_m = -np.inf

    for frame in adapter.iter_forward(bound):
        assert np.isfinite(frame.qpos).all()
        assert np.isfinite(frame.body_pos_world_m).all()
        assert np.isfinite(frame.body_quat_wxyz).all()
        assert np.isfinite(frame.geom_pos_world_m).all()
        assert frame.mujoco_time_s == 0.0
        sole_z_m = frame.geom_pos_world_m[sole_ids, 2]
        min_sole_z_m = min(min_sole_z_m, float(sole_z_m.min()))
        max_sole_z_m = max(max_sole_z_m, float(sole_z_m.max()))
        frame_count += 1

    assert frame_count == motion.num_frames == 3944
    assert -0.20 < min_sole_z_m < 0.20
    assert 0.0 < max_sole_z_m < 2.0
