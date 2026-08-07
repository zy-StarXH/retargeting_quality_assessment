from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from gmr_quality.config import load_config
from gmr_quality.io import load_motion_from_project_config, validate_motion_payload
from gmr_quality.kinematic_metrics import compute_smoothness_metrics
from gmr_quality.model_info import (
    JointOrderError,
    ModelValidationError,
    bind_motion_to_model,
    load_robot_model_from_project_config,
)
from gmr_quality.types import MotionRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "unitree_g1.yaml")
KNOWN_SAMPLE = Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "g1_dance1_subject2.pkl"
EXPECTED_XML_SHA256 = "85a25128d42ad3cc8dd071c089806a2773a59ee0689edcc646aeb0201ae8f4c8"


@pytest.fixture(scope="module")
def model_info():
    return load_robot_model_from_project_config(CONFIG).info


def make_named_motion(joint_names: tuple[str, ...]) -> MotionRecord:
    root_quat = np.zeros((4, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    result = validate_motion_payload(
        {
            "fps": 30.0,
            "root_pos": np.zeros((4, 3), dtype=np.float64),
            "root_rot": root_quat,
            "dof_pos": np.zeros((4, len(joint_names)), dtype=np.float64),
            "dof_names": joint_names,
        },
        "named_synthetic.pkl",
    )
    return result.require_record()


def test_parses_and_locks_the_exact_g1_model_layout(model_info) -> None:
    info = model_info

    assert info.xml_sha256 == EXPECTED_XML_SHA256
    assert (info.nq, info.nv, info.nbody, info.ngeom) == (36, 35, 39, 72)
    assert info.num_dofs == 29
    assert info.root_joint_name == "pelvis"
    assert (info.root_joint_id, info.root_qpos_address, info.root_dof_address) == (0, 0, 0)
    assert info.joint_qpos_addresses == tuple(range(7, 36))
    assert info.joint_dof_addresses == tuple(range(6, 35))
    assert info.joint_names[0] == "left_hip_pitch_joint"
    assert info.joint_names[3] == "left_knee_joint"
    assert info.joint_names[17] == "left_shoulder_yaw_joint"
    assert info.joint_names[-1] == "right_wrist_yaw_joint"
    assert info.joint_limits.joint_names == info.joint_names
    assert info.joint_limits.lower_rad[3] == pytest.approx(-0.087267)
    assert info.joint_limits.upper_rad[3] == pytest.approx(2.8798)


def test_identifies_floor_and_the_eight_configured_sole_spheres(model_info) -> None:
    info = model_info

    assert info.floor_geom_id == 0
    assert info.geom_names[info.floor_geom_id] == "floor"
    assert info.left_sole_geom_ids == (15, 16, 17, 18)
    assert info.right_sole_geom_ids == (30, 31, 32, 33)
    for geom_id in info.left_sole_geom_ids + info.right_sole_geom_ids:
        assert info.geom_sizes[geom_id][0] == pytest.approx(0.005)


def test_model_hash_mismatch_fails_before_model_use() -> None:
    changed = copy.deepcopy(CONFIG)
    changed["robot"]["expected_xml_sha256"] = "0" * 64

    with pytest.raises(ModelValidationError, match="SHA-256 mismatch"):
        load_robot_model_from_project_config(changed)


def test_binds_real_unnamed_gmr_motion_to_xml_qpos_order(model_info) -> None:
    motion = load_motion_from_project_config(KNOWN_SAMPLE, CONFIG).require_record()
    bound = bind_motion_to_model(motion, model_info)

    assert bound.record.joint_pos is motion.joint_pos
    assert bound.record.joint_names == model_info.joint_names
    assert bound.record.joint_order_source == "verified:gmr_qpos_after_free_root"
    assert bound.binding.joint_order_verified is True
    assert bound.binding.joint_order_method == "gmr_qpos_after_free_root"
    assert bound.binding.explicit_joint_names_verified is False
    assert bound.binding.evaluation_model_identity_verified is True
    assert bound.binding.source_model_provenance_verified is False
    assert bound.binding.model_xml_sha256 == EXPECTED_XML_SHA256
    assert bound.binding.warnings == (
        "JOINT_ORDER_INFERRED_FROM_GMR_QPOS_LAYOUT",
        "SOURCE_MODEL_PROVENANCE_UNAVAILABLE",
    )
    assert bound.joint_limits is model_info.joint_limits

    smoothness = compute_smoothness_metrics(bound.record).summary
    assert smoothness.worst_step_joint == "left_shoulder_yaw_joint"


def test_explicit_names_must_exactly_match_model_qpos_order(model_info) -> None:
    correct = make_named_motion(model_info.joint_names)
    bound = bind_motion_to_model(correct, model_info)

    assert bound.binding.joint_order_method == "explicit_dof_names"
    assert bound.binding.explicit_joint_names_verified is True
    assert bound.binding.warnings == ("SOURCE_MODEL_PROVENANCE_UNAVAILABLE",)

    wrong_names = list(model_info.joint_names)
    wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
    wrong = make_named_motion(tuple(wrong_names))
    with pytest.raises(JointOrderError, match="index 0"):
        bind_motion_to_model(wrong, model_info)
