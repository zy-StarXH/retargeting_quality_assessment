from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmr_quality.config import load_config
from gmr_quality.contact_metrics import (
    ContactEvaluationConfig,
    ContactThresholds,
    compute_contact_metrics,
    compute_contact_metrics_from_frames,
)
from gmr_quality.io import load_motion_from_project_config, validate_motion_payload
from gmr_quality.mujoco_kinematics import (
    ForwardKinematicsFrame,
    MujocoKinematics,
    RawContact,
)
from gmr_quality.types import MotionRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "unitree_g1.yaml")
KNOWN_SAMPLE = Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "g1_dance1_subject2.pkl"


@pytest.fixture(scope="module")
def adapter() -> MujocoKinematics:
    return MujocoKinematics.from_project_config(CONFIG)


def make_motion(joint_names: tuple[str, ...], frames: int = 5) -> MotionRecord:
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    return validate_motion_payload(
        {
            "fps": 10.0,
            "root_pos": np.zeros((frames, 3), dtype=np.float64),
            "root_rot": root_quat,
            "dof_pos": np.zeros((frames, len(joint_names)), dtype=np.float64),
            "dof_names": joint_names,
        },
        "contact_synthetic.pkl",
    ).require_record()


def raw_contact(geom1: int, geom2: int, distance_m: float) -> RawContact:
    return RawContact(
        geom1=geom1,
        geom2=geom2,
        distance_m=distance_m,
        position_world_m=np.zeros(3, dtype=np.float64),
        contact_frame_world=np.eye(3, dtype=np.float64),
    )


def synthetic_frames(
    adapter: MujocoKinematics,
) -> tuple[ForwardKinematicsFrame, ...]:
    info = adapter.info
    floor = info.floor_geom_id
    left_sole = info.left_sole_geom_ids[0]
    right_sole = info.right_sole_geom_ids[0]
    contacts = (
        (),
        (
            raw_contact(floor, left_sole, -0.005),
            raw_contact(9, 55, -0.020),
        ),
        (
            raw_contact(floor, left_sole, -0.005),
            raw_contact(floor, 3, -0.030),
            raw_contact(11, 26, -0.040),
        ),
        (
            raw_contact(floor, right_sole, -0.005),
            raw_contact(11, 26, -0.060),
        ),
        (),
    )
    left_x = (0.0, 0.0, 0.02, 0.04, 0.04)
    result = []
    for frame in range(5):
        geom_pos = np.zeros((info.ngeom, 3), dtype=np.float64)
        for geom_id in info.left_sole_geom_ids:
            geom_pos[geom_id] = (left_x[frame], 0.0, 0.005)
        for geom_id in info.right_sole_geom_ids:
            geom_pos[geom_id] = (0.0, 0.0, 0.005)
        body_quat = np.zeros((info.nbody, 4), dtype=np.float64)
        body_quat[:, 0] = 1.0
        result.append(
            ForwardKinematicsFrame(
                frame=frame,
                motion_time_s=frame / 10.0,
                mujoco_time_s=0.0,
                qpos=np.zeros(info.nq, dtype=np.float64),
                body_pos_world_m=np.zeros((info.nbody, 3), dtype=np.float64),
                body_quat_wxyz=body_quat,
                geom_pos_world_m=geom_pos,
                geom_mat_world=np.broadcast_to(
                    np.eye(3, dtype=np.float64),
                    (info.ngeom, 3, 3),
                ).copy(),
                contacts=contacts[frame],
            )
        )
    return tuple(result)


def test_contact_configuration_comes_from_named_policy() -> None:
    feet_only = ContactEvaluationConfig.from_project_config(CONFIG, "feet_only")
    descriptive = ContactEvaluationConfig.from_project_config(
        CONFIG, "descriptive_only"
    )

    assert feet_only.thresholds.penetration_m == pytest.approx(0.01)
    assert feet_only.thresholds.foot_slide_speed_m_s == pytest.approx(0.1)
    assert feet_only.policy.undesired_ground_contact_is_failure is True
    assert descriptive.policy.undesired_ground_contact_is_failure is False
    assert "left_wrist" in feet_only.allowed_hand_contact_patterns
    with pytest.raises(ValueError, match="Unknown contact policy"):
        ContactEvaluationConfig.from_project_config(CONFIG, "not_a_policy")
    with pytest.raises(ValueError, match="at least penetration_m"):
        ContactThresholds(penetration_m=0.05, severe_penetration_m=0.01)


def test_classifies_self_ground_hand_and_penetration_contacts(
    adapter: MujocoKinematics,
) -> None:
    motion = make_motion(adapter.info.joint_names)
    metrics = compute_contact_metrics_from_frames(
        motion,
        adapter.info,
        synthetic_frames(adapter),
        ContactEvaluationConfig.from_project_config(CONFIG, "feet_only"),
        motion_category="dance",
    )
    summary = metrics.summary

    np.testing.assert_array_equal(
        metrics.self_collision_any_frame,
        [False, True, True, True, False],
    )
    np.testing.assert_array_equal(
        metrics.self_collision_nonhand_frame,
        [False, False, True, True, False],
    )
    assert metrics.self_collision_pairs[1] == (
        ("left_hip_yaw_link", "left_wrist_yaw_link"),
    )
    assert metrics.self_collision_nonhand_pairs[1] == ()
    assert summary.self_collision_event_count == 2
    assert summary.self_collision_nonhand_event_count == 1
    assert summary.first_self_collision_frame == 1
    assert summary.max_self_collision_penetration_m == pytest.approx(0.06)
    assert summary.worst_self_collision_body_pair == (
        "left_knee_link",
        "right_knee_link",
    )
    assert summary.worst_self_collision_frame == 3

    np.testing.assert_array_equal(
        metrics.ground_contact_frame,
        [False, True, True, True, False],
    )
    np.testing.assert_array_equal(
        metrics.ground_penetration_frame,
        [False, False, True, False, False],
    )
    np.testing.assert_array_equal(
        metrics.severe_ground_penetration_frame,
        [False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        metrics.undesired_ground_contact_frame,
        [False, False, True, False, False],
    )
    assert metrics.undesired_ground_contact_bodies[2] == ("pelvis",)
    assert summary.ground_penetration_event_count == 1
    assert summary.mean_ground_penetration_m == pytest.approx(0.01125)
    assert summary.max_ground_penetration_m == pytest.approx(0.03)
    assert summary.worst_penetration_body == "pelvis"
    assert summary.worst_penetration_frame == 2
    assert summary.undesired_ground_contact_event_count == 1
    assert summary.undesired_ground_contact_bodies == ("pelvis",)
    assert summary.motion_category == "dance"
    assert summary.contact_policy == "feet_only"
    assert summary.undesired_ground_contact_is_failure is True


def test_sole_surface_proxy_slide_and_airborne_formulas_are_frame_aligned(
    adapter: MujocoKinematics,
) -> None:
    motion = make_motion(adapter.info.joint_names)
    metrics = compute_contact_metrics_from_frames(
        motion,
        adapter.info,
        synthetic_frames(adapter),
        ContactEvaluationConfig.from_project_config(CONFIG, "feet_only"),
    )
    summary = metrics.summary

    np.testing.assert_allclose(metrics.left_foot_height_m, 0.0, atol=1e-15)
    np.testing.assert_allclose(metrics.right_foot_height_m, 0.0, atol=1e-15)
    assert np.isnan(metrics.left_foot_horizontal_speed_m_s[0])
    np.testing.assert_allclose(
        metrics.left_foot_horizontal_speed_m_s[1:],
        [0.0, 0.2, 0.2, 0.0],
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        metrics.left_contact_proxy_frame,
        [False, True, True, True, True],
    )
    np.testing.assert_array_equal(
        metrics.right_contact_proxy_frame,
        [False, True, True, True, True],
    )
    np.testing.assert_array_equal(
        metrics.left_foot_slide_frame,
        [False, False, True, True, False],
    )
    np.testing.assert_array_equal(
        metrics.right_foot_slide_frame,
        [False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        metrics.airborne_frame,
        [True, False, False, False, False],
    )
    assert summary.foot_slide_frame_count == 2
    assert summary.foot_slide_frame_rate == pytest.approx(0.4)
    assert summary.foot_slide_speed_p95_m_s == pytest.approx(0.2)
    assert summary.foot_slide_speed_max_m_s == pytest.approx(0.2)
    assert summary.foot_slide_event_count == 1
    assert summary.airborne_frame_rate == pytest.approx(0.2)
    assert summary.max_airborne_duration_s == pytest.approx(0.1)
    assert summary.left_contact_proxy_mismatch_frame_count == 2
    assert summary.right_contact_proxy_mismatch_frame_count == 3
    assert "CONTACT_PROXY_WITHOUT_SOURCE_LABELS" in metrics.warnings
    assert "MUJOCO_CONTACT_PROXY_MISMATCH_PRESENT" in metrics.warnings
    assert metrics.left_sole_points_world_m.flags.writeable is False
    assert metrics.left_foot_horizontal_speed_m_s.flags.writeable is False


def test_descriptive_policy_preserves_observations_without_calling_them_failures(
    adapter: MujocoKinematics,
) -> None:
    motion = make_motion(adapter.info.joint_names)
    frames = synthetic_frames(adapter)
    descriptive = compute_contact_metrics_from_frames(
        motion,
        adapter.info,
        frames,
        ContactEvaluationConfig.from_project_config(CONFIG, "descriptive_only"),
        motion_category="unknown",
    )

    assert descriptive.summary.undesired_ground_contact_frame_count == 1
    assert descriptive.summary.airborne_frame_count == 1
    assert descriptive.summary.undesired_ground_contact_is_failure is False
    assert descriptive.summary.airborne_is_failure is False
    assert descriptive.summary.motion_category == "unknown"


def test_rejects_missing_or_misaligned_fk_frames(adapter: MujocoKinematics) -> None:
    motion = make_motion(adapter.info.joint_names)
    frames = synthetic_frames(adapter)
    config = ContactEvaluationConfig.from_project_config(CONFIG, "feet_only")

    with pytest.raises(ValueError, match="Received 4 FK frames"):
        compute_contact_metrics_from_frames(
            motion, adapter.info, frames[:-1], config
        )
    with pytest.raises(ValueError, match="frame order mismatch"):
        compute_contact_metrics_from_frames(
            motion, adapter.info, frames[1:], config
        )


def test_real_sample_produces_finite_full_length_contact_metrics(
    adapter: MujocoKinematics,
) -> None:
    motion = load_motion_from_project_config(KNOWN_SAMPLE, CONFIG).require_record()
    bound = adapter.bind_motion(motion)
    metrics = compute_contact_metrics(
        bound,
        adapter,
        ContactEvaluationConfig.from_project_config(CONFIG, "feet_only"),
        motion_category="dance",
    )
    summary = metrics.summary

    assert metrics.left_foot_height_m.shape == (3944,)
    assert len(metrics.self_collision_pairs) == motion.num_frames
    assert np.isfinite(metrics.left_foot_height_m).all()
    assert np.isfinite(metrics.right_foot_height_m).all()
    assert summary.raw_contact_count > 0
    assert summary.self_collision_any_frame_count > 0
    assert summary.ground_contact_frame_count > 0
    assert summary.max_ground_penetration_m == pytest.approx(
        0.07070770008821849
    )
    assert summary.left_foot_height_min_m == pytest.approx(
        -0.07070770008821848
    )
    assert summary.motion_category == "dance"
    for rate in (
        summary.self_collision_any_frame_rate,
        summary.self_collision_nonhand_frame_rate,
        summary.ground_penetration_frame_rate,
        summary.undesired_ground_contact_frame_rate,
        summary.left_contact_proxy_rate,
        summary.right_contact_proxy_rate,
        summary.foot_slide_frame_rate,
        summary.airborne_frame_rate,
    ):
        assert 0.0 <= rate <= 1.0
