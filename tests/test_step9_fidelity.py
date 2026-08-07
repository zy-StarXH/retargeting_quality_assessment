from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from gmr_quality.config import load_config
from gmr_quality.fidelity import (
    FidelityBodyMapping,
    FidelityConfig,
    _aligned_source_frames,
    compute_lafan1_fidelity,
    transform_lafan1_frame_targets,
)
from gmr_quality.manifest import sha256_file
from gmr_quality.mujoco_kinematics import MujocoKinematics
from gmr_quality.runner import discover_inputs, run_evaluations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "unitree_g1.yaml"
CATALOG_PATH = PROJECT_ROOT / "data" / "motion_catalog.csv"
CONFIG = load_config(CONFIG_PATH)
PKL = Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "g1_dance1_subject2.pkl"
SOURCE = Path(
    "/root/gpufree-data/ubisoft-laforge-animation-dataset/lafan1/datasets/"
    "dance1_subject1.bvh"
)
PKL_SHA256 = "47186e8d2e605b0e64851f8d151c0460458618b1e5d9069d043809c958d72dcd"
SOURCE_SHA256 = "9a285b61bb0446aa4c919c62247ff895eb158bc693be3b7477d31ef070995d73"


@pytest.fixture(scope="module")
def verified_specifications():
    return discover_inputs(
        input_path=PKL,
        input_dir=None,
        pattern="*.pkl",
        catalog_path=CATALOG_PATH,
    )


@pytest.fixture(scope="module")
def verified_run(tmp_path_factory, verified_specifications):
    return run_evaluations(
        CONFIG,
        verified_specifications,
        tmp_path_factory.mktemp("step9") / "result",
        plots_enabled=True,
        execution_command=("gmr-quality", "evaluate", "step9-test"),
        input_mode="single",
    )


def test_catalog_locks_the_corrected_replay_verified_pair(
    verified_specifications,
) -> None:
    specification = verified_specifications[0]
    assert specification.motion_name == "dance1_subject1_verified"
    assert specification.path == PKL
    assert specification.source_bvh == SOURCE
    assert specification.source_pair_status == "verified_replay_exact"
    assert specification.expected_pkl_sha256 == PKL_SHA256
    assert specification.expected_source_bvh_sha256 == SOURCE_SHA256
    assert specification.frame_alignment == "source_prefix_drop_last"
    assert sha256_file(PKL) == PKL_SHA256
    assert sha256_file(SOURCE) == SOURCE_SHA256


def test_gmr_target_transform_scales_in_root_frame_and_composes_so3_offsets() -> None:
    quarter_turn_z = Rotation.from_euler("z", 90, degrees=True).as_quat(
        scalar_first=True
    )
    mappings = (
        FidelityBodyMapping(
            robot_body="robot_root",
            human_body="Hips",
            position_enabled=True,
            orientation_enabled=True,
            position_offset_local_m=np.array([1.0, 0.0, 0.0]),
            rotation_offset_wxyz=quarter_turn_z,
        ),
        FidelityBodyMapping(
            robot_body="robot_hand",
            human_body="Hand",
            position_enabled=True,
            orientation_enabled=True,
            position_offset_local_m=np.zeros(3),
            rotation_offset_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    frame = {
        "Hips": [np.array([2.0, 4.0, 6.0]), np.array([1.0, 0.0, 0.0, 0.0])],
        "Hand": [np.array([4.0, 4.0, 6.0]), np.array([1.0, 0.0, 0.0, 0.0])],
    }
    targets = transform_lafan1_frame_targets(
        frame,
        mappings,
        {"Hips": 0.5, "Hand": 0.25},
        "Hips",
    )

    np.testing.assert_allclose(targets["robot_root"][0], [1.0, 3.0, 3.0])
    np.testing.assert_allclose(targets["robot_hand"][0], [1.5, 2.0, 3.0])
    assert (
        Rotation.from_quat(
            targets["robot_root"][1], scalar_first=True
        ).inv()
        * Rotation.from_euler("z", 90, degrees=True)
    ).magnitude() == pytest.approx(0.0, abs=1e-12)
    assert targets["robot_root"][0].flags.writeable is False


def test_frame_alignment_is_explicit_and_never_silently_truncates() -> None:
    frames = [{"frame": index} for index in range(5)]
    assert _aligned_source_frames(frames, 5, "exact") == frames
    assert _aligned_source_frames(
        frames, 4, "source_prefix_drop_last"
    ) == frames[:4]
    with pytest.raises(ValueError, match="exactly one additional"):
        _aligned_source_frames(frames, 3, "source_prefix_drop_last")
    with pytest.raises(ValueError, match="Exact alignment"):
        _aligned_source_frames(frames, 4, "exact")


def test_real_verified_fidelity_has_expected_units_alignment_and_four_dimensional_score(
    verified_run,
) -> None:
    assert verified_run.failed_count == 0
    evaluation = verified_run.evaluations[0]
    fidelity = evaluation.fidelity_metrics
    scores = evaluation.scores
    assert fidelity is not None
    assert scores is not None
    summary = fidelity.summary

    assert evaluation.fidelity_status == "enabled_verified"
    assert summary.fidelity_source_frame_count == 3945
    assert summary.fidelity_robot_frame_count == 3944
    assert summary.fidelity_aligned_frame_count == 3944
    assert summary.fidelity_frame_alignment == "source_prefix_drop_last"
    assert summary.fidelity_source_fps == pytest.approx(30.000300003000028)
    assert summary.fidelity_robot_fps == pytest.approx(30.0)
    assert summary.fidelity_actual_human_height_m == pytest.approx(1.75)
    assert summary.fidelity_height_ratio == pytest.approx(1.75 / 1.8)
    assert summary.fidelity_position_body_count == 13
    assert summary.fidelity_orientation_body_count == 14
    assert summary.fidelity_end_effector_body_count == 4
    assert fidelity.keybody_position_error_m.shape == (3944, 13)
    assert fidelity.link_orientation_error_deg.shape == (3944, 14)
    assert fidelity.end_effector_position_error_m.shape == (3944, 4)

    assert summary.keybody_pos_error_mean_m == pytest.approx(
        0.09475304688235218
    )
    assert summary.keybody_pos_error_p95_m == pytest.approx(
        0.24159706880373422
    )
    assert summary.keybody_pos_error_max_m == pytest.approx(
        0.48294480548308505
    )
    assert summary.end_effector_pos_error_mean_m == pytest.approx(
        0.039388151021749786
    )
    assert summary.link_orientation_error_mean_deg == pytest.approx(
        5.361645653161623
    )
    assert summary.link_orientation_error_p95_deg == pytest.approx(
        16.542048973898055
    )
    assert summary.link_orientation_error_max_deg == pytest.approx(
        60.02819581698569
    )
    assert summary.fidelity_pass_frame_count == 3404
    assert summary.fidelity_pass_frame_rate == pytest.approx(
        0.8630831643002028
    )
    assert summary.worst_fidelity_position_body == "right_shoulder_yaw_link"
    assert summary.worst_fidelity_position_frame == 1853
    assert summary.worst_fidelity_orientation_body == "torso_link"
    assert summary.worst_fidelity_orientation_frame == 1854

    assert scores.fidelity_score == pytest.approx(86.30831643002028)
    assert scores.score_available_dimensions == (
        "smoothness",
        "joint_feasibility",
        "contact",
        "fidelity",
    )
    expected_total = (
        0.30 * scores.smoothness_score
        + 0.25 * scores.joint_feasibility_score
        + 0.30 * scores.contact_score
        + 0.15 * scores.fidelity_score
    )
    assert scores.quality_score_v0 == pytest.approx(expected_total)
    assert scores.quality_score_v0 == pytest.approx(71.89781947261663)
    warning_codes = {reason.code for reason in evaluation.decision.warning_reasons}
    assert "FIDELITY_UNAVAILABLE" not in warning_codes
    assert "SCORE_UNCALIBRATED" in warning_codes


def test_step9_reports_plot_and_manifest_preserve_fidelity_provenance(
    verified_run,
) -> None:
    root = verified_run.output_dir
    summary = pd.read_csv(root / "quality_summary.csv")
    frames = pd.read_csv(
        root / "frame_metrics" / "dance1_subject1_verified.csv"
    )
    manifest = json.loads((root / "run_manifest.json").read_text())
    fidelity_plot = root / "plots" / "dance1_subject1_verified_fidelity.png"

    assert len(summary) == 1
    assert len(frames) == 3944
    assert summary.loc[0, "fidelity_status"] == "enabled_verified"
    assert summary.loc[0, "source_pair_status"] == "verified_replay_exact"
    assert summary.loc[0, "fidelity_score"] == pytest.approx(86.30831643002028)
    assert int(frames["fidelity_valid_flag"].sum()) == 3404
    assert frames["keybody_pos_error_mean_m"].notna().all()
    assert frames["link_orientation_error_mean_deg"].notna().all()
    assert frames.loc[1853, "worst_fidelity_position_body"] == (
        "right_shoulder_yaw_link"
    )
    assert fidelity_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert manifest["execution"]["fidelity_requested"] is True
    assert manifest["execution"]["fidelity_enabled"] is True
    assert manifest["execution"]["fidelity_enabled_count"] == 1
    input_record = manifest["inputs"][0]
    assert input_record["input_sha256"] == PKL_SHA256
    assert input_record["source_bvh_sha256"] == SOURCE_SHA256
    assert input_record["fidelity_status"] == "enabled_verified"
    assert input_record["frame_alignment"] == "source_prefix_drop_last"
    output_paths = {record["path"] for record in manifest["outputs"]}
    assert "plots/dance1_subject1_verified_fidelity.png" in output_paths


def test_unverified_sources_stay_disabled_and_verified_hashes_are_mandatory(
    verified_run,
) -> None:
    explicit = discover_inputs(
        input_path=PKL,
        input_dir=None,
        pattern="*.pkl",
        catalog_path=CATALOG_PATH,
        source_bvh=SOURCE,
    )[0]
    assert explicit.source_pair_status == "provided_unverified"
    assert explicit.expected_pkl_sha256 == ""
    assert explicit.frame_alignment == ""

    unpaired = discover_inputs(
        input_path=Path(CONFIG["paths"]["gmr_root"])
        / "outputs"
        / "amass"
        / "g1_motion.pkl",
        input_dir=None,
        pattern="*.pkl",
        catalog_path=CATALOG_PATH,
    )[0]
    assert unpaired.source_pair_status == "unpaired"
    assert unpaired.source_bvh is None

    evaluation = verified_run.evaluations[0]
    assert evaluation.bound_motion is not None
    with pytest.raises(ValueError, match="PKL SHA-256 mismatch"):
        compute_lafan1_fidelity(
            evaluation.bound_motion,
            MujocoKinematics.from_project_config(CONFIG),
            SOURCE,
            FidelityConfig.from_project_config(CONFIG),
            pair_status="verified_replay_exact",
            expected_pkl_sha256="0" * 64,
            expected_source_bvh_sha256=SOURCE_SHA256,
            frame_alignment="source_prefix_drop_last",
        )
