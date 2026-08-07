from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmr_quality.config import load_config
from gmr_quality.evaluation import QualityEvaluation, evaluate_motion_result
from gmr_quality.io import load_motion_from_project_config, validate_motion_payload
from gmr_quality.mujoco_kinematics import MujocoKinematics
from gmr_quality.reporting import (
    EVENT_COLUMNS,
    FRAME_METRIC_COLUMNS,
    write_evaluation_reports,
)
from gmr_quality.scoring import ScoringConfig, compute_quality_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "unitree_g1.yaml")
KNOWN_SAMPLE = Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "g1_dance1_subject2.pkl"
KNOWN_BVH = Path(
    "/root/gpufree-data/ubisoft-laforge-animation-dataset/lafan1/datasets/"
    "dance1_subject1.bvh"
)


@pytest.fixture(scope="module")
def adapter() -> MujocoKinematics:
    return MujocoKinematics.from_project_config(CONFIG)


@pytest.fixture(scope="module")
def real_evaluation(adapter: MujocoKinematics) -> QualityEvaluation:
    return evaluate_motion_result(
        load_motion_from_project_config(KNOWN_SAMPLE, CONFIG),
        adapter,
        CONFIG,
        motion_name="dance1_subject1_verified",
        motion_category="dance",
        contact_policy="feet_only",
        source_bvh=KNOWN_BVH,
    )


def controlled_load_result(adapter: MujocoKinematics):
    frames = 5
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    joint_pos = np.broadcast_to(
        (adapter.info.joint_limits.lower_rad + adapter.info.joint_limits.upper_rad)
        / 2.0,
        (frames, adapter.info.num_dofs),
    ).copy()
    joint_pos[2, 0] += 0.6
    return validate_motion_payload(
        {
            "fps": 10.0,
            "root_pos": np.broadcast_to(
                np.array([0.0, 0.0, 2.0]), (frames, 3)
            ).copy(),
            "root_rot": root_quat,
            "dof_pos": joint_pos,
            "dof_names": adapter.info.joint_names,
        },
        "controlled_score.pkl",
    )


def test_missing_fidelity_is_renormalized_instead_of_treated_as_full_score(
    adapter: MujocoKinematics,
) -> None:
    evaluation = evaluate_motion_result(
        controlled_load_result(adapter),
        adapter,
        CONFIG,
        motion_name="controlled_score",
        motion_category="synthetic",
        contact_policy="descriptive_only",
    )
    scores = evaluation.scores
    assert scores is not None

    expected = (
        0.30 * scores.smoothness_score
        + 0.25 * scores.joint_feasibility_score
        + 0.30 * scores.contact_score
    ) / 0.85
    assert scores.quality_score_v0 == pytest.approx(expected)
    assert np.isnan(scores.fidelity_score)
    assert scores.score_available_dimensions == (
        "smoothness",
        "joint_feasibility",
        "contact",
    )
    assert dict(scores.normalized_weights) == pytest.approx(
        {
            "smoothness": 0.30 / 0.85,
            "joint_feasibility": 0.25 / 0.85,
            "contact": 0.30 / 0.85,
        }
    )
    assert scores.score_is_calibrated is False
    assert scores.smooth_valid_frame.flags.writeable is False


def test_available_fidelity_uses_all_four_configured_weights(
    adapter: MujocoKinematics,
) -> None:
    base = evaluate_motion_result(
        controlled_load_result(adapter),
        adapter,
        CONFIG,
        motion_name="controlled_fidelity",
        motion_category="synthetic",
        contact_policy="descriptive_only",
    )
    fidelity = np.array([True, True, True, False, False])
    assert base.kinematic_metrics is not None
    assert base.contact_metrics is not None
    scores = compute_quality_scores(
        base.kinematic_metrics,
        base.contact_metrics,
        ScoringConfig.from_project_config(CONFIG),
        fidelity_valid_frame=fidelity,
    )

    assert scores.fidelity_score == pytest.approx(60.0)
    assert scores.score_available_dimensions == (
        "smoothness",
        "joint_feasibility",
        "contact",
        "fidelity",
    )
    expected = (
        0.30 * scores.smoothness_score
        + 0.25 * scores.joint_feasibility_score
        + 0.30 * scores.contact_score
        + 0.15 * scores.fidelity_score
    )
    assert scores.quality_score_v0 == pytest.approx(expected)
    with pytest.raises(ValueError, match="boolean shape"):
        compute_quality_scores(
            base.kinematic_metrics,
            base.contact_metrics,
            ScoringConfig.from_project_config(CONFIG),
            fidelity_valid_frame=np.ones(4, dtype=np.bool_),
        )


def test_events_are_entity_aware_inclusive_and_match_step5_counts(
    real_evaluation: QualityEvaluation,
) -> None:
    assert real_evaluation.contact_metrics is not None
    contact = real_evaluation.contact_metrics.summary
    counts = Counter(event.event_type for event in real_evaluation.events)

    assert counts["SELF_COLLISION_ANY"] == contact.self_collision_event_count
    assert (
        counts["SELF_COLLISION_NONHAND"]
        == contact.self_collision_nonhand_event_count
    )
    assert counts["GROUND_PENETRATION"] == contact.ground_penetration_event_count
    assert (
        counts["UNDESIRED_GROUND_CONTACT"]
        == contact.undesired_ground_contact_event_count
    )
    assert counts["FOOT_SLIDE"] == contact.foot_slide_event_count
    assert counts["AIRBORNE_PROXY"] > 0
    assert len(real_evaluation.events) == 1770

    for event in real_evaluation.events:
        assert event.start_frame <= event.end_frame
        assert event.start_time_s == pytest.approx(
            event.start_frame / real_evaluation.bound_motion.record.fps
        )
        assert event.end_time_s == pytest.approx(
            event.end_frame / real_evaluation.bound_motion.record.fps
        )
        assert event.duration_s == pytest.approx(
            event.frame_count / real_evaluation.bound_motion.record.fps
        )
    worst_self = max(
        event.max_severity
        for event in real_evaluation.events
        if event.event_type == "SELF_COLLISION_ANY"
    )
    assert worst_self == pytest.approx(contact.max_self_collision_penetration_m)


def test_real_decision_keeps_scores_separate_from_stable_hard_fail_reasons(
    real_evaluation: QualityEvaluation,
) -> None:
    scores = real_evaluation.scores
    decision = real_evaluation.decision
    assert scores is not None

    assert scores.smoothness_score == pytest.approx(81.05983772819472)
    assert scores.joint_feasibility_score == pytest.approx(26.74949290060852)
    assert scores.contact_score == pytest.approx(93.15415821501014)
    assert scores.quality_score_v0 == pytest.approx(69.35479059778069)
    contact = real_evaluation.contact_metrics
    assert contact is not None
    policy_invalid = (
        contact.self_collision_nonhand_frame
        | contact.ground_penetration_frame
        | contact.foot_slide_frame
        | contact.undesired_ground_contact_frame
    )
    assert scores.contact_score == pytest.approx(
        100.0 * np.mean(~policy_invalid)
    )
    assert decision.hard_fail is True
    codes = Counter(reason.code for reason in decision.failure_reasons)
    assert codes == {
        "JOINT_LIMIT_EXCEEDED": 1,
        "SEVERE_PENETRATION": 1,
        "SUSTAINED_SELF_COLLISION": 3,
        "UNDESIRED_GROUND_CONTACT": 7,
    }
    tokens = tuple(reason.token for reason in decision.failure_reasons)
    assert "JOINT_LIMIT_EXCEEDED:left_ankle_roll_joint@frame_2819" in tokens
    assert "SEVERE_PENETRATION:left_ankle_roll_link@frame_2879" in tokens
    warning_codes = {reason.code for reason in decision.warning_reasons}
    assert "FIDELITY_UNAVAILABLE" in warning_codes
    assert "SCORE_UNCALIBRATED" in warning_codes


def test_invalid_input_produces_hard_fail_summary_and_header_only_detail_files(
    adapter: MujocoKinematics,
    tmp_path: Path,
) -> None:
    payload = {
        "fps": 30.0,
        "root_pos": np.zeros((4, 3)),
        "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
        "dof_pos": np.zeros((4, 29)),
    }
    payload["dof_pos"][2, 3] = np.nan
    invalid = evaluate_motion_result(
        validate_motion_payload(payload, "invalid_step6.pkl"),
        adapter,
        CONFIG,
        motion_name="invalid_step6",
        motion_category="unknown",
        contact_policy="descriptive_only",
    )

    assert invalid.decision.hard_fail is True
    assert invalid.scores is None
    assert invalid.events == ()
    assert {reason.code for reason in invalid.decision.failure_reasons} == {"DATA_NAN"}
    paths = write_evaluation_reports(invalid, tmp_path / "invalid_report")
    summary = pd.read_csv(paths.quality_summary_csv)
    frames = pd.read_csv(paths.frame_metrics_csv)
    events = pd.read_csv(paths.quality_events_csv)

    assert len(summary) == 1
    assert bool(summary.loc[0, "hard_fail"]) is True
    assert bool(summary.loc[0, "data_valid"]) is False
    assert np.isnan(summary.loc[0, "quality_score_v0"])
    assert "DATA_NAN" in json.loads(summary.loc[0, "failure_reasons"])[0]
    assert list(frames.columns) == list(FRAME_METRIC_COLUMNS)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert frames.empty
    assert events.empty


def test_real_csv_reports_have_required_rows_schema_and_cross_file_alignment(
    real_evaluation: QualityEvaluation,
    tmp_path: Path,
) -> None:
    paths = write_evaluation_reports(real_evaluation, tmp_path / "real_report")
    summary = pd.read_csv(paths.quality_summary_csv)
    frames = pd.read_csv(paths.frame_metrics_csv)
    events = pd.read_csv(paths.quality_events_csv)

    required_summary = {
        "motion_name",
        "pkl_path",
        "robot_name",
        "joint_jump_rate",
        "joint_limit_violation_frame_rate",
        "self_collision_any_frame_rate",
        "ground_penetration_frame_rate",
        "foot_slide_frame_rate",
        "smoothness_score",
        "joint_feasibility_score",
        "contact_score",
        "fidelity_score",
        "quality_score_v0",
        "score_version",
        "score_available_dimensions",
        "score_is_calibrated",
        "hard_fail",
        "failure_reasons",
        "warning_reasons",
    }
    assert required_summary <= set(summary.columns)
    assert len(summary) == 1
    assert len(frames) == 3944
    assert len(events) == len(real_evaluation.events) == 1770
    assert list(frames.columns) == list(FRAME_METRIC_COLUMNS)
    assert list(events.columns) == list(EVENT_COLUMNS)
    np.testing.assert_allclose(
        frames["time_s"].to_numpy(),
        np.arange(3944) / 30.0,
        atol=1e-12,
    )
    assert summary.loc[0, "quality_score_v0"] == pytest.approx(
        real_evaluation.scores.quality_score_v0
    )
    assert np.isnan(summary.loc[0, "fidelity_score"])
    assert json.loads(summary.loc[0, "score_available_dimensions"]) == [
        "smoothness",
        "joint_feasibility",
        "contact",
    ]
    assert int(frames["joint_jump_flag"].sum()) == int(
        summary.loc[0, "joint_jump_count"]
    )
    slide_union = (
        frames["left_foot_slide_flag"].astype(bool)
        | frames["right_foot_slide_flag"].astype(bool)
    )
    assert int(slide_union.sum()) == int(summary.loc[0, "foot_slide_frame_count"])
    event_counts = events["event_type"].value_counts()
    assert event_counts["SELF_COLLISION_ANY"] == int(
        summary.loc[0, "self_collision_event_count"]
    )
    assert event_counts["GROUND_PENETRATION"] == int(
        summary.loc[0, "ground_penetration_event_count"]
    )
    assert np.allclose(
        events["duration_s"],
        (events["end_frame"] - events["start_frame"] + 1) / 30.0,
    )
