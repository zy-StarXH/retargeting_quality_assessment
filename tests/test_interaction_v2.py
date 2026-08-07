from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from gmr_quality.interaction import (
    InteractionRecord,
    build_interaction_frame_rows,
    compute_interaction_frame_metrics,
    compute_interaction_metrics,
    extract_interaction_events,
    load_interaction_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "quality_assessment_v2.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _record(tmp_path: Path, *, frames: int = 14) -> InteractionRecord:
    fps = 50.0
    qpos = np.zeros((frames, 36), dtype=float)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    initial = np.array([0.24, 0.0, 0.90])
    target = initial + np.array([0.20, 0.0, 0.0])
    object_pos = np.repeat(initial[None, :], frames, axis=0)
    # Lift, translate, lower, and settle.
    object_pos[2:6, 2] += np.linspace(0.0, 0.14, 4)
    object_pos[6:9, 2] += 0.14
    object_pos[6:9, 0] += np.linspace(0.0, 0.20, 3)
    object_pos[9:12, 0] += 0.20
    object_pos[9:12, 2] += np.linspace(0.14, 0.0, 3)
    object_pos[12:] = target
    object_velocity = np.zeros_like(object_pos)
    object_velocity[1:] = np.diff(object_pos, axis=0) * fps
    object_velocity[-1] = 0.0

    expected = np.zeros((frames, 2), dtype=bool)
    expected[2:12] = True
    observed = expected.copy()
    hand_offset = 0.05 + 0.01 - 0.020  # 2 cm legal proxy overlap.
    left = object_pos + np.array([0.0, hand_offset, 0.0])
    right = object_pos + np.array([0.0, -hand_offset, 0.0])
    # Hands are open before approach and after release.
    left[~expected[:, 0]] = object_pos[~expected[:, 0]] + np.array([0.0, 0.12, 0.0])
    right[~expected[:, 1]] = object_pos[~expected[:, 1]] + np.array([0.0, -0.12, 0.0])
    identity = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1))
    phase_names = ("approach", "grasp", "carry", "release")
    phase_id = np.zeros(frames, dtype=np.int64)
    phase_id[2:5] = 1
    phase_id[5:12] = 2
    phase_id[12:] = 3
    return InteractionRecord(
        name="controlled_clean",
        task_label="bimanual_lift_translate_place",
        interaction_mode="bimanual_grasp",
        fps=fps,
        robot_qpos=qpos,
        left_hand_pos_m=left,
        right_hand_pos_m=right,
        left_hand_quat_xyzw=identity.copy(),
        right_hand_quat_xyzw=identity.copy(),
        object_pos_m=object_pos,
        object_quat_xyzw=identity.copy(),
        object_linvel_m_s=object_velocity,
        expected_contact=expected,
        observed_contact=observed,
        phase_id=phase_id,
        phase_names=phase_names,
        target_pos_m=target,
        object_half_size_m=np.array([0.05, 0.05, 0.05]),
        hand_proxy_radius_m=0.01,
        source_path=tmp_path / "controlled_clean.npz",
    )


def test_v2_clean_record_has_finite_metrics_and_no_events(tmp_path: Path) -> None:
    record = _record(tmp_path)
    config = _config()
    frame = compute_interaction_frame_metrics(record, config)
    summary = compute_interaction_metrics(record, config)
    rows = build_interaction_frame_rows(record, config, frame)
    events = extract_interaction_events(record, config, frame, summary)
    assert summary["score_version"] == "interaction_v2"
    assert summary["nominal_task_success"] is True
    assert summary["hard_fail"] is False
    assert summary["penetration_score"] == pytest.approx(100.0)
    assert summary["contact_timing_score"] == pytest.approx(100.0)
    assert all(
        not np.isinf(value)
        for value in summary.values()
        if isinstance(value, (int, float, np.number))
    )
    assert len(rows) == record.num_frames
    assert not events


def test_v2_missed_contact_reduces_score_and_creates_localized_event(tmp_path: Path) -> None:
    clean = _record(tmp_path)
    observed = clean.observed_contact.copy()
    observed[5:8, 0] = False
    faulty = replace(clean, name="missed_contact", observed_contact=observed)
    config = _config()
    clean_summary = compute_interaction_metrics(clean, config)
    faulty_summary = compute_interaction_metrics(faulty, config)
    events = extract_interaction_events(faulty, config, summary=faulty_summary)
    misses = [event for event in events if event.event_type == "EXPECTED_CONTACT_MISSED"]
    assert faulty_summary["contact_timing_score"] < clean_summary["contact_timing_score"]
    assert faulty_summary["grasp_stability_score"] < clean_summary["grasp_stability_score"]
    assert len(misses) == 1
    assert (misses[0].start_frame, misses[0].end_frame) == (5, 7)
    assert misses[0].entities == ("left_hand",)


def test_v2_relative_drift_and_illegal_penetration_are_detected(tmp_path: Path) -> None:
    clean = _record(tmp_path)
    left = clean.left_hand_pos_m.copy()
    left[7:10, 0] += 0.06
    left[10, 1] = clean.object_pos_m[10, 1]
    faulty = replace(clean, name="drift_and_penetration", left_hand_pos_m=left)
    config = _config()
    summary = compute_interaction_metrics(faulty, config)
    events = extract_interaction_events(faulty, config, summary=summary)
    event_types = {event.event_type for event in events}
    assert "RELATIVE_POSITION_DRIFT" in event_types
    assert "ILLEGAL_HAND_OBJECT_PENETRATION" in event_types
    assert summary["max_illegal_penetration_m"] > config["offline_metrics"]["severe_penetration_m"]
    assert summary["hard_fail"] is True
    assert "ILLEGAL_HAND_OBJECT_PENETRATION" in summary["failure_reason_codes"]


def test_v2_task_failure_is_a_hard_failure(tmp_path: Path) -> None:
    clean = _record(tmp_path)
    object_pos = clean.object_pos_m.copy()
    object_pos[12:, 0] -= 0.15
    faulty = replace(clean, name="task_failure", object_pos_m=object_pos)
    config = _config()
    summary = compute_interaction_metrics(faulty, config)
    events = extract_interaction_events(faulty, config, summary=summary)
    assert summary["nominal_task_success"] is False
    assert summary["task_effectiveness_score"] < 50.0
    assert summary["hard_fail"] is True
    assert "TASK_NOT_COMPLETED" in summary["failure_reason_codes"]
    assert any(event.event_type == "TASK_NOT_COMPLETED" for event in events)


def test_v2_loader_rejects_nonboolean_contact_and_nonfinite_static_fields(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    path = tmp_path / "bad_contact.npz"
    payload = {
        "schema_version": np.asarray("gmr_interaction_v2"),
        "name": np.asarray(record.name),
        "task_label": np.asarray(record.task_label),
        "interaction_mode": np.asarray(record.interaction_mode),
        "fps": np.asarray(record.fps),
        "robot_qpos": record.robot_qpos,
        "left_hand_pos_m": record.left_hand_pos_m,
        "right_hand_pos_m": record.right_hand_pos_m,
        "left_hand_quat_xyzw": record.left_hand_quat_xyzw,
        "right_hand_quat_xyzw": record.right_hand_quat_xyzw,
        "object_pos_m": record.object_pos_m,
        "object_quat_xyzw": record.object_quat_xyzw,
        "object_linvel_m_s": record.object_linvel_m_s,
        "expected_contact": record.expected_contact.astype(np.int64),
        "observed_contact": record.observed_contact,
        "phase_id": record.phase_id,
        "phase_names": np.asarray(record.phase_names),
        "target_pos_m": np.array([np.nan, 0.0, 0.0]),
        "object_half_size_m": record.object_half_size_m,
        "hand_proxy_radius_m": np.asarray(record.hand_proxy_radius_m),
    }
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="expected_contact must use a boolean dtype"):
        load_interaction_record(path)
