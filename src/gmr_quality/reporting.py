from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contact_metrics import ContactMetricsSummary
from .evaluation import QualityEvaluation
from .fidelity import FidelityMetricsSummary
from .kinematic_metrics import JointFeasibilitySummary, SmoothnessSummary


FIDELITY_SUMMARY_COLUMNS = (
    "ik_error_stage2_mean",
    *(field.name for field in fields(FidelityMetricsSummary)),
)

FRAME_METRIC_COLUMNS = (
    "motion_name",
    "frame",
    "time_s",
    "max_joint_step_rad",
    "max_joint_step_joint",
    "max_joint_vel_rad_s",
    "max_joint_vel_joint",
    "max_joint_acc_rad_s2",
    "max_joint_jerk_rad_s3",
    "min_limit_margin_rad",
    "min_limit_margin_norm",
    "nearest_limit_joint",
    "joint_jump_flag",
    "position_limit_violation_flag",
    "near_limit_0p05rad_flag",
    "near_limit_2pct_flag",
    "velocity_violation_flag",
    "self_collision_any_flag",
    "self_collision_nonhand_flag",
    "self_collision_pairs",
    "ground_contact_flag",
    "penetration_flag",
    "severe_penetration_flag",
    "penetration_depth_m",
    "penetration_bodies",
    "undesired_ground_contact_flag",
    "undesired_ground_contact_bodies",
    "left_foot_height_m",
    "right_foot_height_m",
    "left_mujoco_ground_contact_flag",
    "right_mujoco_ground_contact_flag",
    "left_contact_proxy_flag",
    "right_contact_proxy_flag",
    "left_contact_proxy_mismatch_flag",
    "right_contact_proxy_mismatch_flag",
    "left_slide_speed_m_s",
    "right_slide_speed_m_s",
    "left_foot_slide_flag",
    "right_foot_slide_flag",
    "airborne_flag",
    "smooth_valid_flag",
    "joint_valid_flag",
    "contact_valid_flag",
    "fidelity_valid_flag",
    "ik_error_stage2",
    "keybody_pos_error_mean_m",
    "keybody_pos_error_max_m",
    "end_effector_pos_error_mean_m",
    "link_orientation_error_mean_deg",
    "link_orientation_error_max_deg",
    "worst_fidelity_position_body",
    "worst_fidelity_orientation_body",
)

EVENT_COLUMNS = (
    "motion_name",
    "event_type",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "entities",
    "max_severity",
    "details",
)


@dataclass(frozen=True)
class ReportPaths:
    output_dir: Path
    quality_summary_csv: Path
    frame_metrics_csv: Path
    quality_events_csv: Path


@dataclass(frozen=True)
class BatchReportPaths:
    output_dir: Path
    quality_summary_csv: Path
    frame_metrics_dir: Path
    quality_events_csv: Path
    frame_metrics_csvs: tuple[Path, ...]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list, dict)):
        return _json(value)
    return value


def _reason_tokens(reasons) -> str:
    return _json([reason.token for reason in reasons])


def _reason_details(reasons) -> str:
    return _json([reason.as_dict() for reason in reasons])


def build_summary_row(evaluation: QualityEvaluation) -> dict[str, Any]:
    validation = evaluation.load_result.validation
    metadata = evaluation.metadata
    fps = validation.fps
    num_frames = validation.num_frames
    duration_s = (
        num_frames / fps
        if fps is not None and num_frames is not None and fps > 0
        else float("nan")
    )
    row: dict[str, Any] = {
        "motion_name": metadata.motion_name,
        "source_bvh": metadata.source_bvh,
        "pkl_path": validation.source_path,
        "robot_name": metadata.robot_name,
        "motion_category": metadata.motion_category,
        "contact_policy": metadata.contact_policy,
        "source_pair_status": metadata.source_pair_status,
        "frame_alignment": metadata.frame_alignment,
        "fidelity_status": evaluation.fidelity_status,
        "fidelity_reason": evaluation.fidelity_reason,
        "notes": metadata.notes,
        "fps": fps,
        "num_frames": num_frames,
        "duration_s": duration_s,
        "nan_count": validation.nan_count,
        "inf_count": validation.inf_count,
        "quat_norm_max_error": validation.quat_norm_max_error,
        "data_valid": validation.data_valid,
        "model_xml_sha256": None,
        "joint_order_method": None,
        "source_model_provenance_verified": None,
    }
    if evaluation.bound_motion is not None:
        binding = evaluation.bound_motion.binding
        row.update(
            {
                "model_xml_sha256": binding.model_xml_sha256,
                "joint_order_method": binding.joint_order_method,
                "source_model_provenance_verified": (
                    binding.source_model_provenance_verified
                ),
            }
        )

    summary_types = (
        SmoothnessSummary,
        JointFeasibilitySummary,
        ContactMetricsSummary,
    )
    for summary_type in summary_types:
        for field in fields(summary_type):
            row.setdefault(field.name, float("nan"))
    if evaluation.kinematic_metrics is not None:
        row.update(asdict(evaluation.kinematic_metrics.smoothness.summary))
        row.update(asdict(evaluation.kinematic_metrics.joint_feasibility.summary))
    if evaluation.contact_metrics is not None:
        row.update(asdict(evaluation.contact_metrics.summary))

    for column in FIDELITY_SUMMARY_COLUMNS:
        row[column] = float("nan")
    if evaluation.fidelity_metrics is not None:
        row.update(asdict(evaluation.fidelity_metrics.summary))
    scores = evaluation.scores
    if scores is None:
        row.update(
            {
                "smoothness_score": float("nan"),
                "joint_feasibility_score": float("nan"),
                "contact_score": float("nan"),
                "fidelity_score": float("nan"),
                "quality_score": float("nan"),
                "quality_score_v0": float("nan"),
                "quality_score_v1": float("nan"),
                "score_version": evaluation.scoring_config.version,
                "score_available_dimensions": (),
                "score_is_calibrated": evaluation.scoring_config.calibrated,
                "normalized_score_weights": (),
            }
        )
    else:
        row.update(
            {
                "smoothness_score": scores.smoothness_score,
                "joint_feasibility_score": scores.joint_feasibility_score,
                "contact_score": scores.contact_score,
                "fidelity_score": scores.fidelity_score,
                "quality_score": scores.quality_score,
                "quality_score_v0": scores.quality_score_v0,
                "quality_score_v1": (
                    scores.quality_score
                    if scores.score_version == "v1"
                    else float("nan")
                ),
                "score_version": scores.score_version,
                "score_available_dimensions": scores.score_available_dimensions,
                "score_is_calibrated": scores.score_is_calibrated,
                "normalized_score_weights": scores.normalized_weights,
            }
        )
        if scores.fidelity_valid_frame is not None:
            row["fidelity_pass_frame_rate"] = float(
                np.mean(scores.fidelity_valid_frame)
            )

    decision = evaluation.decision
    row.update(
        {
            "hard_fail": decision.hard_fail,
            "failure_reason_codes": tuple(
                reason.code for reason in decision.failure_reasons
            ),
            "failure_reasons": _reason_tokens(decision.failure_reasons),
            "failure_reason_details": _reason_details(decision.failure_reasons),
            "warning_reason_codes": tuple(
                reason.code for reason in decision.warning_reasons
            ),
            "warning_reasons": _reason_tokens(decision.warning_reasons),
            "warning_reason_details": _reason_details(decision.warning_reasons),
        }
    )
    return {name: _csv_cell(value) for name, value in row.items()}


def _joint_name(names: tuple[str, ...], index: int) -> str | None:
    return names[index] if 0 <= index < len(names) else None


def build_frame_rows(evaluation: QualityEvaluation) -> list[dict[str, Any]]:
    if (
        evaluation.bound_motion is None
        or evaluation.kinematic_metrics is None
        or evaluation.contact_metrics is None
        or evaluation.scores is None
    ):
        return []
    motion = evaluation.bound_motion.record
    names = motion.joint_names or tuple(
        f"joint_{index}" for index in range(motion.num_dofs)
    )
    smooth = evaluation.kinematic_metrics.smoothness
    feasibility = evaluation.kinematic_metrics.joint_feasibility
    contact = evaluation.contact_metrics
    scores = evaluation.scores
    fidelity_flags = scores.fidelity_valid_frame
    fidelity = evaluation.fidelity_metrics
    rows: list[dict[str, Any]] = []
    for frame in range(motion.num_frames):
        pairs = tuple("|".join(pair) for pair in contact.self_collision_pairs[frame])
        row = {
            "motion_name": evaluation.metadata.motion_name,
            "frame": frame,
            "time_s": frame / motion.fps,
            "max_joint_step_rad": smooth.max_joint_step_rad[frame],
            "max_joint_step_joint": _joint_name(
                names, int(smooth.max_joint_step_joint_index[frame])
            ),
            "max_joint_vel_rad_s": smooth.max_joint_vel_rad_s[frame],
            "max_joint_vel_joint": _joint_name(
                names, int(smooth.max_joint_vel_joint_index[frame])
            ),
            "max_joint_acc_rad_s2": smooth.max_joint_acc_rad_s2[frame],
            "max_joint_jerk_rad_s3": smooth.max_joint_jerk_rad_s3[frame],
            "min_limit_margin_rad": feasibility.min_limit_margin_rad[frame],
            "min_limit_margin_norm": feasibility.min_limit_margin_norm[frame],
            "nearest_limit_joint": _joint_name(
                names, int(feasibility.nearest_limit_joint_index[frame])
            ),
            "joint_jump_flag": smooth.joint_jump_frame[frame],
            "position_limit_violation_flag": (
                feasibility.position_violation_frame[frame]
            ),
            "near_limit_0p05rad_flag": feasibility.near_limit_0p05rad_frame[frame],
            "near_limit_2pct_flag": feasibility.near_limit_2pct_frame[frame],
            "velocity_violation_flag": feasibility.velocity_violation_frame[frame],
            "self_collision_any_flag": contact.self_collision_any_frame[frame],
            "self_collision_nonhand_flag": contact.self_collision_nonhand_frame[frame],
            "self_collision_pairs": pairs,
            "ground_contact_flag": contact.ground_contact_frame[frame],
            "penetration_flag": contact.ground_penetration_frame[frame],
            "severe_penetration_flag": contact.severe_ground_penetration_frame[frame],
            "penetration_depth_m": contact.ground_penetration_depth_m[frame],
            "penetration_bodies": contact.ground_penetration_bodies[frame],
            "undesired_ground_contact_flag": (
                contact.undesired_ground_contact_frame[frame]
            ),
            "undesired_ground_contact_bodies": (
                contact.undesired_ground_contact_bodies[frame]
            ),
            "left_foot_height_m": contact.left_foot_height_m[frame],
            "right_foot_height_m": contact.right_foot_height_m[frame],
            "left_mujoco_ground_contact_flag": (
                contact.left_mujoco_ground_contact_frame[frame]
            ),
            "right_mujoco_ground_contact_flag": (
                contact.right_mujoco_ground_contact_frame[frame]
            ),
            "left_contact_proxy_flag": contact.left_contact_proxy_frame[frame],
            "right_contact_proxy_flag": contact.right_contact_proxy_frame[frame],
            "left_contact_proxy_mismatch_flag": (
                contact.left_contact_proxy_mismatch_frame[frame]
            ),
            "right_contact_proxy_mismatch_flag": (
                contact.right_contact_proxy_mismatch_frame[frame]
            ),
            "left_slide_speed_m_s": contact.left_foot_horizontal_speed_m_s[frame],
            "right_slide_speed_m_s": contact.right_foot_horizontal_speed_m_s[frame],
            "left_foot_slide_flag": contact.left_foot_slide_frame[frame],
            "right_foot_slide_flag": contact.right_foot_slide_frame[frame],
            "airborne_flag": contact.airborne_frame[frame],
            "smooth_valid_flag": scores.smooth_valid_frame[frame],
            "joint_valid_flag": scores.joint_valid_frame[frame],
            "contact_valid_flag": scores.contact_valid_frame[frame],
            "fidelity_valid_flag": (
                fidelity_flags[frame] if fidelity_flags is not None else None
            ),
            "ik_error_stage2": float("nan"),
            "keybody_pos_error_mean_m": (
                fidelity.keybody_pos_error_frame_mean_m[frame]
                if fidelity is not None
                else float("nan")
            ),
            "keybody_pos_error_max_m": (
                fidelity.keybody_pos_error_frame_max_m[frame]
                if fidelity is not None
                else float("nan")
            ),
            "end_effector_pos_error_mean_m": (
                fidelity.end_effector_pos_error_frame_mean_m[frame]
                if fidelity is not None
                else float("nan")
            ),
            "link_orientation_error_mean_deg": (
                fidelity.link_orientation_error_frame_mean_deg[frame]
                if fidelity is not None
                else float("nan")
            ),
            "link_orientation_error_max_deg": (
                fidelity.link_orientation_error_frame_max_deg[frame]
                if fidelity is not None
                else float("nan")
            ),
            "worst_fidelity_position_body": (
                fidelity.worst_position_body_by_frame[frame]
                if fidelity is not None
                else None
            ),
            "worst_fidelity_orientation_body": (
                fidelity.worst_orientation_body_by_frame[frame]
                if fidelity is not None
                else None
            ),
        }
        rows.append({name: _csv_cell(row[name]) for name in FRAME_METRIC_COLUMNS})
    return rows


def build_event_rows(evaluation: QualityEvaluation) -> list[dict[str, Any]]:
    return [
        {
            "motion_name": event.motion_name,
            "event_type": event.event_type,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_time_s": event.start_time_s,
            "end_time_s": event.end_time_s,
            "duration_s": event.duration_s,
            "entities": _json(event.entities),
            "max_severity": event.max_severity,
            "details": _json(event.details_dict()),
        }
        for event in evaluation.events
    ]


def _safe_motion_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe:
        raise ValueError("motion_name cannot produce an empty report filename")
    return safe


def write_evaluation_reports(
    evaluation: QualityEvaluation,
    output_dir: str | Path,
) -> ReportPaths:
    """Write the three step-6 CSV products for one evaluation."""
    destination = Path(output_dir).expanduser().resolve()
    frame_dir = destination / "frame_metrics"
    frame_dir.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "quality_summary.csv"
    events_path = destination / "quality_events.csv"
    frame_path = frame_dir / f"{_safe_motion_filename(evaluation.metadata.motion_name)}.csv"

    pd.DataFrame([build_summary_row(evaluation)]).to_csv(
        summary_path, index=False
    )
    pd.DataFrame(
        build_frame_rows(evaluation), columns=FRAME_METRIC_COLUMNS
    ).to_csv(frame_path, index=False)
    pd.DataFrame(build_event_rows(evaluation), columns=EVENT_COLUMNS).to_csv(
        events_path, index=False
    )
    return ReportPaths(
        output_dir=destination,
        quality_summary_csv=summary_path,
        frame_metrics_csv=frame_path,
        quality_events_csv=events_path,
    )


def write_batch_reports(
    evaluations: tuple[QualityEvaluation, ...],
    output_dir: str | Path,
) -> BatchReportPaths:
    """Write deterministic aggregate reports while keeping one frame CSV per motion."""
    if not evaluations:
        raise ValueError("At least one evaluation is required for batch reports")
    names = tuple(evaluation.metadata.motion_name for evaluation in evaluations)
    if len(set(names)) != len(names):
        raise ValueError("Batch motion names must be unique")

    destination = Path(output_dir).expanduser().resolve()
    frame_dir = destination / "frame_metrics"
    frame_dir.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "quality_summary.csv"
    events_path = destination / "quality_events.csv"
    summary_rows = [build_summary_row(evaluation) for evaluation in evaluations]
    event_rows = [
        row
        for evaluation in evaluations
        for row in build_event_rows(evaluation)
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(event_rows, columns=EVENT_COLUMNS).to_csv(events_path, index=False)

    frame_paths: list[Path] = []
    for evaluation in evaluations:
        frame_path = (
            frame_dir / f"{_safe_motion_filename(evaluation.metadata.motion_name)}.csv"
        )
        pd.DataFrame(
            build_frame_rows(evaluation), columns=FRAME_METRIC_COLUMNS
        ).to_csv(frame_path, index=False)
        frame_paths.append(frame_path)
    return BatchReportPaths(
        output_dir=destination,
        quality_summary_csv=summary_path,
        frame_metrics_dir=frame_dir,
        quality_events_csv=events_path,
        frame_metrics_csvs=tuple(frame_paths),
    )
