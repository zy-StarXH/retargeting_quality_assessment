from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluation import QualityEvaluation, evaluate_motion_result
from .io import load_motion_from_project_config, validate_motion_payload
from .manifest import sha256_file
from .mujoco_kinematics import MujocoKinematics
from .runner import EvaluationRunResult, InputSpecification, run_evaluations


@dataclass(frozen=True)
class PipelineValidationResult:
    output_dir: Path
    reports_dir: Path
    validation_json: Path
    evaluation_run: EvaluationRunResult
    checks: tuple[dict[str, Any], ...]
    ok: bool


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_root(destination: Path, source: Path) -> None:
    if _is_within(destination, source.parent):
        raise ValueError("Pipeline-validation output must be outside the source directory")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"Output directory must be empty: {destination}")
    else:
        destination.mkdir(parents=True)


def _base_payload(
    source_record,
    joint_names: tuple[str, ...],
    *,
    source_frame: int,
    frame_count: int,
) -> dict[str, Any]:
    if source_frame < 0 or source_frame >= source_record.num_frames:
        raise ValueError(
            f"source_frame {source_frame} is outside [0, {source_record.num_frames})"
        )
    if frame_count < 8:
        raise ValueError("frame_count must be at least 8 for controlled injections")
    return {
        "fps": source_record.fps,
        "root_pos": np.repeat(
            source_record.root_pos[source_frame][None, :], frame_count, axis=0
        ),
        "root_rot": np.repeat(
            source_record.root_quat_xyzw[source_frame][None, :],
            frame_count,
            axis=0,
        ),
        "dof_pos": np.repeat(
            source_record.joint_pos[source_frame][None, :], frame_count, axis=0
        ),
        "dof_names": joint_names,
    }


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
    }


def _write_pickle(payload: dict[str, Any], destination: Path) -> None:
    with destination.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _evaluate_payload(
    payload: dict[str, Any],
    source_name: str,
    config: dict[str, Any],
    kinematics: MujocoKinematics,
) -> QualityEvaluation:
    return evaluate_motion_result(
        validate_motion_payload(payload, source_name),
        kinematics,
        config,
        motion_name=source_name,
        motion_category="synthetic_validation",
        contact_policy="descriptive_only",
    )


def _choose_jump(
    pose: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    minimum_step_rad: float,
) -> tuple[int, float]:
    desired = minimum_step_rad + 0.1
    for index, value in enumerate(pose):
        if upper[index] - value > desired:
            return index, float(value + desired)
        if value - lower[index] > desired:
            return index, float(value - desired)
    raise ValueError("No joint has enough in-limit room for the jump injection")


def _choose_near_limit(
    pose: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[int, float]:
    joint_range = upper - lower
    normalized_margin = np.minimum(pose - lower, upper - pose) / joint_range
    candidates: list[tuple[float, int, float]] = []
    for index in np.flatnonzero(normalized_margin > 0.02):
        low_target = float(lower[index] + 0.01 * joint_range[index])
        high_target = float(upper[index] - 0.01 * joint_range[index])
        for target in (low_target, high_target):
            candidates.append((abs(target - pose[index]), int(index), target))
    if not candidates:
        raise ValueError("No baseline joint is outside the 2% near-limit band")
    _, index, target = min(candidates)
    return index, target


def _inject_ramped_peak(
    joint_position: np.ndarray,
    *,
    joint_index: int,
    peak_frame: int,
    peak_value: float,
    maximum_step_rad: float,
) -> int:
    baseline = float(joint_position[peak_frame, joint_index])
    delta = peak_value - baseline
    transition_frames = max(1, int(np.ceil(abs(delta) / maximum_step_rad)))
    available = min(peak_frame, joint_position.shape[0] - 1 - peak_frame)
    if transition_frames > available:
        raise ValueError(
            "Not enough frames to inject a near-limit peak below the jump threshold"
        )
    for offset in range(1, transition_frames + 1):
        fraction = offset / transition_frames
        joint_position[
            peak_frame - transition_frames + offset, joint_index
        ] = baseline + fraction * delta
        joint_position[peak_frame + offset, joint_index] = peak_value - fraction * delta
    return transition_frames


def _check(
    records: list[dict[str, Any]],
    name: str,
    condition: bool,
    *,
    actual: Any,
    expected: Any,
) -> None:
    records.append(
        {
            "name": name,
            "passed": bool(condition),
            "actual": actual,
            "expected": expected,
        }
    )


def _scenario_summary(evaluation: QualityEvaluation) -> dict[str, Any]:
    validation = evaluation.load_result.validation
    kinematic = evaluation.kinematic_metrics
    contact = evaluation.contact_metrics
    return {
        "data_valid": validation.data_valid,
        "failure_codes": list(validation.failure_codes),
        "hard_fail": evaluation.decision.hard_fail,
        "quality_score_v0": (
            evaluation.scores.quality_score_v0 if evaluation.scores else None
        ),
        "joint_jump_count": (
            kinematic.smoothness.summary.joint_jump_count if kinematic else None
        ),
        "near_limit_2pct_frame_count": (
            int(np.count_nonzero(kinematic.joint_feasibility.near_limit_2pct_frame))
            if kinematic
            else None
        ),
        "joint_limit_violation_frame_count": (
            kinematic.joint_feasibility.summary.joint_limit_violation_frame_count
            if kinematic
            else None
        ),
        "ground_penetration_frame_count": (
            contact.summary.ground_penetration_frame_count if contact else None
        ),
        "severe_ground_penetration_frame_count": (
            contact.summary.severe_ground_penetration_frame_count
            if contact
            else None
        ),
        "foot_slide_frame_count": (
            contact.summary.foot_slide_frame_count if contact else None
        ),
    }


def run_pipeline_validation(
    project_config: dict[str, Any],
    source_pkl: str | Path,
    output_dir: str | Path,
    *,
    source_frame: int = 0,
    frame_count: int = 12,
    injection_frame: int = 5,
    root_drop_m: float = 0.12,
    foot_slide_displacement_m: float = 0.02,
    plots_enabled: bool = True,
    execution_command: Sequence[str] = (),
) -> PipelineValidationResult:
    """Run controlled faults through the complete offline evaluation pipeline."""
    source = Path(source_pkl).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    _prepare_root(destination, source)
    if not 1 <= injection_frame < frame_count - 1:
        raise ValueError("injection_frame must have both a previous and next frame")
    if root_drop_m <= 0 or foot_slide_displacement_m <= 0:
        raise ValueError("Injection magnitudes must be positive")

    source_result = load_motion_from_project_config(source, project_config)
    source_record = source_result.require_record()
    kinematics = MujocoKinematics.from_project_config(project_config)
    base = _base_payload(
        source_record,
        kinematics.info.joint_names,
        source_frame=source_frame,
        frame_count=frame_count,
    )
    baseline_preview = _evaluate_payload(
        base, "baseline_preview", project_config, kinematics
    )
    if baseline_preview.contact_metrics is None:
        raise ValueError("Baseline preview did not produce contact metrics")
    foot_height = min(
        baseline_preview.contact_metrics.summary.left_foot_height_min_m,
        baseline_preview.contact_metrics.summary.right_foot_height_min_m,
    )

    joint_limits = kinematics.info.joint_limits
    jump_index, jump_value = _choose_jump(
        base["dof_pos"][injection_frame - 1],
        joint_limits.lower_rad,
        joint_limits.upper_rad,
        float(project_config["thresholds"]["joint_jump_step_rad"]),
    )
    near_index, near_value = _choose_near_limit(
        base["dof_pos"][injection_frame],
        joint_limits.lower_rad,
        joint_limits.upper_rad,
    )

    joint_jump = _copy_payload(base)
    joint_jump["dof_pos"][injection_frame, jump_index] = jump_value
    near_limit = _copy_payload(base)
    near_transition_frames = _inject_ramped_peak(
        near_limit["dof_pos"],
        joint_index=near_index,
        peak_frame=injection_frame,
        peak_value=near_value,
        maximum_step_rad=0.9
        * float(project_config["thresholds"]["joint_jump_step_rad"]),
    )
    root_down = _copy_payload(base)
    root_down["root_pos"][injection_frame, 2] -= root_drop_m
    contact_control = _copy_payload(base)
    contact_control["root_pos"][:, 2] -= foot_height
    foot_slide = _copy_payload(contact_control)
    foot_slide["root_pos"][injection_frame, 0] += foot_slide_displacement_m
    invalid_nan = _copy_payload(base)
    invalid_nan["dof_pos"][injection_frame, jump_index] = np.nan

    scenario_payloads = {
        "baseline": base,
        "joint_jump": joint_jump,
        "near_limit": near_limit,
        "root_down": root_down,
        "foot_contact_control": contact_control,
        "foot_slide": foot_slide,
        "invalid_nan": invalid_nan,
    }
    input_dir = destination / "injected_inputs"
    input_dir.mkdir()
    specifications: list[InputSpecification] = []
    for name, payload in scenario_payloads.items():
        path = input_dir / f"{name}.pkl"
        _write_pickle(payload, path)
        specifications.append(
            InputSpecification(
                path=path,
                motion_name=name,
                source_bvh=None,
                motion_category="synthetic_validation",
                contact_policy="descriptive_only",
                source_pair_status="unpaired",
                expected_pkl_sha256="",
                expected_source_bvh_sha256="",
                frame_alignment="",
                notes="Controlled step-8 pipeline validation scenario",
            )
        )
    report_dir = destination / "reports"
    evaluation_run = run_evaluations(
        project_config,
        tuple(specifications),
        report_dir,
        plots_enabled=plots_enabled,
        execution_command=execution_command,
        input_mode="pipeline_validation",
        protected_input_dirs=(input_dir,),
    )
    evaluations = {
        evaluation.metadata.motion_name: evaluation
        for evaluation in evaluation_run.evaluations
    }
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "all_scenarios_reported",
        set(evaluations) == set(scenario_payloads),
        actual=sorted(evaluations),
        expected=sorted(scenario_payloads),
    )
    baseline = evaluations["baseline"]
    baseline_kinematic = baseline.kinematic_metrics
    baseline_contact = baseline.contact_metrics
    assert baseline_kinematic is not None and baseline_contact is not None
    _check(
        checks,
        "baseline_clean",
        (
            baseline_kinematic.smoothness.summary.joint_jump_count == 0
            and baseline_kinematic.joint_feasibility.summary.joint_limit_violation_frame_count
            == 0
            and not np.any(
                baseline_kinematic.joint_feasibility.near_limit_2pct_frame
            )
            and baseline_contact.summary.ground_penetration_frame_count == 0
            and baseline_contact.summary.foot_slide_frame_count == 0
        ),
        actual=_scenario_summary(baseline),
        expected="zero injected-anomaly counts",
    )

    jump_evaluation = evaluations["joint_jump"]
    jump_metrics = jump_evaluation.kinematic_metrics
    assert jump_metrics is not None
    jump_frames = np.flatnonzero(jump_metrics.smoothness.joint_jump_frame).tolist()
    _check(
        checks,
        "joint_jump_detected_and_localized",
        jump_frames == [injection_frame, injection_frame + 1]
        and jump_metrics.smoothness.summary.worst_step_joint
        == kinematics.info.joint_names[jump_index],
        actual={
            "frames": jump_frames,
            "joint": jump_metrics.smoothness.summary.worst_step_joint,
        },
        expected={
            "frames": [injection_frame, injection_frame + 1],
            "joint": kinematics.info.joint_names[jump_index],
        },
    )

    near_evaluation = evaluations["near_limit"]
    near_metrics = near_evaluation.kinematic_metrics
    assert near_metrics is not None
    near_frames = np.flatnonzero(
        near_metrics.joint_feasibility.near_limit_2pct_frame
    ).tolist()
    _check(
        checks,
        "near_limit_detected_without_limit_violation",
        near_frames == [injection_frame]
        and near_metrics.joint_feasibility.summary.joint_limit_violation_frame_count
        == 0
        and near_metrics.smoothness.summary.joint_jump_count == 0
        and int(
            near_metrics.joint_feasibility.nearest_limit_joint_index[
                injection_frame
            ]
        )
        == near_index,
        actual={
            "frames": near_frames,
            "nearest_joint_at_injection": kinematics.info.joint_names[
                int(
                    near_metrics.joint_feasibility.nearest_limit_joint_index[
                        injection_frame
                    ]
                )
            ],
            "joint_jump_count": near_metrics.smoothness.summary.joint_jump_count,
        },
        expected={
            "frames": [injection_frame],
            "joint": kinematics.info.joint_names[near_index],
            "hard_limit_violations": 0,
            "joint_jump_count": 0,
        },
    )

    root_evaluation = evaluations["root_down"]
    root_contact = root_evaluation.contact_metrics
    assert root_contact is not None
    penetration_frames = np.flatnonzero(
        root_contact.ground_penetration_frame
    ).tolist()
    _check(
        checks,
        "root_down_penetration_detected_and_localized",
        injection_frame in penetration_frames
        and root_contact.summary.worst_penetration_frame == injection_frame,
        actual={
            "frames": penetration_frames,
            "worst_frame": root_contact.summary.worst_penetration_frame,
            "max_depth_m": root_contact.summary.max_ground_penetration_m,
        },
        expected={"contains_frame": injection_frame, "worst_frame": injection_frame},
    )

    control_evaluation = evaluations["foot_contact_control"]
    control_contact = control_evaluation.contact_metrics
    slide_evaluation = evaluations["foot_slide"]
    slide_contact = slide_evaluation.contact_metrics
    assert control_contact is not None and slide_contact is not None
    slide_frames = np.flatnonzero(slide_contact.foot_slide_frame).tolist()
    _check(
        checks,
        "foot_slide_detected_against_contact_control",
        control_contact.summary.foot_slide_frame_count == 0
        and slide_frames == [injection_frame, injection_frame + 1],
        actual={
            "control_count": control_contact.summary.foot_slide_frame_count,
            "injected_frames": slide_frames,
            "max_speed_m_s": slide_contact.summary.foot_slide_speed_max_m_s,
        },
        expected={
            "control_count": 0,
            "injected_frames": [injection_frame, injection_frame + 1],
        },
    )

    invalid_evaluation = evaluations["invalid_nan"]
    _check(
        checks,
        "nan_rejected_before_fk_and_scoring",
        not invalid_evaluation.data_valid
        and invalid_evaluation.bound_motion is None
        and invalid_evaluation.kinematic_metrics is None
        and invalid_evaluation.contact_metrics is None
        and invalid_evaluation.fidelity_metrics is None
        and invalid_evaluation.scores is None
        and invalid_evaluation.load_result.validation.failure_codes == ("DATA_NAN",),
        actual=_scenario_summary(invalid_evaluation),
        expected="DATA_NAN with no downstream metrics or score",
    )
    ok = all(record["passed"] for record in checks)
    validation_path = destination / "pipeline_validation.json"
    payload = {
        "schema_version": 1,
        "ok": ok,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "source_frame": source_frame,
        },
        "parameters": {
            "frame_count": frame_count,
            "injection_frame": injection_frame,
            "root_drop_m": root_drop_m,
            "foot_alignment_translation_m": -foot_height,
            "foot_slide_displacement_m": foot_slide_displacement_m,
            "joint_jump_joint": kinematics.info.joint_names[jump_index],
            "joint_jump_value_rad": jump_value,
            "near_limit_joint": kinematics.info.joint_names[near_index],
            "near_limit_value_rad": near_value,
            "near_limit_transition_frames": near_transition_frames,
        },
        "checks": checks,
        "scenarios": {
            name: _scenario_summary(evaluation)
            for name, evaluation in evaluations.items()
        },
        "reports": {
            "directory": str(report_dir),
            "run_manifest": str(evaluation_run.run_manifest_json),
            "runner_succeeded": evaluation_run.succeeded_count,
            "runner_failed": evaluation_run.failed_count,
            "expected_failed_validation": 1,
        },
        "injected_inputs": [
            {
                "path": str(specification.path),
                "sha256": sha256_file(specification.path),
            }
            for specification in specifications
        ],
    }
    with validation_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return PipelineValidationResult(
        output_dir=destination,
        reports_dir=report_dir,
        validation_json=validation_path,
        evaluation_run=evaluation_run,
        checks=tuple(checks),
        ok=ok,
    )
