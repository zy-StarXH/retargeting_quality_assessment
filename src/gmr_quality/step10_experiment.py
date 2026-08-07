from __future__ import annotations

import csv
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gmr-quality-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .evaluation import QualityEvaluation
from .manifest import runtime_info, sha256_file
from .mujoco_kinematics import MujocoKinematics
from .plotting import _save_figure
from .runner import EvaluationRunResult, discover_inputs, run_evaluations
from .step10_dataset import STEP10_STRATA, load_step10_selection


SCORE_COLUMNS = (
    "smoothness_score",
    "joint_feasibility_score",
    "contact_score",
    "fidelity_score",
    "quality_score",
)
SENSITIVITY_GRID = {
    "joint_jump_step_rad": (0.4, 0.5, 0.6),
    "velocity_limit_rad_s": (2.5 * np.pi, 3.0 * np.pi, 3.5 * np.pi),
    "joint_limit_tolerance_rad": (0.0, 1e-6, 1e-5),
    "near_limit_fraction": (0.01, 0.02, 0.03),
    "penetration_m": (0.005, 0.01, 0.02),
    "severe_penetration_m": (0.04, 0.05, 0.06),
    "foot_slide_speed_m_s": (0.05, 0.10, 0.15),
    "fidelity_position_m": (0.08, 0.10, 0.12),
    "fidelity_orientation_deg": (8.0, 10.0, 12.0),
    "sustained_collision_frames": (3.0, 5.0, 10.0),
}


@dataclass(frozen=True)
class Step10ExperimentResult:
    output_dir: Path
    evaluation_run: EvaluationRunResult
    automated_analysis_json: Path
    review_candidates_csv: Path
    ok: bool


@dataclass(frozen=True)
class Step10FinalizationResult:
    output_dir: Path
    review_csv: Path
    review_summary_json: Path
    manifest_json: Path
    ok: bool


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Experiment output is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"Experiment output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _max_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=np.bool_):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _has_sustained_nonhand_collision(
    evaluation: QualityEvaluation,
    minimum_frames: int,
) -> bool:
    assert evaluation.contact_metrics is not None
    frames = evaluation.contact_metrics.self_collision_nonhand_pairs
    pairs = {pair for frame in frames for pair in frame}
    return any(
        _max_run(np.asarray([pair in frame for frame in frames])) >= minimum_frames
        for pair in pairs
    )


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    raw = config["thresholds"]
    return {
        "joint_jump_step_rad": float(raw["joint_jump_step_rad"]),
        "velocity_limit_rad_s": float(raw["velocity_limit_rad_s"]),
        "joint_limit_tolerance_rad": float(
            raw.get("joint_limit_tolerance_rad", 0.0)
        ),
        "near_limit_fraction": float(raw["near_limit_fraction"]),
        "penetration_m": float(raw["penetration_m"]),
        "severe_penetration_m": float(raw["severe_penetration_m"]),
        "foot_slide_speed_m_s": float(raw["foot_slide_speed_m_s"]),
        "fidelity_position_m": float(raw["fidelity_position_m"]),
        "fidelity_orientation_deg": float(raw["fidelity_orientation_deg"]),
        "sustained_collision_frames": float(raw["sustained_collision_frames"]),
    }


def _score_under_thresholds(
    evaluation: QualityEvaluation,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if (
        evaluation.bound_motion is None
        or evaluation.kinematic_metrics is None
        or evaluation.contact_metrics is None
        or evaluation.fidelity_metrics is None
        or evaluation.scores is None
    ):
        raise ValueError("Step-10 sensitivity requires a valid verified-pair evaluation")
    motion = evaluation.bound_motion.record
    kinematic = evaluation.kinematic_metrics
    contact = evaluation.contact_metrics
    fidelity = evaluation.fidelity_metrics
    derivatives = kinematic.smoothness.derivatives
    step = np.asarray(derivatives.joint_step_abs_rad)
    velocity = np.asarray(derivatives.joint_vel_rad_s)
    joint_jump = np.any(
        np.isfinite(step) & (step > thresholds["joint_jump_step_rad"]), axis=1
    )
    velocity_violation = np.any(
        np.isfinite(velocity)
        & (np.abs(velocity) > thresholds["velocity_limit_rad_s"]),
        axis=1,
    )
    smooth_valid = ~(joint_jump | velocity_violation)

    q = np.asarray(motion.joint_pos)
    limits = evaluation.bound_motion.joint_limits
    tolerance = thresholds["joint_limit_tolerance_rad"]
    position_violation = np.any(
        (q < limits.lower_rad - tolerance) | (q > limits.upper_rad + tolerance),
        axis=1,
    )
    near_limit = np.asarray(
        kinematic.joint_feasibility.min_limit_margin_norm
    ) < thresholds["near_limit_fraction"]
    joint_valid = ~(position_violation | near_limit)

    penetration = (
        np.asarray(contact.ground_penetration_depth_m)
        > thresholds["penetration_m"]
    )
    left_slide = (
        contact.left_contact_proxy_frame
        & np.isfinite(contact.left_foot_horizontal_speed_m_s)
        & (
            contact.left_foot_horizontal_speed_m_s
            > thresholds["foot_slide_speed_m_s"]
        )
    )
    right_slide = (
        contact.right_contact_proxy_frame
        & np.isfinite(contact.right_foot_horizontal_speed_m_s)
        & (
            contact.right_foot_horizontal_speed_m_s
            > thresholds["foot_slide_speed_m_s"]
        )
    )
    contact_invalid = contact.self_collision_nonhand_frame | penetration | left_slide | right_slide
    if contact.summary.undesired_ground_contact_is_failure:
        contact_invalid = contact_invalid | contact.undesired_ground_contact_frame
    if contact.summary.airborne_is_failure:
        contact_invalid = contact_invalid | contact.airborne_frame
    contact_valid = ~contact_invalid

    fidelity_valid = (
        fidelity.keybody_pos_error_frame_mean_m
        < thresholds["fidelity_position_m"]
    ) & (
        fidelity.link_orientation_error_frame_mean_deg
        < thresholds["fidelity_orientation_deg"]
    )
    dimension_scores = {
        "smoothness_score": 100.0 * float(np.mean(smooth_valid)),
        "joint_feasibility_score": 100.0 * float(np.mean(joint_valid)),
        "contact_score": 100.0 * float(np.mean(contact_valid)),
        "fidelity_score": 100.0 * float(np.mean(fidelity_valid)),
    }
    weights = config["scoring"]["weights"]
    quality_score = sum(
        dimension_scores[f"{dimension}_score"] * float(weight)
        for dimension, weight in weights.items()
    ) / sum(float(value) for value in weights.values())

    joint_hard = bool(np.any(position_violation))
    severe_hard = bool(
        np.max(contact.ground_penetration_depth_m)
        > thresholds["severe_penetration_m"]
    )
    collision_hard = _has_sustained_nonhand_collision(
        evaluation, int(thresholds["sustained_collision_frames"])
    )
    policy_hard = bool(
        (
            contact.summary.undesired_ground_contact_is_failure
            and np.any(contact.undesired_ground_contact_frame)
        )
        or (
            contact.summary.airborne_is_failure
            and np.any(contact.airborne_frame)
        )
    )
    return {
        **dimension_scores,
        "quality_score": float(quality_score),
        "hard_fail": joint_hard or severe_hard or collision_hard or policy_hard,
        "joint_limit_hard_fail": joint_hard,
        "severe_penetration_hard_fail": severe_hard,
        "sustained_collision_hard_fail": collision_hard,
        "contact_policy_hard_fail": policy_hard,
    }


def _sensitivity_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = _thresholds(config)
    scenarios: list[dict[str, Any]] = [
        {
            "scenario_id": "baseline",
            "changed_parameter": "none",
            "changed_value": float("nan"),
            "thresholds": baseline,
        }
    ]
    for parameter, values in SENSITIVITY_GRID.items():
        base_value = baseline[parameter]
        for value in values:
            if np.isclose(value, base_value, rtol=0.0, atol=1e-15):
                continue
            modified = dict(baseline)
            modified[parameter] = float(value)
            scenarios.append(
                {
                    "scenario_id": f"{parameter}={value:.12g}",
                    "changed_parameter": parameter,
                    "changed_value": float(value),
                    "thresholds": modified,
                }
            )
    return scenarios


def build_sensitivity_tables(
    evaluations: Sequence[QualityEvaluation],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    scenarios = _sensitivity_scenarios(config)
    for scenario in scenarios:
        for evaluation in evaluations:
            values = _score_under_thresholds(
                evaluation, config, scenario["thresholds"]
            )
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "changed_parameter": scenario["changed_parameter"],
                    "changed_value": scenario["changed_value"],
                    "motion_name": evaluation.metadata.motion_name,
                    "motion_category": evaluation.metadata.motion_category,
                    **values,
                }
            )
    runs = pd.DataFrame(rows)
    baseline = runs.loc[runs["scenario_id"] == "baseline"].set_index(
        "motion_name"
    )
    runs["baseline_quality_score"] = runs["motion_name"].map(
        baseline["quality_score"]
    )
    runs["quality_score_delta"] = (
        runs["quality_score"] - runs["baseline_quality_score"]
    )
    runs["baseline_hard_fail"] = runs["motion_name"].map(baseline["hard_fail"])
    runs["hard_fail_changed"] = runs["hard_fail"] != runs["baseline_hard_fail"]
    runs["global_rank"] = runs.groupby("scenario_id")["quality_score"].rank(
        ascending=False, method="min"
    )
    runs["stratum_rank"] = runs.groupby(
        ["scenario_id", "motion_category"]
    )["quality_score"].rank(ascending=False, method="min")

    summary_rows: list[dict[str, Any]] = []
    for scenario_id, scenario_rows in runs.groupby("scenario_id", sort=False):
        for scope, scoped in [("all", scenario_rows), *(
            (stratum, scenario_rows.loc[scenario_rows["motion_category"] == stratum])
            for stratum in STEP10_STRATA
        )]:
            baseline_scores = scoped["motion_name"].map(baseline["quality_score"])
            if len(scoped) >= 2 and scoped["quality_score"].nunique() > 1 and baseline_scores.nunique() > 1:
                rho = float(spearmanr(baseline_scores, scoped["quality_score"]).statistic)
            else:
                rho = float("nan")
            first = scoped.iloc[0]
            summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "changed_parameter": first["changed_parameter"],
                    "changed_value": first["changed_value"],
                    "scope": scope,
                    "motion_count": len(scoped),
                    "mean_quality_score": float(scoped["quality_score"].mean()),
                    "mean_quality_score_delta": float(scoped["quality_score_delta"].mean()),
                    "max_abs_quality_score_delta": float(scoped["quality_score_delta"].abs().max()),
                    "hard_fail_count": int(scoped["hard_fail"].sum()),
                    "hard_fail_flip_count": int(scoped["hard_fail_changed"].sum()),
                    "baseline_rank_spearman_rho": rho,
                }
            )
    return runs, pd.DataFrame(summary_rows)


def build_dimension_correlations(motions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", motions)] + [
        (stratum, motions.loc[motions["motion_category"] == stratum])
        for stratum in STEP10_STRATA
    ]
    for scope, data in scopes:
        for left, right in itertools.combinations(SCORE_COLUMNS, 2):
            constant = data[left].nunique() < 2 or data[right].nunique() < 2
            if constant or len(data) < 3:
                rho = p_value = float("nan")
            else:
                result = spearmanr(data[left], data[right])
                rho = float(result.statistic)
                p_value = float(result.pvalue)
            rows.append(
                {
                    "scope": scope,
                    "n": len(data),
                    "left_dimension": left,
                    "right_dimension": right,
                    "spearman_rho": rho,
                    "p_value_two_sided": p_value,
                    "constant_input": constant,
                    "inferential_use_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def _motion_table(evaluations: Sequence[QualityEvaluation]) -> pd.DataFrame:
    rows = []
    for evaluation in evaluations:
        scores = evaluation.scores
        fidelity = evaluation.fidelity_metrics
        contact = evaluation.contact_metrics
        kinematic = evaluation.kinematic_metrics
        assert scores and fidelity and contact and kinematic and evaluation.bound_motion
        rows.append(
            {
                "motion_name": evaluation.metadata.motion_name,
                "motion_category": evaluation.metadata.motion_category,
                "contact_policy": evaluation.metadata.contact_policy,
                "source_bvh": str(evaluation.metadata.source_bvh),
                "pkl_path": str(evaluation.bound_motion.record.source_path),
                "num_frames": evaluation.bound_motion.record.num_frames,
                "duration_s": evaluation.bound_motion.record.duration_s,
                "smoothness_score": scores.smoothness_score,
                "joint_feasibility_score": scores.joint_feasibility_score,
                "contact_score": scores.contact_score,
                "fidelity_score": scores.fidelity_score,
                "quality_score": scores.quality_score,
                "score_version": scores.score_version,
                "score_is_calibrated": scores.score_is_calibrated,
                "hard_fail": evaluation.decision.hard_fail,
                "failure_reason_codes": json.dumps(
                    list(
                        dict.fromkeys(
                            reason.code
                            for reason in evaluation.decision.failure_reasons
                        )
                    ),
                    separators=(",", ":"),
                ),
                "joint_jump_rate": kinematic.smoothness.summary.joint_jump_rate,
                "joint_limit_violation_frame_rate": kinematic.joint_feasibility.summary.joint_limit_violation_frame_rate,
                "near_limit_2pct_frame_rate": kinematic.joint_feasibility.summary.near_limit_2pct_frame_rate,
                "self_collision_nonhand_frame_rate": contact.summary.self_collision_nonhand_frame_rate,
                "ground_penetration_frame_rate": contact.summary.ground_penetration_frame_rate,
                "foot_slide_frame_rate": contact.summary.foot_slide_frame_rate,
                "fidelity_position_mean_m": fidelity.summary.keybody_pos_error_mean_m,
                "fidelity_orientation_mean_deg": fidelity.summary.link_orientation_error_mean_deg,
            }
        )
    result = pd.DataFrame(rows)
    result["global_rank"] = result["quality_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["stratum_rank"] = result.groupby("motion_category")[
        "quality_score"
    ].rank(ascending=False, method="min").astype(int)
    return result.sort_values(
        ["motion_category", "stratum_rank", "motion_name"]
    ).reset_index(drop=True)


def _stratum_summary(motions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in STEP10_STRATA:
        group = motions.loc[motions["motion_category"] == stratum]
        row: dict[str, Any] = {
            "motion_category": stratum,
            "motion_count": len(group),
            "total_frames": int(group["num_frames"].sum()),
            "total_duration_s": float(group["duration_s"].sum()),
            "hard_fail_count": int(group["hard_fail"].sum()),
            "hard_fail_rate": float(group["hard_fail"].mean()),
        }
        for column in SCORE_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _review_frames(evaluation: QualityEvaluation) -> tuple[int, ...]:
    assert evaluation.bound_motion and evaluation.kinematic_metrics
    assert evaluation.contact_metrics and evaluation.fidelity_metrics
    count = evaluation.bound_motion.record.num_frames
    candidates = {
        0,
        count - 1,
        *np.linspace(0, count - 1, 6, dtype=int).tolist(),
        evaluation.kinematic_metrics.smoothness.summary.worst_step_frame,
        evaluation.contact_metrics.summary.worst_self_collision_frame,
        evaluation.contact_metrics.summary.worst_penetration_frame,
        evaluation.fidelity_metrics.summary.worst_fidelity_position_frame,
    }
    valid = sorted(int(value) for value in candidates if value is not None and 0 <= int(value) < count)
    if len(valid) <= 6:
        return tuple(valid)
    anchor = {
        0,
        count - 1,
        evaluation.kinematic_metrics.smoothness.summary.worst_step_frame,
        evaluation.contact_metrics.summary.worst_self_collision_frame,
        evaluation.contact_metrics.summary.worst_penetration_frame,
        evaluation.fidelity_metrics.summary.worst_fidelity_position_frame,
    }
    result = sorted(int(value) for value in anchor if value is not None and 0 <= int(value) < count)
    for value in np.linspace(0, count - 1, 6, dtype=int):
        if int(value) not in result:
            result.append(int(value))
        if len(result) == 6:
            break
    return tuple(sorted(result[:6]))


def _plot_anomaly_timeline(frame_metrics: pd.DataFrame, path: Path) -> None:
    required = {
        "time_s",
        "joint_jump_flag",
        "near_limit_2pct_flag",
        "self_collision_nonhand_flag",
        "penetration_flag",
        "undesired_ground_contact_flag",
        "left_foot_slide_flag",
        "right_foot_slide_flag",
        "fidelity_valid_flag",
    }
    missing = sorted(required - set(frame_metrics.columns))
    if missing:
        raise ValueError(
            "Step-10 frame metrics are missing anomaly columns: "
            + ", ".join(missing)
        )
    time_s = frame_metrics["time_s"].to_numpy(dtype=np.float64)
    flags = (
        ("joint jump", frame_metrics["joint_jump_flag"].to_numpy(dtype=bool)),
        ("near joint limit", frame_metrics["near_limit_2pct_flag"].to_numpy(dtype=bool)),
        (
            "non-hand self collision",
            frame_metrics["self_collision_nonhand_flag"].to_numpy(dtype=bool),
        ),
        ("ground penetration", frame_metrics["penetration_flag"].to_numpy(dtype=bool)),
        (
            "undesired ground contact",
            frame_metrics["undesired_ground_contact_flag"].to_numpy(dtype=bool),
        ),
        (
            "foot slide",
            frame_metrics["left_foot_slide_flag"].to_numpy(dtype=bool)
            | frame_metrics["right_foot_slide_flag"].to_numpy(dtype=bool),
        ),
        (
            "fidelity threshold failure",
            ~frame_metrics["fidelity_valid_flag"].to_numpy(dtype=bool),
        ),
    )
    figure, axis = plt.subplots(figsize=(13, 4.8))
    for row_index, (label, mask) in enumerate(flags):
        axis.fill_between(
            time_s,
            row_index - 0.35,
            row_index + 0.35,
            where=mask,
            step="mid",
            color="tab:red",
            linewidth=0.0,
        )
    axis.set_yticks(range(len(flags)), [label for label, _ in flags])
    axis.set_ylim(len(flags) - 0.45, -0.55)
    axis.set_xlim(float(time_s[0]), float(time_s[-1]))
    axis.set_xlabel("time [s]")
    axis.set_title("Full-sequence anomaly timeline (red = flagged frame)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_pose_montage(
    evaluation: QualityEvaluation,
    kinematics: MujocoKinematics,
    frames: tuple[int, ...],
    path: Path,
) -> None:
    assert evaluation.bound_motion is not None
    info = kinematics.info
    figure = plt.figure(figsize=(3.2 * len(frames), 4.0))
    pelvis_id = info.body_names.index("pelvis")
    left_foot = info.body_names.index("left_ankle_roll_link")
    right_foot = info.body_names.index("right_ankle_roll_link")
    for plot_index, frame_index in enumerate(frames, start=1):
        snapshot = kinematics.forward_frame(evaluation.bound_motion, frame_index)
        positions = snapshot.body_pos_world_m.copy()
        positions[:, :2] -= positions[pelvis_id, :2]
        axis = figure.add_subplot(1, len(frames), plot_index, projection="3d")
        for body_id in range(1, info.nbody):
            parent = info.body_parent_ids[body_id]
            if parent <= 0:
                continue
            points = positions[[parent, body_id]]
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color="0.25", linewidth=1.2)
        axis.scatter(
            positions[1:, 0], positions[1:, 1], positions[1:, 2],
            color="tab:blue", s=7,
        )
        axis.scatter(
            positions[[left_foot, right_foot], 0],
            positions[[left_foot, right_foot], 1],
            positions[[left_foot, right_foot], 2],
            color="tab:red", s=16,
        )
        axis.set_xlim(-0.8, 0.8)
        axis.set_ylim(-0.8, 0.8)
        axis.set_zlim(-0.15, 1.75)
        axis.view_init(elev=15, azim=-65)
        axis.set_title(f"f={frame_index}\nt={frame_index / evaluation.bound_motion.record.fps:.1f}s", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([0.0, 1.0])
    figure.suptitle(
        f"{evaluation.metadata.motion_name} | {evaluation.metadata.motion_category} | "
        f"score={evaluation.scores.quality_score:.2f} | hard_fail={evaluation.decision.hard_fail}"
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _review_candidates(
    motions: pd.DataFrame,
    evaluations: Sequence[QualityEvaluation],
    run: EvaluationRunResult,
    analysis_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    by_name = {evaluation.metadata.motion_name: evaluation for evaluation in evaluations}
    outcome_by_name = {
        outcome.specification.motion_name: outcome for outcome in run.outcomes
    }
    selected: list[tuple[str, str]] = []
    for stratum in STEP10_STRATA:
        group = motions.loc[motions["motion_category"] == stratum]
        selected.append((str(group.loc[group["quality_score"].idxmin(), "motion_name"]), "lowest_score"))
        selected.append((str(group.loc[group["quality_score"].idxmax(), "motion_name"]), "highest_score"))
    montage_dir = analysis_dir / "manual_review_packet"
    montage_dir.mkdir()
    kinematics = MujocoKinematics.from_project_config(config)
    rows = []
    for name, reason in selected:
        evaluation = by_name[name]
        frame_indices = _review_frames(evaluation)
        montage = montage_dir / f"{name}_pose_montage.png"
        _plot_pose_montage(evaluation, kinematics, frame_indices, montage)
        frame_metrics_path = run.output_dir / "frame_metrics" / f"{name}.csv"
        anomaly_timeline = montage_dir / f"{name}_anomaly_timeline.png"
        _plot_anomaly_timeline(pd.read_csv(frame_metrics_path), anomaly_timeline)
        outcome = outcome_by_name[name]
        assert outcome.plots is not None and outcome.plots.fidelity_png is not None
        motion_row = motions.loc[motions["motion_name"] == name].iloc[0]
        rows.append(
            {
                "motion_name": name,
                "motion_category": evaluation.metadata.motion_category,
                "selection_reason": reason,
                "quality_score": motion_row["quality_score"],
                "hard_fail": motion_row["hard_fail"],
                "failure_reason_codes": motion_row["failure_reason_codes"],
                "review_frame_indices": json.dumps(frame_indices),
                "pose_montage": str(montage),
                "anomaly_timeline": str(anomaly_timeline),
                "joint_diagnostic": str(outcome.plots.joint_velocity_png),
                "contact_diagnostic": str(outcome.plots.foot_contact_png),
                "fidelity_diagnostic": str(outcome.plots.fidelity_png),
            }
        )
    return pd.DataFrame(rows)


def refresh_step10_review_packet(output_dir: str | Path) -> Path:
    """Add deterministic full-sequence anomaly timelines to an existing run."""

    destination = Path(output_dir).expanduser().resolve()
    if (destination / "step10_manifest.json").exists():
        raise ValueError(
            "A finalized step-10 run is immutable; refresh before finalization"
        )
    candidates_path = destination / "analysis" / "manual_review_candidates.csv"
    motion_results_path = destination / "analysis" / "motion_results.csv"
    if not candidates_path.is_file() or not motion_results_path.is_file():
        raise ValueError("Step-10 automated experiment outputs are incomplete")

    motions = pd.read_csv(motion_results_path)
    candidates = pd.read_csv(candidates_path)
    for table in (motions, candidates):
        if "failure_reason_codes" in table:
            table["failure_reason_codes"] = table["failure_reason_codes"].map(
                lambda value: json.dumps(
                    list(dict.fromkeys(json.loads(value))), separators=(",", ":")
                )
            )
    motions.to_csv(motion_results_path, index=False)

    timeline_paths: list[str] = []
    for row_index, row in candidates.iterrows():
        name = str(row["motion_name"])
        frame_metrics_path = (
            destination / "evaluation" / "frame_metrics" / f"{name}.csv"
        )
        if not frame_metrics_path.is_file():
            raise ValueError(f"Missing frame metrics for review candidate: {name}")
        timeline_path = (
            destination
            / "analysis"
            / "manual_review_packet"
            / f"{name}_anomaly_timeline.png"
        )
        _plot_anomaly_timeline(pd.read_csv(frame_metrics_path), timeline_path)
        timeline_paths.append(str(timeline_path))
        candidates.loc[row_index, "anomaly_timeline"] = str(timeline_path)
    candidates.to_csv(candidates_path, index=False)

    automated_path = destination / "automated_analysis.json"
    automated = json.loads(automated_path.read_text(encoding="utf-8"))
    automated["artifacts"]["review_anomaly_timelines"] = timeline_paths
    with automated_path.open("w", encoding="utf-8") as stream:
        json.dump(automated, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return candidates_path


def _plot_cohort_scores(motions: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(STEP10_STRATA))
    values = [
        motions.loc[motions["motion_category"] == stratum, "quality_score"].to_numpy()
        for stratum in STEP10_STRATA
    ]
    axis.boxplot(values, positions=positions, widths=0.5, showmeans=True)
    for index, group in enumerate(values):
        axis.scatter(np.full(group.shape, index), group, color="tab:blue", zorder=3)
    axis.set_xticks(positions, STEP10_STRATA, rotation=20, ha="right")
    axis.set_ylabel("v1 quality score [0,100]")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("Frozen step-10 cohort: within-stratum score distribution (n=3)")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_sensitivity(summary: pd.DataFrame, path: Path) -> None:
    data = summary.loc[(summary["scope"] == "all") & (summary["scenario_id"] != "baseline")].copy()
    data = data.sort_values("baseline_rank_spearman_rho")
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.barh(data["scenario_id"], data["baseline_rank_spearman_rho"], color="tab:purple")
    axis.axvline(1.0, color="black", linewidth=0.8)
    axis.set_xlim(-1.0, 1.05)
    axis.set_xlabel("Spearman rho versus baseline global ranking")
    axis.set_title("One-at-a-time threshold sensitivity")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_correlation(correlations: pd.DataFrame, path: Path) -> None:
    matrix = np.eye(len(SCORE_COLUMNS), dtype=np.float64)
    overall = correlations.loc[correlations["scope"] == "all"]
    for _, row in overall.iterrows():
        left = SCORE_COLUMNS.index(row["left_dimension"])
        right = SCORE_COLUMNS.index(row["right_dimension"])
        matrix[left, right] = matrix[right, left] = row["spearman_rho"]
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(SCORE_COLUMNS)), SCORE_COLUMNS, rotation=35, ha="right")
    axis.set_yticks(range(len(SCORE_COLUMNS)), SCORE_COLUMNS)
    for row in range(len(SCORE_COLUMNS)):
        for column in range(len(SCORE_COLUMNS)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Spearman rho")
    axis.set_title("Overall dimension correlations (n=15; descriptive only)")
    figure.tight_layout()
    _save_figure(figure, path)


def run_step10_experiment(
    project_config: dict[str, Any],
    catalog_csv: str | Path,
    selection_csv: str | Path,
    output_dir: str | Path,
    *,
    execution_command: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> Step10ExperimentResult:
    destination = Path(output_dir).expanduser().resolve()
    _prepare_output(destination)
    catalog = Path(catalog_csv).expanduser().resolve()
    selection = Path(selection_csv).expanduser().resolve()
    selections = load_step10_selection(selection)
    specifications = discover_inputs(
        input_path=None,
        input_dir=Path(catalog).parent / "retargeted",
        pattern="*.pkl",
        catalog_path=catalog,
    )
    run = run_evaluations(
        project_config,
        specifications,
        destination / "evaluation",
        plots_enabled=True,
        execution_command=execution_command,
        pattern="*.pkl",
        input_mode="step10_frozen_cohort",
        protected_input_dirs=(Path(catalog).parent / "retargeted",),
        progress=progress,
    )
    analysis_dir = destination / "analysis"
    analysis_dir.mkdir()
    motions = _motion_table(run.evaluations)
    themes = {row.motion_name: row.source_theme for row in selections}
    motions.insert(2, "source_theme", motions["motion_name"].map(themes))
    strata = _stratum_summary(motions)
    sensitivity_runs, sensitivity_summary = build_sensitivity_tables(
        run.evaluations, project_config
    )
    correlations = build_dimension_correlations(motions)
    candidates = _review_candidates(
        motions, run.evaluations, run, analysis_dir, project_config
    )

    paths = {
        "motion_results": analysis_dir / "motion_results.csv",
        "stratum_summary": analysis_dir / "stratum_summary.csv",
        "sensitivity_runs": analysis_dir / "sensitivity_runs.csv",
        "sensitivity_summary": analysis_dir / "sensitivity_summary.csv",
        "dimension_correlations": analysis_dir / "dimension_correlations_spearman.csv",
        "review_candidates": analysis_dir / "manual_review_candidates.csv",
        "cohort_plot": analysis_dir / "cohort_score_distribution.png",
        "sensitivity_plot": analysis_dir / "sensitivity_rank_stability.png",
        "correlation_plot": analysis_dir / "dimension_correlation_heatmap.png",
    }
    motions.to_csv(paths["motion_results"], index=False)
    strata.to_csv(paths["stratum_summary"], index=False)
    sensitivity_runs.to_csv(paths["sensitivity_runs"], index=False)
    sensitivity_summary.to_csv(paths["sensitivity_summary"], index=False)
    correlations.to_csv(paths["dimension_correlations"], index=False)
    candidates.to_csv(paths["review_candidates"], index=False)
    _plot_cohort_scores(motions, paths["cohort_plot"])
    _plot_sensitivity(sensitivity_summary, paths["sensitivity_plot"])
    _plot_correlation(correlations, paths["correlation_plot"])

    counts = motions["motion_category"].value_counts().to_dict()
    baseline_sensitivity = sensitivity_runs.loc[
        sensitivity_runs["scenario_id"] == "baseline"
    ].set_index("motion_name")
    automated_checks = {
        "all_15_evaluated": run.succeeded_count == 15 and run.failed_count == 0,
        "three_motions_per_stratum": all(counts.get(name) == 3 for name in STEP10_STRATA),
        "all_fidelity_enabled": all(evaluation.fidelity_metrics is not None for evaluation in run.evaluations),
        "all_exact_pairs": all(evaluation.metadata.frame_alignment == "exact" for evaluation in run.evaluations),
        "v1_score_frozen_but_uncalibrated": all(
            evaluation.scores is not None
            and evaluation.scores.score_version == "v1"
            and not evaluation.scores.score_is_calibrated
            for evaluation in run.evaluations
        ),
        "sensitivity_baseline_matches_formal_score": all(
            np.isclose(
                baseline_sensitivity.loc[evaluation.metadata.motion_name, "quality_score"],
                evaluation.scores.quality_score,
                atol=1e-12,
            )
            for evaluation in run.evaluations
        ),
        "sensitivity_baseline_matches_formal_hard_fail": all(
            bool(baseline_sensitivity.loc[evaluation.metadata.motion_name, "hard_fail"])
            == evaluation.decision.hard_fail
            for evaluation in run.evaluations
        ),
        "review_packet_frozen": len(candidates) == 10,
    }
    automated = {
        "schema_version": 1,
        "ok": all(automated_checks.values()),
        "manual_review_status": "pending",
        "automated_checks": automated_checks,
        "cohort": {
            "motion_count": len(motions),
            "total_frames": int(motions["num_frames"].sum()),
            "stratum_counts": counts,
            "catalog": str(catalog),
            "catalog_sha256": sha256_file(catalog),
            "selection": str(selection),
            "selection_sha256": sha256_file(selection),
        },
        "configuration": {
            "path": project_config["_config_path"],
            "sha256": sha256_file(project_config["_config_path"]),
            "score_version": project_config["scoring"]["version"],
            "score_is_calibrated": project_config["scoring"]["calibrated"],
        },
        "evaluation_manifest": str(run.run_manifest_json),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    automated_path = destination / "automated_analysis.json"
    with automated_path.open("w", encoding="utf-8") as stream:
        json.dump(automated, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return Step10ExperimentResult(
        output_dir=destination,
        evaluation_run=run,
        automated_analysis_json=automated_path,
        review_candidates_csv=paths["review_candidates"],
        ok=bool(automated["ok"]),
    )


def finalize_step10_experiment(
    output_dir: str | Path,
    review_yaml: str | Path,
) -> Step10FinalizationResult:
    destination = Path(output_dir).expanduser().resolve()
    automated_path = destination / "automated_analysis.json"
    candidates_path = destination / "analysis" / "manual_review_candidates.csv"
    if not automated_path.is_file() or not candidates_path.is_file():
        raise ValueError("Step-10 automated experiment outputs are incomplete")
    automated = json.loads(automated_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path)
    if "anomaly_timeline" not in candidates.columns:
        raise ValueError(
            "Step-10 review packet has no anomaly timelines; refresh it before review"
        )
    missing_review_artifacts = [
        path
        for column in (
            "pose_montage",
            "anomaly_timeline",
            "joint_diagnostic",
            "contact_diagnostic",
            "fidelity_diagnostic",
        )
        for path in candidates[column]
        if not Path(path).is_file()
    ]
    if missing_review_artifacts:
        raise ValueError(
            "Step-10 review artifacts are missing: "
            + ", ".join(sorted(missing_review_artifacts))
        )
    review_path = Path(review_yaml).expanduser().resolve()
    with review_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("reviews"), list):
        raise ValueError("Manual review YAML must contain a reviews list")
    reviews = pd.DataFrame(raw["reviews"])
    required = {
        "motion_name",
        "semantic_plausibility",
        "visible_artifact_severity",
        "metric_agreement",
        "review_notes",
        "reviewer",
        "reviewed_at",
    }
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError("Manual reviews are missing fields: " + ", ".join(missing))
    expected = set(candidates["motion_name"])
    actual = set(reviews["motion_name"])
    if expected != actual or len(reviews) != len(expected):
        raise ValueError(
            f"Manual review coverage mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    allowed = {
        "semantic_plausibility": {"plausible", "uncertain", "implausible"},
        "visible_artifact_severity": {"none", "mild", "moderate", "severe"},
        "metric_agreement": {"agree", "partial", "disagree"},
    }
    for column, values in allowed.items():
        invalid = sorted(set(reviews[column]) - values)
        if invalid:
            raise ValueError(f"Invalid {column} values: {invalid}")
    merged = candidates.merge(reviews, on="motion_name", validate="one_to_one")
    review_csv = destination / "analysis" / "manual_review.csv"
    merged.to_csv(review_csv, index=False)
    summary = {
        "schema_version": 1,
        "complete": True,
        "review_protocol": str(raw.get("protocol", "")),
        "review_protocol_sha256": (
            sha256_file(raw["protocol"]) if raw.get("protocol") else None
        ),
        "review_input": str(review_path),
        "review_input_sha256": sha256_file(review_path),
        "reviewed_motion_count": len(merged),
        "stratum_coverage": merged["motion_category"].value_counts().to_dict(),
        "semantic_plausibility_counts": merged["semantic_plausibility"].value_counts().to_dict(),
        "visible_artifact_severity_counts": merged["visible_artifact_severity"].value_counts().to_dict(),
        "metric_agreement_counts": merged["metric_agreement"].value_counts().to_dict(),
        "limitations": [
            "Six static FK snapshots plus full-sequence diagnostic plots are not a dynamics rollout.",
            "Three motions per stratum support descriptive statistics, not population inference.",
            "This review was a Codex structured visual audit, not an independent domain-expert human review.",
            "Structured visual review does not calibrate downstream robot-learning value.",
        ],
    }
    review_summary = destination / "analysis" / "manual_review_summary.json"
    with review_summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    automated["manual_review_status"] = "complete"
    with automated_path.open("w", encoding="utf-8") as stream:
        json.dump(automated, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if path.name == "step10_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(destination)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "complete": bool(automated["ok"] and summary["complete"]),
        "automated_analysis": automated,
        "manual_review": summary,
        "runtime": runtime_info(),
        "files": files,
    }
    manifest_path = destination / "step10_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return Step10FinalizationResult(
        output_dir=destination,
        review_csv=review_csv,
        review_summary_json=review_summary,
        manifest_json=manifest_path,
        ok=bool(manifest["complete"]),
    )
