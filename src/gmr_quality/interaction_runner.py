from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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
from scipy.stats import rankdata, spearmanr

from .config import load_config
from .evaluation import QualityEvaluation, evaluate_motion_result
from .interaction import (
    InteractionEvent,
    InteractionFrameMetrics,
    InteractionRecord,
    build_interaction_frame_rows,
    compute_interaction_frame_metrics,
    compute_interaction_metrics,
    extract_interaction_events,
    load_interaction_record,
)
from .io import validate_motion_payload
from .manifest import (
    git_state,
    package_version,
    runtime_info,
    sha256_file,
    write_resolved_config,
    write_run_manifest,
)
from .mujoco_kinematics import MujocoKinematics
from .reporting import build_event_rows, build_frame_rows, build_summary_row
from .types import MotionValidationConfig


@dataclass(frozen=True)
class InteractionRunResult:
    output_dir: Path
    discovered_count: int
    succeeded_count: int
    failed_count: int
    hard_fail_count: int
    quality_summary_csv: Path
    quality_events_csv: Path
    frame_metrics_dir: Path
    run_manifest_json: Path


@dataclass(frozen=True)
class QualityV2ValidationResult:
    output_dir: Path
    motion_count: int
    correlation_csv: Path
    group_summary_csv: Path
    decision_json: Path
    report_markdown: Path
    run_manifest_json: Path


def load_quality_assessment_v2_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("protocol_version") != "quality_assessment_v2":
        raise ValueError("Unsupported quality-assessment v2 protocol")
    robot_xml = Path(config["robot"]["xml"]).expanduser().resolve()
    if sha256_file(robot_xml) != config["robot"]["expected_xml_sha256"]:
        raise ValueError("Frozen G1 XML hash changed")
    base_config_path = Path(config["base_motion"]["project_config"]).expanduser().resolve()
    base_config = load_config(base_config_path)
    if Path(base_config["paths"]["robot_xml"]).resolve() != robot_xml:
        raise ValueError("Base-motion and interaction configurations use different robot XML files")
    combined_weights = config["combined_scoring"]["weights"]
    expected_dimensions = {
        "smoothness", "joint_feasibility", "robot_contact", "fidelity",
        "hand_object_proximity", "contact_timing", "relative_pose", "penetration",
        "grasp_stability", "task_effectiveness",
    }
    if set(combined_weights) != expected_dimensions or not np.isclose(
        sum(float(value) for value in combined_weights.values()), 1.0
    ):
        raise ValueError("combined_scoring weights must match all dimensions and sum to one")
    config["_config_path"] = str(source)
    config["_base_project_config"] = base_config
    return config


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return value or "interaction"


def _interaction_to_motion_load_result(
    record: InteractionRecord,
    project_config: dict[str, Any],
    joint_names: tuple[str, ...],
):
    qpos = np.asarray(record.robot_qpos, dtype=np.float64)
    payload = {
        "fps": record.fps,
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:36],
        "dof_names": list(joint_names),
    }
    return validate_motion_payload(
        payload,
        record.source_path,
        MotionValidationConfig.from_project_config(project_config),
    )


def _combined_score(
    base: QualityEvaluation,
    interaction: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if base.scores is None:
        raise ValueError("Combined scoring requires a valid base-motion evaluation")
    values = {
        "smoothness": base.scores.smoothness_score,
        "joint_feasibility": base.scores.joint_feasibility_score,
        "robot_contact": base.scores.contact_score,
        "fidelity": base.scores.fidelity_score,
        "hand_object_proximity": interaction["hand_object_proximity_score"],
        "contact_timing": interaction["contact_timing_score"],
        "relative_pose": interaction["relative_pose_score"],
        "penetration": interaction["penetration_score"],
        "grasp_stability": interaction["grasp_stability_score"],
        "task_effectiveness": interaction["task_effectiveness_score"],
    }
    weights = config["combined_scoring"]["weights"]
    available = tuple(name for name, value in values.items() if np.isfinite(value))
    unavailable = tuple(name for name in values if name not in available)
    weight_sum = sum(float(weights[name]) for name in available)
    if weight_sum <= 0:
        raise ValueError("No combined score dimension is available")
    normalized = {name: float(weights[name]) / weight_sum for name in available}
    score = float(sum(normalized[name] * float(values[name]) for name in available))
    failure_codes = tuple(
        dict.fromkeys(
            [
                *(reason.code for reason in base.decision.failure_reasons),
                *interaction["failure_reason_codes"],
            ]
        )
    )
    return {
        "quality_score_v2": score,
        "quality_score_v2_available_dimensions": available,
        "quality_score_v2_unavailable_dimensions": unavailable,
        "quality_score_v2_normalized_weights": normalized,
        "quality_score_v2_is_calibrated": False,
        "quality_score_v2_hard_fail": bool(base.decision.hard_fail or interaction["hard_fail"]),
        "quality_score_v2_failure_reason_codes": failure_codes,
    }


def _csv_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _interaction_event_row(event: InteractionEvent) -> dict[str, Any]:
    row = asdict(event)
    row["metric_domain"] = "interaction_task"
    row["entities"] = json.dumps(row["entities"], ensure_ascii=False)
    return row


def _base_event_rows(evaluation: QualityEvaluation) -> list[dict[str, Any]]:
    result = []
    for row in build_event_rows(evaluation):
        result.append({**row, "metric_domain": "robot_motion"})
    return result


def _merge_frame_rows(
    base: QualityEvaluation,
    interaction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rows = build_frame_rows(base)
    if len(base_rows) != len(interaction_rows):
        raise ValueError("Base-motion and interaction frame reports are not aligned")
    merged: list[dict[str, Any]] = []
    for base_row, interaction_row in zip(base_rows, interaction_rows):
        if base_row["frame"] != interaction_row["frame"]:
            raise ValueError("Frame indices are not aligned")
        extra = {
            key: value
            for key, value in interaction_row.items()
            if key not in {"motion_name", "frame", "time_s"}
        }
        merged.append({**base_row, **extra})
    return merged


def _plot_interaction(
    record: InteractionRecord,
    frame: InteractionFrameMetrics,
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    time = np.arange(record.num_frames) / record.fps
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, frame.signed_distance_m[:, 0], label="left signed distance")
    axes[0].plot(time, frame.signed_distance_m[:, 1], label="right signed distance")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("distance (m)")
    axes[0].legend(loc="best")
    axes[1].step(time, record.expected_contact[:, 0], where="post", label="left expected")
    axes[1].step(time, record.observed_contact[:, 0], where="post", label="left observed")
    axes[1].step(time, record.expected_contact[:, 1], where="post", label="right expected")
    axes[1].step(time, record.observed_contact[:, 1], where="post", label="right observed")
    axes[1].set_ylabel("contact")
    axes[1].legend(loc="best", ncol=2)
    target_error = np.linalg.norm(record.object_pos_m - record.target_pos_m, axis=1)
    axes[2].plot(time, target_error, label="object target error")
    axes[2].plot(time, np.linalg.norm(record.object_linvel_m_s, axis=1), label="object speed")
    axes[2].set_ylabel("m or m/s")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best")
    figure.suptitle(
        f"{record.name} | interaction={summary['interaction_quality_score']:.2f} | "
        f"task_success={summary['nominal_task_success']}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _discover(
    *, input_path: str | Path | None, input_dir: str | Path | None, pattern: str
) -> tuple[Path, ...]:
    if (input_path is None) == (input_dir is None):
        raise ValueError("Provide exactly one of input_path or input_dir")
    if input_path is not None:
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        return (source,)
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    sources = tuple(sorted(path.resolve() for path in directory.glob(pattern) if path.is_file()))
    if not sources:
        raise ValueError(f"No interaction files matched {pattern!r} in {directory}")
    return sources


def _all_output_files(output: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    ]


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(count, len(x)))
    xr = rankdata(x[indices], axis=1)
    yr = rankdata(y[indices], axis=1)
    xr -= xr.mean(axis=1, keepdims=True)
    yr -= yr.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(xr**2, axis=1) * np.sum(yr**2, axis=1))
    valid = denominator > 0
    values = np.sum(xr[valid] * yr[valid], axis=1) / denominator[valid]
    if not len(values):
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _spearman_diagnostic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[float, float, float, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    clean_x = x[finite]
    clean_y = y[finite]
    if (
        len(clean_x) < 3
        or np.ptp(clean_x) == 0.0
        or np.ptp(clean_y) == 0.0
    ):
        return float("nan"), float("nan"), float("nan"), len(clean_x)
    rho = float(spearmanr(clean_x, clean_y).statistic)
    low, high = _bootstrap_spearman(
        clean_x,
        clean_y,
        count=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return rho, low, high, len(clean_x)


def finalize_quality_v2_validation(
    config_path: str | Path,
    output_dir: str | Path,
    simulation_results_csv: str | Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260807,
) -> QualityV2ValidationResult:
    """Attach controlled-simulation diagnostics without claiming calibration."""

    config = load_quality_assessment_v2_config(config_path)
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "quality_assessment_v2_run_v1":
        raise ValueError("Output directory is not a quality-assessment v2 run")
    if not manifest.get("complete"):
        raise RuntimeError("Cannot finalize an incomplete quality-assessment v2 run")
    summary = pd.read_csv(output / "quality_summary.csv")
    simulation_path = Path(simulation_results_csv).expanduser().resolve()
    simulation = pd.read_csv(simulation_path)
    required_target_columns = (
        "motion_name", "quality_level", "robust_task_success_rate",
        "robust_final_position_error_mean_m", "robust_max_lift_mean_m",
    )
    if not set(required_target_columns) <= set(simulation.columns):
        raise ValueError("Simulation results are missing required robust task columns")
    target = simulation[list(required_target_columns)].copy()
    joined = summary.merge(target, on="motion_name", validate="one_to_one")
    if len(joined) != len(summary) or len(joined) < 4:
        raise ValueError("Every evaluated motion needs one robust simulation target; need at least four")
    analysis_dir = output / "validation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(analysis_dir / "quality_with_simulation_targets.csv", index=False)
    metrics = [
        "base_quality_score",
        "hand_object_proximity_score",
        "contact_timing_score",
        "relative_pose_score",
        "penetration_score",
        "grasp_stability_score",
        "task_effectiveness_score",
        "interaction_quality_score",
        "quality_score_v2",
    ]
    targets = {
        "robust_task_success_rate": joined["robust_task_success_rate"].to_numpy(dtype=float),
        "negative_robust_final_position_error": -joined[
            "robust_final_position_error_mean_m"
        ].to_numpy(dtype=float),
        "robust_max_lift": joined["robust_max_lift_mean_m"].to_numpy(dtype=float),
    }
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        x = joined[metric].to_numpy(dtype=float)
        for target_name, y in targets.items():
            rho, low, high, n_valid = _spearman_diagnostic(
                x,
                y,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
            rows.append(
                {
                    "offline_metric": metric,
                    "simulation_target": target_name,
                    "spearman_rho": rho,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "n_motions": n_valid,
                }
            )
    correlations = pd.DataFrame(rows)
    correlation_path = analysis_dir / "offline_simulation_correlations.csv"
    correlations.to_csv(correlation_path, index=False)
    group_summary = (
        joined.groupby("quality_level", sort=False)
        .agg(
            motion_count=("motion_name", "size"),
            base_quality_score_mean=("base_quality_score", "mean"),
            interaction_quality_score_mean=("interaction_quality_score", "mean"),
            task_effectiveness_score_mean=("task_effectiveness_score", "mean"),
            quality_score_v2_mean=("quality_score_v2", "mean"),
            robust_task_success_rate_mean=("robust_task_success_rate", "mean"),
            hard_fail_rate=("quality_score_v2_hard_fail", "mean"),
        )
        .reset_index()
    )
    group_path = analysis_dir / "quality_level_summary.csv"
    group_summary.to_csv(group_path, index=False)
    primary = correlations[
        (correlations.offline_metric == "quality_score_v2")
        & (correlations.simulation_target == "robust_task_success_rate")
    ].iloc[0]
    decision = {
        "score_is_calibrated": False,
        "diagnostic_spearman_rho": float(primary.spearman_rho),
        "diagnostic_bootstrap_ci95": [
            float(primary.bootstrap_ci95_low),
            float(primary.bootstrap_ci95_high),
        ],
        "n_motions": len(joined),
        "human_object_external_validation": False,
        "skill_learning_validation": False,
        "reason": (
            "The v2 proxy correction and weights were defined after inspecting the v1 "
            "controlled benchmark. These results are post-hoc controlled diagnostics, "
            "not independent calibration or evidence from human-object skill learning."
        ),
        "required_next_evidence": (
            "Freeze v2 before a new human-object or held-out operation cohort and evaluate "
            "policy learning value with source-motion-grouped validation."
        ),
    }
    decision_path = analysis_dir / "calibration_decision.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Quality assessment v2 controlled validation",
        "",
        f"- Evaluated interaction records: {len(joined)}",
        f"- Robust simulation target: `{simulation_path}`",
        f"- Diagnostic quality-v2 Spearman: {primary.spearman_rho:.4f}",
        f"- Bootstrap 95% CI: [{primary.bootstrap_ci95_low:.4f}, {primary.bootstrap_ci95_high:.4f}]",
        "- `score_is_calibrated=false`",
        "- `human_object_external_validation=false`",
        "- `skill_learning_validation=false`",
        "",
        "| Level | Base | Interaction | Task | Overall v2 | Robust success | Hard-fail rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in group_summary.iterrows():
        report_lines.append(
            f"| {row.quality_level} | {row.base_quality_score_mean:.3f} | "
            f"{row.interaction_quality_score_mean:.3f} | "
            f"{row.task_effectiveness_score_mean:.3f} | {row.quality_score_v2_mean:.3f} | "
            f"{row.robust_task_success_rate_mean:.3f} | {row.hard_fail_rate:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "This diagnostic demonstrates that the corrected evaluator is sensitive to the "
            "controlled task defects. It must not be described as independent calibration, "
            "validation on MotionDecode/human-captured object data, or robot skill-learning evidence.",
            "",
        ]
    )
    report_path = analysis_dir / "QUALITY_V2_RESULTS.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    manifest["validation"] = {
        **decision,
        "simulation_results_path": str(simulation_path),
        "simulation_results_sha256": sha256_file(simulation_path),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }
    project_root = Path(__file__).resolve().parents[2]
    for entry in manifest["source_code"]:
        source = project_root / entry["path"]
        entry.update(size_bytes=source.stat().st_size, sha256=sha256_file(source))
    manifest["outputs"] = _all_output_files(output)
    write_run_manifest(manifest, manifest_path)
    return QualityV2ValidationResult(
        output_dir=output,
        motion_count=len(joined),
        correlation_csv=correlation_path,
        group_summary_csv=group_path,
        decision_json=decision_path,
        report_markdown=report_path,
        run_manifest_json=manifest_path,
    )


def run_interaction_evaluations(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    pattern: str = "*.npz",
    plots_enabled: bool = True,
    execution_command: Iterable[str] = (),
) -> InteractionRunResult:
    config = load_quality_assessment_v2_config(config_path)
    base_config = config["_base_project_config"]
    config_source = Path(config["_config_path"])
    sources = _discover(input_path=input_path, input_dir=input_dir, pattern=pattern)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Interaction output directory must be empty: {output}")
    if input_dir is not None:
        protected = Path(input_dir).expanduser().resolve()
        if output == protected or protected in output.parents:
            raise ValueError("Output directory must not be inside the interaction input directory")
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "frame_metrics"
    plot_dir = output / "plots"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if plots_enabled:
        plot_dir.mkdir(parents=True, exist_ok=True)

    kinematics = MujocoKinematics.from_project_config(base_config)
    joint_names = tuple(kinematics.info.joint_names)
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for source in sources:
        outcome: dict[str, Any] = {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
        try:
            record = load_interaction_record(source)
            load_result = _interaction_to_motion_load_result(record, base_config, joint_names)
            base = evaluate_motion_result(
                load_result,
                kinematics,
                base_config,
                motion_name=record.name,
                motion_category=str(config["base_motion"]["motion_category"]),
                contact_policy=str(config["base_motion"]["contact_policy"]),
                notes="quality_assessment_v2 interaction record",
            )
            if not base.data_valid:
                raise ValueError(
                    "robot_qpos failed base-motion validation: "
                    + ",".join(base.load_result.validation.failure_codes)
                )
            frame = compute_interaction_frame_metrics(record, config)
            interaction = compute_interaction_metrics(record, config)
            combined = _combined_score(base, interaction, config)
            base_summary = build_summary_row(base)
            base_summary["base_quality_score"] = base_summary.pop("quality_score_v0")
            base_summary["base_score_version"] = base_summary.pop("score_version")
            base_summary["base_score_is_calibrated"] = base_summary.pop("score_is_calibrated")
            base_summary["base_hard_fail"] = base_summary.pop("hard_fail")
            base_summary["base_failure_reason_codes"] = base_summary.pop("failure_reason_codes")
            interaction_for_summary = {
                ("interaction_hard_fail" if key == "hard_fail" else key): value
                for key, value in interaction.items()
                if key not in {"motion_name", "num_frames", "duration_s"}
            }
            interaction_for_summary["interaction_failure_reason_codes"] = interaction_for_summary.pop(
                "failure_reason_codes"
            )
            row = {**base_summary, **interaction_for_summary, **combined}
            summary_rows.append({key: _csv_safe(value) for key, value in row.items()})
            interaction_rows = build_interaction_frame_rows(record, config, frame)
            merged_rows = _merge_frame_rows(base, interaction_rows)
            frame_path = frame_dir / f"{_safe_name(record.name)}.csv"
            pd.DataFrame(merged_rows).to_csv(frame_path, index=False)
            interaction_events = extract_interaction_events(record, config, frame, interaction)
            event_rows.extend(_base_event_rows(base))
            event_rows.extend(_interaction_event_row(event) for event in interaction_events)
            plots: list[str] = []
            if plots_enabled:
                from .plotting import plot_evaluation

                base_plots = plot_evaluation(base, plot_dir)
                plots.extend(
                    str(path.relative_to(output))
                    for path in (
                        base_plots.joint_velocity_png,
                        base_plots.foot_contact_png,
                        base_plots.fidelity_png,
                    )
                    if path is not None
                )
                interaction_plot = _plot_interaction(
                    record,
                    frame,
                    interaction,
                    plot_dir / f"{_safe_name(record.name)}_interaction.png",
                )
                plots.append(str(interaction_plot.relative_to(output)))
            outcome.update(
                {
                    "status": "evaluated",
                    "motion_name": record.name,
                    "hard_fail": bool(combined["quality_score_v2_hard_fail"]),
                    "failure_reason_codes": list(
                        combined["quality_score_v2_failure_reason_codes"]
                    ),
                    "frame_metrics": str(frame_path.relative_to(output)),
                    "plots": plots,
                }
            )
        except Exception as exc:
            outcome.update(
                {
                    "status": "failed_validation",
                    "motion_name": source.stem,
                    "hard_fail": True,
                    "failure_reason_codes": ["INTERACTION_INPUT_INVALID"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "plots": [],
                }
            )
            summary_rows.append(
                {
                    "motion_name": source.stem,
                    "interaction_path": str(source),
                    "data_valid": False,
                    "quality_score_v2": np.nan,
                    "quality_score_v2_is_calibrated": False,
                    "quality_score_v2_hard_fail": True,
                    "quality_score_v2_failure_reason_codes": json.dumps(
                        ["INTERACTION_INPUT_INVALID"]
                    ),
                    "error": outcome["error"],
                }
            )
        outcomes.append(outcome)

    summary_path = output / "quality_summary.csv"
    events_path = output / "quality_events.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    event_columns = [
        "motion_name", "metric_domain", "event_type", "start_frame", "end_frame",
        "start_time_s", "end_time_s", "duration_s", "entities", "max_severity", "details",
    ]
    pd.DataFrame(event_rows, columns=event_columns).to_csv(events_path, index=False)
    numeric = pd.read_csv(summary_path).select_dtypes(include=[np.number]).to_numpy()
    if np.isinf(numeric).any():
        raise RuntimeError("Formal quality summary contains Inf")
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    resolved_path = write_resolved_config(resolved, output / "resolved_config.yaml")
    source_entries = []
    for relative in config["reporting"]["hash_source_files"]:
        path = Path(__file__).resolve().parents[2] / relative
        source_entries.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": "quality_assessment_v2_run_v1",
        "complete": all(outcome["status"] == "evaluated" for outcome in outcomes),
        "score_is_calibrated": False,
        "human_object_external_validation": False,
        "tool": {
            "name": "gmr-quality-eval",
            "version": package_version(),
            "git": git_state(Path(__file__).resolve().parents[2]),
        },
        "runtime": runtime_info(),
        "configuration": {
            "path": str(config_source),
            "sha256": sha256_file(config_source),
            "resolved_path": str(resolved_path.relative_to(output)),
            "resolved_sha256": sha256_file(resolved_path),
            "base_project_config": str(Path(config["base_motion"]["project_config"]).resolve()),
            "base_project_config_sha256": sha256_file(config["base_motion"]["project_config"]),
        },
        "robot_model": {
            "xml": config["robot"]["xml"],
            "sha256": config["robot"]["expected_xml_sha256"],
            "nq": kinematics.info.nq,
            "nv": kinematics.info.nv,
        },
        "execution": {"command": list(execution_command), "pattern": pattern, "plots": plots_enabled},
        "counts": {
            "discovered": len(outcomes),
            "succeeded": sum(outcome["status"] == "evaluated" for outcome in outcomes),
            "failed": sum(outcome["status"] != "evaluated" for outcome in outcomes),
            "hard_fail": sum(bool(outcome["hard_fail"]) for outcome in outcomes),
        },
        "inputs": outcomes,
        "source_code": source_entries,
        "outputs": _all_output_files(output),
    }
    manifest_path = write_run_manifest(manifest, output / "run_manifest.json")
    return InteractionRunResult(
        output_dir=output,
        discovered_count=len(outcomes),
        succeeded_count=manifest["counts"]["succeeded"],
        failed_count=manifest["counts"]["failed"],
        hard_fail_count=manifest["counts"]["hard_fail"],
        quality_summary_csv=summary_path,
        quality_events_csv=events_path,
        frame_metrics_dir=frame_dir,
        run_manifest_json=manifest_path,
    )
