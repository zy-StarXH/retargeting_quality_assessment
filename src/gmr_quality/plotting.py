from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gmr-quality-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import QualityEvaluation


@dataclass(frozen=True)
class PlotPaths:
    joint_velocity_png: Path
    foot_contact_png: Path
    fidelity_png: Path | None


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe:
        raise ValueError("motion_name cannot produce an empty plot filename")
    return safe


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=140,
        bbox_inches="tight",
        metadata={"Software": "gmr-quality-eval"},
    )
    plt.close(figure)


def plot_evaluation(
    evaluation: QualityEvaluation,
    output_dir: str | Path,
) -> PlotPaths:
    """Create deterministic diagnostic plots for one valid reference trajectory."""
    if (
        evaluation.bound_motion is None
        or evaluation.kinematic_metrics is None
        or evaluation.contact_metrics is None
    ):
        raise ValueError("Diagnostic plots require a valid evaluated motion")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    name = _safe_name(evaluation.metadata.motion_name)
    motion = evaluation.bound_motion.record
    kinematic = evaluation.kinematic_metrics
    contact = evaluation.contact_metrics
    time_s = np.arange(motion.num_frames, dtype=np.float64) / motion.fps

    joint_path = destination / f"{name}_joint_velocity.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_s, kinematic.smoothness.max_joint_step_rad, linewidth=0.8)
    axes[0].axhline(
        kinematic.smoothness.summary.joint_jump_threshold_rad,
        color="tab:red",
        linestyle="--",
        linewidth=0.9,
        label="joint-jump threshold",
    )
    axes[0].set_ylabel("max step [rad]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, kinematic.smoothness.max_joint_vel_rad_s, linewidth=0.8)
    axes[1].axhline(
        float(np.max(kinematic.joint_feasibility.velocity_limit_rad_s)),
        color="tab:red",
        linestyle="--",
        linewidth=0.9,
        label="configured velocity threshold",
    )
    axes[1].set_ylabel("max |velocity| [rad/s]")
    axes[1].legend(loc="upper right")

    axes[2].plot(time_s, kinematic.smoothness.max_joint_jerk_rad_s3, linewidth=0.8)
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].set_ylabel("max |jerk| [rad/s³]")
    axes[2].set_xlabel("time [s]")
    figure.suptitle(
        f"{evaluation.metadata.motion_name}: reference-trajectory kinematics"
    )
    figure.tight_layout()
    _save_figure(figure, joint_path)

    foot_path = destination / f"{name}_foot_contact.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_s, contact.left_foot_height_m, label="left")
    axes[0].plot(time_s, contact.right_foot_height_m, label="right")
    near = contact.summary.foot_near_ground_threshold_m
    axes[0].axhspan(-near, near, color="tab:green", alpha=0.12)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("sole height [m]")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        time_s,
        contact.left_foot_horizontal_speed_m_s,
        label="left",
    )
    axes[1].plot(
        time_s,
        contact.right_foot_horizontal_speed_m_s,
        label="right",
    )
    axes[1].axhline(
        contact.summary.foot_slide_speed_threshold_m_s,
        color="tab:red",
        linestyle="--",
        linewidth=0.9,
        label="slide threshold",
    )
    axes[1].set_ylabel("horizontal speed [m/s]")
    axes[1].legend(loc="upper right")

    axes[2].step(
        time_s,
        contact.left_contact_proxy_frame.astype(float),
        where="post",
        label="left proxy",
    )
    axes[2].step(
        time_s,
        contact.right_contact_proxy_frame.astype(float) + 1.2,
        where="post",
        label="right proxy (+1.2)",
    )
    axes[2].step(
        time_s,
        contact.left_mujoco_ground_contact_frame.astype(float) + 2.4,
        where="post",
        label="left MuJoCo (+2.4)",
    )
    axes[2].step(
        time_s,
        contact.right_mujoco_ground_contact_frame.astype(float) + 3.6,
        where="post",
        label="right MuJoCo (+3.6)",
    )
    axes[2].set_ylabel("contact flags")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", ncol=2)
    figure.suptitle(
        f"{evaluation.metadata.motion_name}: heuristic and MuJoCo foot contact"
    )
    figure.tight_layout()
    _save_figure(figure, foot_path)
    fidelity_path: Path | None = None
    fidelity = evaluation.fidelity_metrics
    if fidelity is not None:
        fidelity_path = destination / f"{name}_fidelity.png"
        figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(
            time_s,
            fidelity.keybody_pos_error_frame_mean_m,
            label="key-body mean",
        )
        axes[0].plot(
            time_s,
            fidelity.end_effector_pos_error_frame_mean_m,
            label="end-effector mean",
        )
        axes[0].plot(
            time_s,
            fidelity.keybody_pos_error_frame_max_m,
            linewidth=0.7,
            alpha=0.65,
            label="key-body max",
        )
        axes[0].axhline(
            fidelity.summary.fidelity_position_threshold_m,
            color="tab:red",
            linestyle="--",
            linewidth=0.9,
            label="frame-mean threshold",
        )
        axes[0].set_ylabel("position error [m]")
        axes[0].legend(loc="upper right")

        axes[1].plot(
            time_s,
            fidelity.link_orientation_error_frame_mean_deg,
            label="link mean",
        )
        axes[1].plot(
            time_s,
            fidelity.link_orientation_error_frame_max_deg,
            linewidth=0.7,
            alpha=0.65,
            label="link max",
        )
        axes[1].axhline(
            fidelity.summary.fidelity_orientation_threshold_deg,
            color="tab:red",
            linestyle="--",
            linewidth=0.9,
            label="frame-mean threshold",
        )
        axes[1].set_ylabel("orientation error [deg]")
        axes[1].legend(loc="upper right")

        axes[2].step(
            time_s,
            fidelity.fidelity_valid_frame.astype(float),
            where="post",
        )
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("fidelity pass")
        axes[2].set_xlabel("time [s]")
        figure.suptitle(
            f"{evaluation.metadata.motion_name}: verified source-motion fidelity"
        )
        figure.tight_layout()
        _save_figure(figure, fidelity_path)
    return PlotPaths(
        joint_velocity_png=joint_path,
        foot_contact_png=foot_path,
        fidelity_png=fidelity_path,
    )
