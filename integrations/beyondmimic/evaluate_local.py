"""Deterministic downstream evaluation for one Step-11 policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a local Step-11 BeyondMimic checkpoint")
parser.add_argument("--task", required=True)
parser.add_argument("--motion-file", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--result-json", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--episodes-per-motion", type=int, default=100)
parser.add_argument("--seed", type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

for path in (args.motion_file, args.checkpoint):
    if not path.is_file():
        raise FileNotFoundError(path)

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

from boundary_aware_command import BoundaryAwareMotionCommand  # noqa: E402


def _convergence_iteration(run_dir: Path) -> int | None:
    """First iteration reaching 90% of the final 10%-window timeout rate."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        return None
    accumulator = EventAccumulator(str(run_dir)).Reload()
    candidates = [tag for tag in accumulator.Tags().get("scalars", []) if "Termination/time_out" in tag]
    if not candidates:
        return None
    values = accumulator.Scalars(candidates[0])
    if len(values) < 10:
        return None
    y = np.asarray([value.value for value in values], dtype=float)
    window = max(5, len(y) // 20)
    smooth = np.convolve(y, np.ones(window) / window, mode="valid")
    plateau = float(np.mean(y[-max(5, len(y) // 10) :]))
    if plateau <= 0.05:
        return None
    reached = np.flatnonzero(smooth >= 0.9 * plateau)
    return int(values[int(reached[0]) + window - 1].step) if len(reached) else None


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_DEVICE_UNAVAILABLE: Isaac Lab policy evaluation requires CUDA")
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    env_cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    env_cfg.commands.motion.class_type = BoundaryAwareMotionCommand
    env_cfg.observations.policy.enable_corruption = False
    # Keep physical domain randomization enabled and seed-controlled; disabling
    # it would measure nominal replay only and overstate robustness.
    agent_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    command = wrapped.unwrapped.command_manager.get_term("motion")
    # CommandTerm.reset() resamples and writes the robot state, but the derived
    # aligned body target is first populated by _update_command().  Populate it
    # before the first measured step so MPJPE does not accidentally include
    # the vectorized environment-origin grid.
    command._update_command()
    obs, _ = wrapped.get_observations()
    names = list(command.motion_names)
    target = int(args.episodes_per_motion)
    completed = np.zeros(len(names), dtype=np.int64)
    success = np.zeros(len(names), dtype=np.int64)
    falls = np.zeros(len(names), dtype=np.int64)
    episode_lengths: list[list[float]] = [[] for _ in names]
    mpjpe: list[list[float]] = [[] for _ in names]
    weighted_mpjpe: list[list[float]] = [[] for _ in names]
    error_sum = torch.zeros(args.num_envs, device=wrapped.unwrapped.device)
    weighted_error_sum = torch.zeros_like(error_sum)
    step_count = torch.zeros(args.num_envs, dtype=torch.long, device=wrapped.unwrapped.device)
    body_weights = torch.tensor(
        [1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2],
        dtype=torch.float32,
        device=wrapped.unwrapped.device,
    )
    while np.any(completed < target):
        motion_ids = command.current_motion_ids.clone()
        lengths_before = wrapped.unwrapped.episode_length_buf.clone()
        errors = torch.norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)
        error_sum += errors.mean(dim=-1)
        weighted_error_sum += (errors * body_weights).sum(dim=-1) / body_weights.sum()
        step_count += 1
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if len(done_ids) == 0:
            continue
        terminations = wrapped.unwrapped.termination_manager
        timed_out = terminations.time_outs[done_ids]
        fell = (
            terminations.get_term("anchor_pos")[done_ids]
            | terminations.get_term("anchor_ori")[done_ids]
        )
        for env_id, motion_id, timeout, fall in zip(
            done_ids.cpu().tolist(),
            motion_ids[done_ids].cpu().tolist(),
            timed_out.cpu().tolist(),
            fell.cpu().tolist(),
        ):
            if completed[motion_id] >= target:
                continue
            completed[motion_id] += 1
            success[motion_id] += int(timeout)
            falls[motion_id] += int(fall)
            episode_lengths[motion_id].append(float(lengths_before[env_id].item() + 1) * wrapped.unwrapped.step_dt)
            count = max(int(step_count[env_id].item()), 1)
            mpjpe[motion_id].append(float(error_sum[env_id].item()) / count)
            weighted_mpjpe[motion_id].append(float(weighted_error_sum[env_id].item()) / count)
        error_sum[done_ids] = 0.0
        weighted_error_sum[done_ids] = 0.0
        step_count[done_ids] = 0

    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "motion_name": name,
                "episodes": int(completed[index]),
                "success_rate": float(success[index] / completed[index]),
                "episode_length_s": float(np.mean(episode_lengths[index])),
                "mpjpe_m": float(np.mean(mpjpe[index])),
                "weighted_mpjpe_m": float(np.mean(weighted_mpjpe[index])),
                "fall_rate": float(falls[index] / completed[index]),
            }
        )
    convergence = _convergence_iteration(args.checkpoint.resolve().parent)
    result = {
        "schema_version": "step11_policy_result_v1",
        "seed": args.seed,
        "motion_file": str(args.motion_file.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "deterministic_policy": True,
        "episodes_per_motion": target,
        "convergence_iteration": convergence,
        "converged": convergence is not None,
        "motions": rows,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
