"""Train the pinned BeyondMimic G1 task from a local, boundary-aware NPZ.

Isaac Sim must be launched before most Isaac Lab imports, hence the seemingly
late imports below.  This script intentionally avoids WandB and writes only to
the explicitly supplied run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Local Step-11 BeyondMimic trainer")
parser.add_argument("--task", required=True)
parser.add_argument("--motion-file", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--max-iterations", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.run_dir.exists():
    raise FileExistsError(f"Refusing to overwrite existing run directory: {args.run_dir}")
if not args.motion_file.is_file():
    raise FileNotFoundError(args.motion_file)
with np.load(args.motion_file, allow_pickle=False) as motion_data:
    required = {
        "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w", "motion_starts", "motion_ends",
        "motion_names", "body_names",
    }
    missing = sorted(required.difference(motion_data.files))
    if missing:
        raise ValueError(f"Motion bundle missing fields: {missing}")
    if motion_data["joint_pos"].shape[1:] != (29,) or motion_data["body_pos_w"].shape[1:] != (30, 3):
        raise ValueError("Motion bundle has incompatible G1 dimensions")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner  # noqa: E402

from boundary_aware_command import BoundaryAwareMotionCommand  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_DEVICE_UNAVAILABLE: Isaac Lab policy training requires CUDA")
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    env_cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    env_cfg.commands.motion.class_type = BoundaryAwareMotionCommand
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = args.run_dir.name
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "launch.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "motion_file": str(args.motion_file.resolve()),
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations,
                "seed": args.seed,
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "sampler": "uniform_over_motion_then_time",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    runner = MotionOnPolicyRunner(
        wrapped,
        agent_cfg.to_dict(),
        log_dir=str(args.run_dir),
        device=agent_cfg.device,
        registry_name=None,
    )
    runner.add_git_repo_to_log(__file__)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # Upstream RSL-RL only checkpoints on save_interval boundaries (including
    # iteration zero).  Always persist the final formal policy explicitly.
    runner.save(str(args.run_dir / f"model_{agent_cfg.max_iterations}.pt"))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
