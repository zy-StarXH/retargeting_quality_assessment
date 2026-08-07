"""Resumable GPU orchestrator for the frozen Step-11 jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


def checkpoint_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def run_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_DEVICE_UNAVAILABLE")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    jobs_path = output / "training_jobs.csv"
    jobs = pd.read_csv(jobs_path)
    integration = Path(__file__).resolve().parent
    python = str(Path(config["framework"]["python"]).resolve())
    task = config["framework"]["task"]
    max_iterations = int(config["training"]["max_iterations"])
    eval_cfg = config["evaluation"]
    jobs["max_iterations"] = max_iterations
    jobs["num_envs"] = int(config["training"]["num_envs"])
    jobs["status"] = jobs["status"].replace({"blocked_gpu": "ready"})
    jobs.to_csv(jobs_path, index=False)

    pending = [index for index, row in jobs.iterrows() if not Path(row.result_json).is_file()]
    for wave_start in range(0, len(pending), args.max_parallel):
        wave = pending[wave_start : wave_start + args.max_parallel]
        processes: list[tuple[int, subprocess.Popen, object]] = []
        for index in wave:
            row = jobs.loc[index]
            run_dir = Path(row.run_dir)
            command = [
                python, str(integration / "train_local.py"), "--task", task,
                "--motion-file", row.motion_bundle,
                "--num-envs", str(config["training"]["num_envs"]),
                "--max-iterations", str(max_iterations), "--seed", str(int(row.seed)),
                "--run-dir", str(run_dir), "--headless",
            ]
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            log_path = output / "logs" / f"{row.job_id}_train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            processes.append((index, process, stream))
            jobs.loc[index, "status"] = "training"
            jobs.loc[index, "command_json"] = json.dumps(command)
            print(f"START train {row.job_id} pid={process.pid}", flush=True)
        jobs.to_csv(jobs_path, index=False)
        for index, process, stream in processes:
            code = process.wait()
            stream.close()
            row = jobs.loc[index]
            if code != 0:
                jobs.loc[index, "status"] = "train_failed"
                jobs.to_csv(jobs_path, index=False)
                raise RuntimeError(f"Training failed for {row.job_id}; see logs")
            checkpoints = sorted(Path(row.run_dir).glob("model_*.pt"), key=checkpoint_key)
            if not checkpoints or checkpoint_key(checkpoints[-1]) != max_iterations:
                raise RuntimeError(f"Final checkpoint missing for {row.job_id}")
            jobs.loc[index, "checkpoint"] = str(checkpoints[-1])
            jobs.loc[index, "status"] = "trained"
            print(f"DONE train {row.job_id} checkpoint={checkpoints[-1].name}", flush=True)
        jobs.to_csv(jobs_path, index=False)

        eval_processes: list[tuple[int, subprocess.Popen, object]] = []
        for index in wave:
            row = jobs.loc[index]
            result = Path(row.result_json)
            command = [
                python, str(integration / "evaluate_local.py"), "--task", task,
                "--motion-file", row.motion_bundle, "--checkpoint", row.checkpoint,
                "--result-json", str(result), "--num-envs", "512",
                "--episodes-per-motion", str(eval_cfg["episodes_per_motion_per_seed"]),
                "--seed", str(int(row.seed)), "--headless",
            ]
            log_path = output / "logs" / f"{row.job_id}_eval.log"
            stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            eval_processes.append((index, process, stream))
            jobs.loc[index, "status"] = "evaluating"
            jobs.loc[index, "evaluation_command_json"] = json.dumps(command)
            print(f"START eval {row.job_id} pid={process.pid}", flush=True)
        jobs.to_csv(jobs_path, index=False)
        for index, process, stream in eval_processes:
            code = process.wait()
            stream.close()
            row = jobs.loc[index]
            if code != 0 or not Path(row.result_json).is_file():
                jobs.loc[index, "status"] = "eval_failed"
                jobs.to_csv(jobs_path, index=False)
                raise RuntimeError(f"Evaluation failed for {row.job_id}; see logs")
            jobs.loc[index, "status"] = "complete"
            print(f"DONE eval {row.job_id}", flush=True)
        jobs.to_csv(jobs_path, index=False)
    print(f"ALL COMPLETE results={sum(Path(path).is_file() for path in jobs.result_json)}/{len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
