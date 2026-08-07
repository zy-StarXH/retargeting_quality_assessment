from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .beyondmimic import convert_gmr_motion, inspect_bmimic_file, merge_bmimic_motions
from .config import load_config
from .manifest import git_state, sha256_file


@dataclass(frozen=True)
class Step11Preparation:
    output_dir: Path
    cohort_csv: Path
    jobs_csv: Path
    environment_audit_json: Path
    data_manifest_json: Path
    ready_for_training: bool
    blocker_codes: tuple[str, ...]


def load_step11_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Step-11 config must be a mapping")
    if data.get("protocol_version") != "step11_beyondmimic_v1":
        raise ValueError("Unsupported step-11 protocol version")
    return data


def _run(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def audit_step11_environment(config: dict[str, Any]) -> dict[str, Any]:
    framework = config["framework"]
    python = str(Path(framework["python"]).expanduser().resolve())
    cuda_probe = _run(
        [
            python,
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),"
                "'cuda_device_count':torch.cuda.device_count()}))"
            ),
        ]
    )
    cuda_payload: dict[str, Any] = {}
    if cuda_probe["returncode"] == 0:
        try:
            cuda_payload = json.loads(cuda_probe["stdout"].splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            pass
    nvidia_devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    nvidia_compute_devices = sorted(
        str(path) for path in Path("/dev").glob("nvidia[0-9]*") if path.name[6:].isdigit()
    )
    repository = Path(framework["repository"]).expanduser().resolve()
    isaaclab = Path(framework["isaaclab_repository"]).expanduser().resolve()
    repository_state = git_state(repository)
    isaaclab_state = git_state(isaaclab)
    checks = {
        "framework_repository_exists": repository.is_dir(),
        "framework_commit_matches": repository_state.get("commit") == framework["expected_commit"],
        "framework_worktree_clean": repository_state.get("dirty") is False,
        "isaaclab_repository_exists": isaaclab.is_dir(),
        "isaaclab_commit_matches": isaaclab_state.get("commit") == framework["expected_isaaclab_commit"],
        "isaaclab_python_exists": Path(python).is_file(),
        "cuda_available": cuda_payload.get("cuda_available") is True,
        "cuda_device_count_positive": int(cuda_payload.get("cuda_device_count", 0)) > 0,
        "nvidia_device_nodes_present": bool(nvidia_devices),
        "nvidia_compute_device_present": bool(nvidia_compute_devices),
        "nvidia_visible_devices_not_void": os.environ.get("NVIDIA_VISIBLE_DEVICES") not in (None, "", "void"),
    }
    blocker_codes: list[str] = []
    if not all(
        checks[key]
        for key in ("framework_repository_exists", "framework_commit_matches", "isaaclab_repository_exists", "isaaclab_commit_matches", "isaaclab_python_exists")
    ):
        blocker_codes.append("TRAINING_FRAMEWORK_MISMATCH")
    if not all(
        checks[key]
        for key in ("cuda_available", "cuda_device_count_positive", "nvidia_compute_device_present", "nvidia_visible_devices_not_void")
    ):
        blocker_codes.append("GPU_DEVICE_UNAVAILABLE")
    return {
        "audit_version": "step11_environment_v1",
        "hostname": platform.node(),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "nvidia_device_nodes": nvidia_devices,
        "nvidia_compute_devices": nvidia_compute_devices,
        "checks": checks,
        "blocker_codes": blocker_codes,
        "ready_for_training": not blocker_codes,
        "cuda_probe": cuda_probe,
        "cuda": cuda_payload,
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"]),
        "framework_git": repository_state,
        "isaaclab_git": isaaclab_state,
    }


def build_step11_cohorts(config: dict[str, Any]) -> pd.DataFrame:
    results_path = Path(config["inputs"]["step10_motion_results"]).expanduser().resolve()
    catalog_path = Path(config["inputs"]["step10_catalog"]).expanduser().resolve()
    results = pd.read_csv(results_path)
    catalog = pd.read_csv(catalog_path)[["motion_name", "pkl_path", "pkl_sha256"]]
    table = results.merge(catalog, on=["motion_name", "pkl_path"], how="inner", validate="one_to_one")
    strata = list(config["cohorts"]["strata"])
    if len(table) != 15 or set(table["motion_category"]) != set(strata):
        raise ValueError("Step-11 requires the complete frozen 15-motion/five-stratum Step-10 cohort")
    if set(table.groupby("motion_category").size()) != {3}:
        raise ValueError("Each Step-11 stratum must contain exactly three motions")
    if not all(sha256_file(path) == expected for path, expected in zip(table["pkl_path"], table["pkl_sha256"])):
        raise ValueError("A frozen Step-10 PKL hash changed")

    rng = np.random.default_rng(int(config["cohorts"]["random_selection_seed"]))
    rows: list[dict[str, Any]] = []
    for stratum in strata:
        group = table[table["motion_category"] == stratum].sort_values(
            ["quality_score", "motion_name"], ascending=[False, True]
        )
        top = group.iloc[0]
        bottom = group.iloc[-1]
        random = group.iloc[int(rng.integers(0, len(group)))]
        selections = {"top_k": top, "random_k": random, "bottom_k": bottom}
        for cohort, item in selections.items():
            rows.append(
                {
                    "cohort": cohort,
                    "motion_name": item["motion_name"],
                    "motion_category": stratum,
                    "quality_score_v1": float(item["quality_score"]),
                    "stratum_rank": int(item["stratum_rank"]),
                    "pkl_path": item["pkl_path"],
                    "pkl_sha256": item["pkl_sha256"],
                    "selection_seed": int(config["cohorts"]["random_selection_seed"]) if cohort == "random_k" else "",
                }
            )
    for _, item in table.sort_values(["motion_category", "motion_name"]).iterrows():
        rows.append(
            {
                "cohort": "full",
                "motion_name": item["motion_name"],
                "motion_category": item["motion_category"],
                "quality_score_v1": float(item["quality_score"]),
                "stratum_rank": int(item["stratum_rank"]),
                "pkl_path": item["pkl_path"],
                "pkl_sha256": item["pkl_sha256"],
                "selection_seed": "",
            }
        )
    return pd.DataFrame(rows)


def _write_jobs(config: dict[str, Any], cohort_bundles: dict[str, Path], output_dir: Path) -> Path:
    integrations = Path(__file__).resolve().parents[2] / "integrations" / "beyondmimic"
    rows: list[dict[str, Any]] = []
    for cohort in config["cohorts"]["groups"]:
        for seed in config["training"]["policy_seeds"]:
            run_dir = output_dir / "runs" / cohort / f"seed_{seed}"
            command = [
                config["framework"]["python"],
                str(integrations / "train_local.py"),
                "--task", config["framework"]["task"],
                "--motion-file", str(cohort_bundles[cohort]),
                "--num-envs", str(config["training"]["num_envs"]),
                "--max-iterations", str(config["training"]["max_iterations"]),
                "--seed", str(seed),
                "--run-dir", str(run_dir),
                "--headless",
            ]
            rows.append(
                {
                    "job_id": f"{cohort}_seed_{seed}",
                    "cohort": cohort,
                    "seed": seed,
                    "motion_bundle": str(cohort_bundles[cohort]),
                    "motion_bundle_sha256": sha256_file(cohort_bundles[cohort]),
                    "run_dir": str(run_dir),
                    "result_json": str(run_dir / "evaluation" / "result.json"),
                    "status": "blocked_gpu",
                    "command_json": json.dumps(command, ensure_ascii=False),
                }
            )
    destination = output_dir / "training_jobs.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def prepare_step11_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    progress: Any = None,
) -> Step11Preparation:
    config_path = Path(config_path).expanduser().resolve()
    config = load_step11_config(config_path)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Step-11 output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    audit = audit_step11_environment(config)
    audit_path = destination / "environment_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cohorts = build_step11_cohorts(config)
    cohort_csv = destination / "cohort_selection.csv"
    cohorts.to_csv(cohort_csv, index=False)
    project_config = load_config(config["inputs"]["project_config"])
    common_frames = int(config["cohorts"]["common_source_frames_per_motion"])
    unique = cohorts.drop_duplicates("motion_name").sort_values("motion_name")
    converted: dict[str, Path] = {}
    conversion_rows: list[dict[str, Any]] = []
    for _, row in unique.iterrows():
        if progress:
            progress(f"Converting {row['motion_name']} to BeyondMimic NPZ")
        result = convert_gmr_motion(
            project_config,
            row["motion_name"],
            row["pkl_path"],
            destination / "motions" / f"{row['motion_name']}.npz",
            selected_frames=common_frames,
            output_fps=float(config["cohorts"]["training_fps"]),
            normalize_xy_origin=bool(config["cohorts"]["normalize_xy_origin"]),
        )
        converted[result.motion_name] = result.output_npz
        conversion_rows.append({**asdict(result), "source_pkl": str(result.source_pkl), "output_npz": str(result.output_npz)})

    bundles: dict[str, Path] = {}
    for cohort in config["cohorts"]["groups"]:
        names = cohorts.loc[cohorts["cohort"] == cohort, "motion_name"].tolist()
        bundles[cohort] = merge_bmimic_motions(
            [converted[name] for name in names], destination / "cohorts" / f"{cohort}.npz"
        )
    jobs_csv = _write_jobs(config, bundles, destination)
    resolved = destination / "resolved_step11_config.yaml"
    resolved.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    files = [config_path, Path(config["inputs"]["step10_manifest"]), cohort_csv, jobs_csv, audit_path, resolved]
    files.extend(converted.values())
    files.extend(bundles.values())
    manifest = {
        "manifest_version": "step11_preparation_v1",
        "complete": False,
        "preparation_complete": True,
        "training_complete": False,
        "ready_for_training": audit["ready_for_training"],
        "blocker_codes": audit["blocker_codes"],
        "cohort_counts": cohorts.groupby("cohort").size().astype(int).to_dict(),
        "training_job_count": len(config["cohorts"]["groups"]) * len(config["training"]["policy_seeds"]),
        "conversion": conversion_rows,
        "cohort_bundles": {name: inspect_bmimic_file(path) for name, path in bundles.items()},
        "files": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    manifest_path = destination / "step11_preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return Step11Preparation(
        output_dir=destination,
        cohort_csv=cohort_csv,
        jobs_csv=jobs_csv,
        environment_audit_json=audit_path,
        data_manifest_json=manifest_path,
        ready_for_training=bool(audit["ready_for_training"]),
        blocker_codes=tuple(audit["blocker_codes"]),
    )
