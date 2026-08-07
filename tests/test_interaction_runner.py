from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmr_quality.interaction_runner import (
    finalize_quality_v2_validation,
    load_quality_assessment_v2_config,
    run_interaction_evaluations,
)
from gmr_quality.manifest import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "quality_assessment_v2.yaml"
DEMONSTRATIONS = PROJECT_ROOT / "outputs" / "step12_interaction_experiment" / "demonstrations"
SIMULATION_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "step12_interaction_experiment"
    / "interaction_learning_value.csv"
)


def test_quality_v2_config_locks_base_and_robot_model() -> None:
    config = load_quality_assessment_v2_config(CONFIG)
    assert config["combined_scoring"]["calibrated"] is False
    assert sum(config["combined_scoring"]["weights"].values()) == pytest.approx(1.0)
    assert config["robot"]["expected_xml_sha256"] == config["_base_project_config"]["robot"][
        "expected_xml_sha256"
    ]


def test_single_interaction_run_combines_all_quality_domains(tmp_path: Path) -> None:
    result = run_interaction_evaluations(
        CONFIG,
        tmp_path / "result",
        input_path=DEMONSTRATIONS / "clean_seed_101.npz",
        plots_enabled=True,
        execution_command=("test", "evaluate-interaction"),
    )
    assert result.succeeded_count == 1
    assert result.failed_count == 0
    assert result.hard_fail_count == 0
    summary = pd.read_csv(result.quality_summary_csv)
    required = {
        "smoothness_score", "joint_feasibility_score", "contact_score",
        "hand_object_proximity_score", "contact_timing_score", "relative_pose_score",
        "penetration_score", "grasp_stability_score", "task_effectiveness_score",
        "quality_score_v2", "quality_score_v2_is_calibrated",
        "quality_score_v2_hard_fail",
    }
    assert required <= set(summary.columns)
    assert np.isfinite(summary[list(required - {"quality_score_v2_is_calibrated", "quality_score_v2_hard_fail"})].to_numpy()).all()
    assert bool(summary.loc[0, "quality_score_v2_is_calibrated"]) is False
    frame = pd.read_csv(next(result.frame_metrics_dir.glob("*.csv")))
    assert len(frame) == 300
    assert {"joint_jump_flag", "left_contact_miss_flag", "object_x_m"} <= set(frame.columns)
    assert (result.output_dir / "plots" / "clean_seed_101_interaction.png").is_file()
    manifest = json.loads(result.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["score_is_calibrated"] is False
    assert manifest["human_object_external_validation"] is False
    assert {entry["path"] for entry in manifest["source_code"]} == {
        "src/gmr_quality/interaction.py",
        "src/gmr_quality/interaction_runner.py",
        "src/gmr_quality/cli.py",
    }
    for entry in manifest["outputs"]:
        path = result.output_dir / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert sha256_file(path) == entry["sha256"]


def test_batch_isolates_invalid_record_and_keeps_valid_report(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copy2(DEMONSTRATIONS / "clean_seed_101.npz", inputs / "clean.npz")
    np.savez_compressed(inputs / "bad.npz", schema_version=np.asarray("gmr_interaction_v2"))
    result = run_interaction_evaluations(
        CONFIG,
        tmp_path / "result",
        input_dir=inputs,
        plots_enabled=False,
    )
    assert result.discovered_count == 2
    assert result.succeeded_count == 1
    assert result.failed_count == 1
    summary = pd.read_csv(result.quality_summary_csv)
    assert len(summary) == 2
    assert set(summary["data_valid"].astype(bool)) == {False, True}
    manifest = json.loads(result.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["counts"] == {
        "discovered": 2,
        "succeeded": 1,
        "failed": 1,
        "hard_fail": 1,
    }


def test_output_must_not_overwrite_or_live_under_input_directory(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copy2(DEMONSTRATIONS / "clean_seed_101.npz", inputs / "clean.npz")
    with pytest.raises(ValueError, match="must not be inside"):
        run_interaction_evaluations(
            CONFIG,
            inputs / "reports",
            input_dir=inputs,
            plots_enabled=False,
        )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        run_interaction_evaluations(
            CONFIG,
            occupied,
            input_path=inputs / "clean.npz",
            plots_enabled=False,
        )
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_controlled_validation_is_diagnostic_not_calibration(tmp_path: Path) -> None:
    output = tmp_path / "quality"
    run_interaction_evaluations(
        CONFIG,
        output,
        input_dir=DEMONSTRATIONS,
        pattern="*_seed_101.npz",
        plots_enabled=False,
    )
    selected_names = set(pd.read_csv(output / "quality_summary.csv")["motion_name"])
    simulation = pd.read_csv(SIMULATION_RESULTS)
    target = simulation[simulation["motion_name"].isin(selected_names)].copy()
    target_path = tmp_path / "simulation_targets.csv"
    target.to_csv(target_path, index=False)

    result = finalize_quality_v2_validation(
        CONFIG,
        output,
        target_path,
        bootstrap_resamples=200,
        bootstrap_seed=7,
    )
    assert result.motion_count == 5
    decision = json.loads(result.decision_json.read_text(encoding="utf-8"))
    assert decision["score_is_calibrated"] is False
    assert decision["human_object_external_validation"] is False
    assert decision["skill_learning_validation"] is False
    correlations = pd.read_csv(result.correlation_csv)
    assert len(correlations) == 27
    assert set(correlations["n_motions"]) == {5}
    manifest = json.loads(result.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["validation"]["score_is_calibrated"] is False
    assert manifest["validation"]["simulation_results_sha256"] == sha256_file(target_path)
    for entry in manifest["outputs"]:
        path = output / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert sha256_file(path) == entry["sha256"]
