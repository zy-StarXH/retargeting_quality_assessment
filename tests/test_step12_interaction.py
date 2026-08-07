from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from gmr_quality.interaction import compute_interaction_metrics, load_interaction_record
from gmr_quality.manifest import sha256_file
from gmr_quality.step12_experiment import (
    _hand_targets,
    _write_demo,
    build_interaction_scene,
    load_step12_config,
    replay_object_physics,
    solve_hand_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "step12_interaction.yaml"


def test_step12_protocol_is_explicitly_controlled_synthetic() -> None:
    config = load_step12_config(CONFIG_PATH)
    assert config["provenance"]["source_kind"] == "controlled_synthetic_physics_demonstrations"
    assert config["provenance"]["human_object_dataset"] is None
    assert config["task"]["interaction_mode"] == "bimanual_grasp"
    assert sum(config["offline_metrics"]["weights"].values()) == 1.0
    assert len(config["demonstrations"]["levels"]) * len(config["demonstrations"]["seeds"]) == 15


def test_clean_g1_interaction_schema_metrics_and_physics(tmp_path: Path) -> None:
    config = load_step12_config(CONFIG_PATH)
    scene = build_interaction_scene(config, tmp_path / "scene.xml")
    model = mujoco.MjModel.from_xml_path(config["robot"]["xml"])
    targets, expected, phase_ids, phase_names = _hand_targets(config, model, 0.0, 101)
    robot_qpos, left_pos, right_pos, left_quat, right_quat = solve_hand_trajectory(
        config, model, targets
    )
    physics = replay_object_physics(
        scene, config, robot_qpos, expected_contact=expected, record_trajectory=True
    )
    assert physics["success"] is True
    assert physics["max_lift_m"] >= config["task"]["success_min_lift_m"]
    assert physics["grasp_latch_rate"] > 0.4
    demo = tmp_path / "clean.npz"
    _write_demo(
        demo,
        config,
        "clean",
        robot_qpos,
        left_pos,
        right_pos,
        left_quat,
        right_quat,
        expected,
        phase_ids,
        phase_names,
        physics,
    )
    record = load_interaction_record(demo)
    metrics = compute_interaction_metrics(record, config)
    assert record.task_label == "bimanual_lift_translate_place"
    assert record.object_pos_m.shape == (300, 3)
    assert record.object_quat_xyzw.shape == (300, 4)
    assert set(np.unique(record.phase_id)) == set(range(7))
    assert metrics["nominal_task_success"] is True
    assert 0.0 <= metrics["interaction_quality_score"] <= 100.0
    assert metrics["score_is_calibrated"] is False


def test_step12_final_manifest_if_present() -> None:
    output = PROJECT_ROOT / "outputs" / "step12_interaction_experiment"
    path = output / "step12_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["benchmark_scope"] == "controlled_synthetic_g1_interaction"
    assert manifest["human_object_external_validation"] is False
    assert manifest["demonstration_count"] == 15
    assert manifest["robustness_episode_count"] == 1500
    assert manifest["score_is_calibrated"] is False
    for entry in manifest["files"]:
        source = Path(entry["path"])
        assert source.stat().st_size == entry["size_bytes"]
        assert sha256_file(source) == entry["sha256"]
    catalog = pd.read_csv(output / "interaction_catalog.csv")
    episodes = pd.read_csv(output / "robustness_episodes.csv")
    correlations = pd.read_csv(output / "offline_simulation_correlations.csv")
    assert len(catalog) == 15
    assert len(episodes) == 1500
    assert set(episodes.groupby("motion_name").size()) == {100}
    task_effectiveness = correlations[
        (correlations.offline_metric == "task_effectiveness_score")
        & (correlations.simulation_target == "robust_task_success_rate")
    ].iloc[0]
    assert task_effectiveness.spearman_rho > 0
    assert task_effectiveness.bootstrap_ci95_low > 0
