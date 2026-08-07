from __future__ import annotations

import ast
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import pytest

from gmr_quality.beyondmimic import (
    BMIMIC_BODY_NAMES,
    ISAAC_TO_GMR_DOF,
    convert_gmr_motion,
    merge_bmimic_motions,
    validate_bmimic_payload,
)
from gmr_quality.config import load_config
from gmr_quality.step11_protocol import (
    audit_step11_environment,
    build_step11_cohorts,
    load_step11_config,
)
from gmr_quality.step11_analysis import finalize_step11_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP11_CONFIG = PROJECT_ROOT / "configs" / "step11_beyondmimic.yaml"


def test_frozen_cohorts_are_stratified_and_ranked() -> None:
    config = load_step11_config(STEP11_CONFIG)
    cohorts = build_step11_cohorts(config)
    assert cohorts.groupby("cohort").size().to_dict() == {
        "bottom_k": 5,
        "full": 15,
        "random_k": 5,
        "top_k": 5,
    }
    assert set(cohorts.loc[cohorts.cohort == "top_k", "stratum_rank"]) == {1}
    assert set(cohorts.loc[cohorts.cohort == "bottom_k", "stratum_rank"]) == {3}
    for cohort in ("top_k", "random_k", "bottom_k"):
        assert set(cohorts.loc[cohorts.cohort == cohort, "motion_category"]) == set(
            config["cohorts"]["strata"]
        )


def test_converter_schema_order_fk_and_boundaries(tmp_path: Path) -> None:
    step11 = load_step11_config(STEP11_CONFIG)
    cohorts = build_step11_cohorts(step11)
    source = Path(cohorts.iloc[0]["pkl_path"])
    project_config = load_config(step11["inputs"]["project_config"])
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    first = convert_gmr_motion(
        project_config, "first", source, first_path, selected_frames=12, output_fps=50.0
    )
    convert_gmr_motion(
        project_config, "second", source, second_path, selected_frames=12, output_fps=50.0
    )
    assert first.output_frames == 19
    with np.load(first_path, allow_pickle=False) as data:
        validate_bmimic_payload(data)
        assert tuple(data["body_names"].tolist()) == BMIMIC_BODY_NAMES
        assert data["joint_pos"].shape == (19, 29)
        # Rebuild one GMR qpos from the Isaac/BeyondMimic DoF order and verify
        # that the stored body targets are true FK, not reordered guesses.
        model = mujoco.MjModel.from_xml_path(project_config["paths"]["robot_xml"])
        state = mujoco.MjData(model)
        qpos = np.zeros(model.nq)
        qpos[:3] = data["body_pos_w"][0, 0]
        qpos[3:7] = data["body_quat_w"][0, 0]
        qpos[7 + ISAAC_TO_GMR_DOF] = data["joint_pos"][0]
        state.qpos[:] = qpos
        mujoco.mj_forward(model, state)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BMIMIC_BODY_NAMES]
        np.testing.assert_allclose(state.xpos[ids], data["body_pos_w"][0], atol=2e-6)

    bundle = merge_bmimic_motions([first_path, second_path], tmp_path / "bundle.npz")
    with np.load(bundle, allow_pickle=False) as data:
        validate_bmimic_payload(data)
        assert data["motion_names"].tolist() == ["first", "second"]
        assert data["motion_starts"].tolist() == [0, 19]
        assert data["motion_ends"].tolist() == [19, 38]


def test_environment_audit_reports_real_gpu_blocker() -> None:
    audit = audit_step11_environment(load_step11_config(STEP11_CONFIG))
    assert audit["checks"]["framework_commit_matches"] is True
    assert audit["checks"]["isaaclab_commit_matches"] is True
    if not audit["checks"]["cuda_available"]:
        assert "GPU_DEVICE_UNAVAILABLE" in audit["blocker_codes"]
        assert audit["ready_for_training"] is False


def test_local_integration_scripts_are_parseable_and_do_not_edit_upstream() -> None:
    integration = PROJECT_ROOT / "integrations" / "beyondmimic"
    for name in (
        "train_local.py",
        "evaluate_local.py",
        "boundary_aware_command.py",
        "run_formal.py",
        "gpu_audit.py",
    ):
        source = (integration / name).read_text(encoding="utf-8")
        ast.parse(source)
    repository_status = (
        Path("/root/Downloads/nmr_pipeline/repos/whole_body_tracking") / ".git"
    )
    assert repository_status.exists()


def test_finalizer_requires_all_policy_results_or_finalizes_complete_set() -> None:
    output = PROJECT_ROOT / "outputs" / "step11_learning_experiment"
    if not (output / "training_jobs.csv").is_file():
        return
    jobs = pd.read_csv(output / "training_jobs.csv")
    missing = sum(not Path(path).is_file() for path in jobs["result_json"])
    if missing:
        with pytest.raises(RuntimeError, match=rf"STEP11_RESULTS_INCOMPLETE: {missing}/12"):
            finalize_step11_experiment(STEP11_CONFIG, output)
    else:
        manifest = finalize_step11_experiment(STEP11_CONFIG, output)
        assert manifest["complete"] is True
        assert manifest["policy_result_count"] == 12


def test_formal_preparation_manifest_if_present() -> None:
    output = PROJECT_ROOT / "outputs" / "step11_learning_experiment"
    manifest_path = output / "step11_preparation_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["preparation_complete"] is True
    assert manifest["training_complete"] is False
    assert manifest["complete"] is False
    assert manifest["training_job_count"] == 12
    assert len(manifest["conversion"]) == 15
    assert {value["frames"] for value in manifest["cohort_bundles"].values()} == {
        32_770,
        98_310,
    }
    # The preparation manifest is the pre-training snapshot. The live protocol
    # config and jobs table are intentionally revised/updated before and during
    # formal execution; the final manifest owns their terminal hashes.
    mutable_names = {
        "step11_beyondmimic.yaml",
        "resolved_step11_config.yaml",
        "training_jobs.csv",
    }
    for entry in manifest["files"]:
        path = Path(entry["path"])
        if path.name in mutable_names:
            continue
        assert path.stat().st_size == entry["size_bytes"]
        from gmr_quality.manifest import sha256_file

        assert sha256_file(path) == entry["sha256"]
    jobs = pd.read_csv(output / "training_jobs.csv")
    assert len(jobs) == 12
    assert set(jobs["status"]) <= {
        "blocked_gpu", "ready", "training", "trained", "evaluating", "complete"
    }


def test_final_manifest_hashes_if_present() -> None:
    output = PROJECT_ROOT / "outputs" / "step11_learning_experiment"
    manifest_path = output / "step11_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["training_job_count"] == manifest["policy_result_count"] == 12
    assert manifest["all_training_jobs_complete"] is True
    from gmr_quality.manifest import sha256_file

    for entry in manifest["files"]:
        path = output / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert sha256_file(path) == entry["sha256"]
