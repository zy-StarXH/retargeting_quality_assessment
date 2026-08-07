from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gmr_quality.cli import main
from gmr_quality.config import load_config
from gmr_quality.manifest import sha256_file
from gmr_quality.pipeline_validation import (
    PipelineValidationResult,
    run_pipeline_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "unitree_g1.yaml"
CONFIG = load_config(CONFIG_PATH)
AMASS_SAMPLE = (
    Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "amass" / "g1_motion.pkl"
)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory) -> PipelineValidationResult:
    source_hash_before = sha256_file(AMASS_SAMPLE)
    result = run_pipeline_validation(
        CONFIG,
        AMASS_SAMPLE,
        tmp_path_factory.mktemp("step8") / "result",
        plots_enabled=True,
        execution_command=("gmr-quality", "validate-pipeline", "step8-test"),
    )
    assert sha256_file(AMASS_SAMPLE) == source_hash_before
    return result


def _flagged_frames(frames: pd.DataFrame, column: str) -> list[int]:
    return frames.loc[frames[column].astype(bool), "frame"].astype(int).tolist()


def _single_event(
    events: pd.DataFrame,
    motion_name: str,
    event_type: str,
) -> pd.Series:
    selected = events.loc[
        (events["motion_name"] == motion_name)
        & (events["event_type"] == event_type)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_controlled_faults_pass_all_end_to_end_invariants(
    pipeline_run: PipelineValidationResult,
) -> None:
    assert pipeline_run.ok is True
    assert len(pipeline_run.checks) == 7
    assert all(check["passed"] for check in pipeline_run.checks)
    assert pipeline_run.evaluation_run.succeeded_count == 6
    assert pipeline_run.evaluation_run.failed_count == 1
    assert pipeline_run.evaluation_run.hard_fail_count == 2

    report = json.loads(pipeline_run.validation_json.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["source"]["sha256"] == sha256_file(AMASS_SAMPLE)
    assert report["parameters"]["frame_count"] == 12
    assert report["parameters"]["injection_frame"] == 5
    assert report["reports"]["runner_succeeded"] == 6
    assert report["reports"]["runner_failed"] == 1
    assert report["reports"]["expected_failed_validation"] == 1


def test_frame_reports_localize_each_injected_anomaly(
    pipeline_run: PipelineValidationResult,
) -> None:
    frame_dir = pipeline_run.reports_dir / "frame_metrics"
    baseline = pd.read_csv(frame_dir / "baseline.csv")
    jump = pd.read_csv(frame_dir / "joint_jump.csv")
    near = pd.read_csv(frame_dir / "near_limit.csv")
    root = pd.read_csv(frame_dir / "root_down.csv")
    control = pd.read_csv(frame_dir / "foot_contact_control.csv")
    slide = pd.read_csv(frame_dir / "foot_slide.csv")
    invalid = pd.read_csv(frame_dir / "invalid_nan.csv")

    assert len(baseline) == len(jump) == len(near) == len(root) == 12
    assert len(control) == len(slide) == 12
    assert invalid.empty
    for column in (
        "joint_jump_flag",
        "near_limit_2pct_flag",
        "penetration_flag",
        "left_foot_slide_flag",
        "right_foot_slide_flag",
    ):
        assert not baseline[column].astype(bool).any()

    assert _flagged_frames(jump, "joint_jump_flag") == [5, 6]
    assert _flagged_frames(near, "near_limit_2pct_flag") == [5]
    assert not near["joint_jump_flag"].astype(bool).any()
    assert not near["position_limit_violation_flag"].astype(bool).any()
    assert _flagged_frames(root, "penetration_flag") == [5]
    assert _flagged_frames(root, "severe_penetration_flag") == [5]
    assert not control["left_foot_slide_flag"].astype(bool).any()
    assert not control["right_foot_slide_flag"].astype(bool).any()
    assert _flagged_frames(slide, "left_foot_slide_flag") == [5, 6]
    assert _flagged_frames(slide, "right_foot_slide_flag") == [5, 6]
    assert slide.loc[5, "left_slide_speed_m_s"] == pytest.approx(0.6)

    summary = pd.read_csv(pipeline_run.reports_dir / "quality_summary.csv")
    invalid_summary = summary.set_index("motion_name").loc["invalid_nan"]
    assert bool(invalid_summary["data_valid"]) is False
    assert "DATA_NAN" in invalid_summary["failure_reason_codes"]
    assert pd.isna(invalid_summary["quality_score_v0"])


def test_events_plots_and_hash_manifests_close_the_reporting_chain(
    pipeline_run: PipelineValidationResult,
) -> None:
    events = pd.read_csv(pipeline_run.reports_dir / "quality_events.csv")
    jump_event = _single_event(events, "joint_jump", "JOINT_JUMP")
    near_event = _single_event(events, "near_limit", "NEAR_JOINT_LIMIT_2PCT")
    root_event = _single_event(events, "root_down", "GROUND_PENETRATION")
    slide_event = _single_event(events, "foot_slide", "FOOT_SLIDE")
    assert int(jump_event["start_frame"]) == 5
    assert int(jump_event["end_frame"]) == 6
    assert int(near_event["start_frame"]) == 5
    assert int(root_event["start_frame"]) == 5
    assert int(slide_event["start_frame"]) == 5
    assert int(slide_event["end_frame"]) == 6
    assert json.loads(jump_event["entities"]) == ["left_hip_pitch_joint"]
    assert json.loads(near_event["entities"]) == ["left_knee_joint"]
    assert "left_ankle_roll_link" in json.loads(root_event["entities"])
    assert json.loads(slide_event["entities"]) == ["left_foot", "right_foot"]

    plots = sorted((pipeline_run.reports_dir / "plots").glob("*.png"))
    assert len(plots) == 12
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in plots)

    manifest = json.loads(
        pipeline_run.evaluation_run.run_manifest_json.read_text(encoding="utf-8")
    )
    assert manifest["counts"] == {
        "discovered": 7,
        "succeeded": 6,
        "failed": 1,
        "skipped": 0,
        "hard_fail": 2,
    }
    assert manifest["execution"]["input_mode"] == "pipeline_validation"
    assert [row["status"] for row in manifest["inputs"]].count(
        "failed_validation"
    ) == 1
    for record in manifest["outputs"]:
        output = pipeline_run.reports_dir / record["path"]
        assert record["sha256"] == sha256_file(output)
        assert record["size_bytes"] == output.stat().st_size

    report = json.loads(pipeline_run.validation_json.read_text(encoding="utf-8"))
    for record in report["injected_inputs"]:
        injected = Path(record["path"])
        assert record["sha256"] == sha256_file(injected)


def test_validate_pipeline_cli_reports_success_despite_expected_invalid_fixture(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "cli_result"
    exit_code = main(
        [
            "validate-pipeline",
            "--input",
            str(AMASS_SAMPLE),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output),
            "--no-plots",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["checks_passed"] == payload["checks_total"] == 7
    assert payload["scenarios"] == 7
    assert payload["runner_succeeded"] == 6
    assert payload["runner_failed_as_expected"] == 1


def test_pipeline_validation_refuses_nonempty_output_without_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        run_pipeline_validation(CONFIG, AMASS_SAMPLE, output, plots_enabled=False)
    assert marker.read_text(encoding="utf-8") == "user data"
