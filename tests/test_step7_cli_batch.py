from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from gmr_quality.cli import main
from gmr_quality.config import load_config
from gmr_quality.manifest import sha256_file
from gmr_quality.reporting import EVENT_COLUMNS, FRAME_METRIC_COLUMNS
from gmr_quality.runner import discover_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "unitree_g1.yaml"
CATALOG_PATH = PROJECT_ROOT / "data" / "motion_catalog.csv"
CONFIG = load_config(CONFIG_PATH)
AMASS_SAMPLE = (
    Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "amass" / "g1_motion.pkl"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_discovery_uses_catalog_and_conservative_metadata_fallback(
    tmp_path: Path,
) -> None:
    known = discover_inputs(
        input_path=AMASS_SAMPLE,
        input_dir=None,
        pattern="*.pkl",
        catalog_path=CATALOG_PATH,
    )
    assert len(known) == 1
    assert known[0].motion_name == "amass_g1_sample"
    assert known[0].motion_category == "unknown"
    assert known[0].contact_policy == "descriptive_only"
    assert known[0].source_bvh is None

    nested = tmp_path / "inputs" / "nested"
    nested.mkdir(parents=True)
    unregistered = nested / "new_motion.pkl"
    unregistered.write_bytes(b"discovery does not deserialize inputs")
    fallback = discover_inputs(
        input_path=None,
        input_dir=tmp_path / "inputs",
        pattern="**/*.pkl",
        catalog_path=CATALOG_PATH,
    )
    assert fallback[0].motion_name == "nested__new_motion"
    assert fallback[0].motion_category == "unknown"
    assert fallback[0].contact_policy == "descriptive_only"
    assert fallback[0].notes == ""


def test_single_cli_writes_complete_hashed_and_reproducible_products(
    tmp_path: Path,
) -> None:
    source_hash_before = sha256_file(AMASS_SAMPLE)
    first = tmp_path / "run_first"
    second = tmp_path / "run_second"
    base_args = [
        "evaluate",
        "--input",
        str(AMASS_SAMPLE),
        "--config",
        str(CONFIG_PATH),
        "--catalog",
        str(CATALOG_PATH),
        "--json",
    ]
    assert main([*base_args, "--output-dir", str(first)]) == 0
    assert main([*base_args, "--output-dir", str(second)]) == 0
    assert sha256_file(AMASS_SAMPLE) == source_hash_before

    expected_relative = (
        "quality_summary.csv",
        "quality_events.csv",
        "frame_metrics/amass_g1_sample.csv",
        "plots/amass_g1_sample_joint_velocity.png",
        "plots/amass_g1_sample_foot_contact.png",
        "resolved_config.yaml",
    )
    for relative in expected_relative:
        first_path = first / relative
        second_path = second / relative
        assert first_path.is_file()
        assert _sha256(first_path) == _sha256(second_path), relative

    summary = pd.read_csv(first / "quality_summary.csv")
    frames = pd.read_csv(first / "frame_metrics" / "amass_g1_sample.csv")
    assert len(summary) == 1
    assert len(frames) == 89
    assert summary.loc[0, "motion_name"] == "amass_g1_sample"
    assert bool(summary.loc[0, "data_valid"]) is True
    assert bool(summary.loc[0, "score_is_calibrated"]) is False
    assert np.isnan(summary.loc[0, "fidelity_score"])
    for relative in expected_relative[3:5]:
        image = first / relative
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 1_000

    manifest = json.loads((first / "run_manifest.json").read_text())
    assert manifest["counts"] == {
        "discovered": 1,
        "succeeded": 1,
        "failed": 0,
        "skipped": 0,
        "hard_fail": 0,
    }
    assert manifest["inputs"][0]["input_sha256"] == source_hash_before
    assert manifest["inputs"][0]["status"] == "evaluated"
    assert manifest["execution"]["fidelity_enabled"] is False
    assert manifest["configuration"]["source_sha256"] == sha256_file(CONFIG_PATH)
    assert manifest["robot_model"]["xml_sha256"] == CONFIG["robot"][
        "expected_xml_sha256"
    ]
    output_records = {row["path"]: row for row in manifest["outputs"]}
    assert set(output_records) == set(expected_relative)
    for relative, record in output_records.items():
        assert record["sha256"] == sha256_file(first / relative)
        assert record["size_bytes"] == (first / relative).stat().st_size


def test_directory_batch_isolates_invalid_motion_and_keeps_batch_reports(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_inputs"
    input_dir.mkdir()
    good = input_dir / "good.pkl"
    bad = input_dir / "bad.pkl"
    shutil.copyfile(AMASS_SAMPLE, good)
    bad.write_bytes(b"this is deliberately not a pickle")
    hashes_before = {path.name: sha256_file(path) for path in (bad, good)}
    output_dir = tmp_path / "batch_output"

    exit_code = main(
        [
            "evaluate",
            "--input-dir",
            str(input_dir),
            "--pattern",
            "*.pkl",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--no-plots",
            "--json",
        ]
    )
    assert exit_code == 2
    assert {path.name: sha256_file(path) for path in (bad, good)} == hashes_before

    summary = pd.read_csv(output_dir / "quality_summary.csv")
    events = pd.read_csv(output_dir / "quality_events.csv")
    bad_frames = pd.read_csv(output_dir / "frame_metrics" / "bad.csv")
    good_frames = pd.read_csv(output_dir / "frame_metrics" / "good.csv")
    assert summary["motion_name"].tolist() == ["bad", "good"]
    assert summary["data_valid"].astype(bool).tolist() == [False, True]
    assert "PICKLE_READ_ERROR" in summary.loc[0, "failure_reason_codes"]
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert list(bad_frames.columns) == list(FRAME_METRIC_COLUMNS)
    assert bad_frames.empty
    assert len(good_frames) == 89

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["counts"] == {
        "discovered": 2,
        "succeeded": 1,
        "failed": 1,
        "skipped": 0,
        "hard_fail": 1,
    }
    assert [row["status"] for row in manifest["inputs"]] == [
        "failed_validation",
        "evaluated",
    ]
    assert manifest["inputs"][0]["failure_codes"] == ["PICKLE_READ_ERROR"]
    assert manifest["execution"]["plots_enabled"] is False


def test_output_inside_input_tree_is_rejected_without_writes_or_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    input_dir = tmp_path / "protected_inputs"
    input_dir.mkdir()
    shutil.copyfile(AMASS_SAMPLE, input_dir / "good.pkl")
    forbidden = input_dir / "results"

    exit_code = main(
        [
            "evaluate",
            "--input-dir",
            str(input_dir),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(forbidden),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "must be outside the input directory" in captured.err
    assert "Traceback" not in captured.err
    assert not forbidden.exists()


def test_nonempty_output_directory_is_rejected(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "existing_output"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    exit_code = main(
        [
            "evaluate",
            "--input",
            str(AMASS_SAMPLE),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "must be empty" in captured.err
    assert marker.read_text(encoding="utf-8") == "user data"
