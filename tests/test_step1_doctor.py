from pathlib import Path

from gmr_quality.config import load_config
from gmr_quality.doctor import run_doctor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "unitree_g1.yaml"
CATALOG_PATH = PROJECT_ROOT / "data" / "motion_catalog.csv"


def test_step1_environment_doctor_passes_without_hidden_failures() -> None:
    config = load_config(CONFIG_PATH)
    report = run_doctor(config, CATALOG_PATH)

    assert report.ok, report.as_json()
    assert report.counts["fail"] == 0

    checks = {check.name: check for check in report.checks}
    assert checks["hinge_dof_count"].status == "pass"
    assert checks["hinge_qpos_order"].status == "pass"
    assert checks["floor_geom"].status == "pass"
    assert checks["left_sole_geoms"].status == "pass"
    assert checks["right_sole_geoms"].status == "pass"


def test_verified_pair_and_remaining_provenance_gaps_are_explicit() -> None:
    config = load_config(CONFIG_PATH)
    report = run_doctor(config, CATALOG_PATH)
    checks = {check.name: check for check in report.checks}

    frame_check = checks["motion:dance1_subject1_verified:source_frames"]
    assert frame_check.status == "pass"
    assert "delta=1" in frame_check.details
    assert "source_prefix_drop_last" in frame_check.details
    assert checks["motion:dance1_subject1_verified:source_hashes"].status == "pass"

    unpaired = checks["motion:walk_unpaired:source_pair"]
    assert unpaired.status == "warn"
    assert "fidelity must remain disabled" in unpaired.details


def test_config_keeps_unavailable_fidelity_out_of_automatic_full_credit() -> None:
    config = load_config(CONFIG_PATH)

    assert config["scoring"]["calibrated"] is False
    assert config["scoring"]["weights"]["fidelity"] > 0
    assert config["contact_policies"]["descriptive_only"][
        "undesired_ground_contact_is_failure"
    ] is False
