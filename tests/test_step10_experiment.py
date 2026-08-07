from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gmr_quality.config import load_config
from gmr_quality.manifest import sha256_file
from gmr_quality.step10_dataset import STEP10_STRATA, load_step10_selection
from gmr_quality.step10_experiment import _sensitivity_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP10_OUTPUT = PROJECT_ROOT / "outputs" / "step10_offline_experiment"


def test_frozen_selection_has_three_existing_sources_per_stratum() -> None:
    selections = load_step10_selection(
        PROJECT_ROOT / "data" / "step10_source_selection.csv"
    )
    assert len(selections) == 15
    assert {item.motion_category for item in selections} == set(STEP10_STRATA)
    for stratum in STEP10_STRATA:
        group = [item for item in selections if item.motion_category == stratum]
        assert len(group) == 3
        assert all(item.source_bvh.is_file() for item in group)


def test_v1_configuration_is_frozen_but_not_calibrated() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "unitree_g1_v1.yaml")
    assert config["scoring"]["version"] == "v1"
    assert config["scoring"]["calibrated"] is False
    assert config["freeze"]["status"] == "frozen_step10_offline"
    assert config["freeze"]["score_calibration"].startswith("deferred_to_step11")
    assert len(_sensitivity_scenarios(config)) == 21


def test_formal_step10_automated_outputs_are_complete() -> None:
    automated = json.loads(
        (STEP10_OUTPUT / "automated_analysis.json").read_text(encoding="utf-8")
    )
    assert automated["ok"] is True
    assert all(automated["automated_checks"].values())
    assert automated["cohort"]["motion_count"] == 15
    assert automated["cohort"]["total_frames"] == 90_540
    assert set(automated["cohort"]["stratum_counts"]) == set(STEP10_STRATA)
    assert set(automated["cohort"]["stratum_counts"].values()) == {3}

    motions = pd.read_csv(STEP10_OUTPUT / "analysis" / "motion_results.csv")
    strata = pd.read_csv(STEP10_OUTPUT / "analysis" / "stratum_summary.csv")
    correlations = pd.read_csv(
        STEP10_OUTPUT / "analysis" / "dimension_correlations_spearman.csv"
    )
    assert len(motions) == 15
    assert motions["num_frames"].sum() == 90_540
    assert set(motions["score_version"]) == {"v1"}
    assert not motions["score_is_calibrated"].any()
    assert len(strata) == 5
    assert len(correlations) == 60
    assert not correlations["inferential_use_allowed"].any()


def test_review_packet_and_final_manifest_are_complete_and_hashed() -> None:
    candidates = pd.read_csv(
        STEP10_OUTPUT / "analysis" / "manual_review_candidates.csv"
    )
    review = pd.read_csv(STEP10_OUTPUT / "analysis" / "manual_review.csv")
    assert len(candidates) == len(review) == 10
    assert set(candidates["motion_name"]) == set(review["motion_name"])
    assert set(candidates.groupby("motion_category").size()) == {2}
    for column in (
        "pose_montage",
        "anomaly_timeline",
        "joint_diagnostic",
        "contact_diagnostic",
        "fidelity_diagnostic",
    ):
        assert all(Path(path).is_file() for path in candidates[column])

    manifest = json.loads(
        (STEP10_OUTPUT / "step10_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["manual_review"]["reviewed_motion_count"] == 10
    assert set(manifest["manual_review"]["stratum_coverage"].values()) == {2}
    for entry in manifest["files"]:
        path = STEP10_OUTPUT / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert sha256_file(path) == entry["sha256"]
