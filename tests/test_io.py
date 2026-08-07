from __future__ import annotations

import pickle
import json
from pathlib import Path

import numpy as np
import pytest

from gmr_quality.config import load_config
from gmr_quality.io import (
    load_motion_from_project_config,
    load_motion_pickle,
    validate_motion_payload,
)
from gmr_quality.types import InvalidMotionError, MotionValidationConfig, ValidationCode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "unitree_g1.yaml")
KNOWN_SAMPLE = Path(CONFIG["paths"]["gmr_root"]) / "outputs" / "g1_dance1_subject2.pkl"


def valid_payload(num_frames: int = 4, num_dofs: int = 29) -> dict[str, object]:
    root_quat = np.zeros((num_frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    return {
        "fps": 30.0,
        "root_pos": np.zeros((num_frames, 3), dtype=np.float64),
        "root_rot": root_quat,
        "dof_pos": np.zeros((num_frames, num_dofs), dtype=np.float64),
        "local_body_pos": None,
        "link_body_list": None,
    }


def dump_payload(path: Path, payload: object) -> Path:
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def codes(result) -> set[str]:
    return set(result.validation.failure_codes)


def test_loads_real_gmr_sample_into_explicit_immutable_motion_record() -> None:
    result = load_motion_from_project_config(KNOWN_SAMPLE, CONFIG)
    record = result.require_record()

    assert result.validation.data_valid is True
    assert result.validation.shape_valid is True
    assert result.validation.hard_fail is False
    assert result.validation.nan_count == 0
    assert result.validation.inf_count == 0
    assert result.validation.quat_norm_max_error < 1e-12
    assert result.validation.warning_codes == (ValidationCode.JOINT_ORDER_IMPLICIT.value,)
    assert record.fps == 30.0
    assert record.format_name == "gmr_pickle_v1"
    assert record.root_pos.shape == (3944, 3)
    assert record.root_quat_xyzw.shape == (3944, 4)
    assert record.joint_pos.shape == (3944, 29)
    assert record.units.root_position == "m"
    assert record.units.joint_position == "rad"
    assert record.units.time == "s"
    assert record.joint_names is None
    assert record.joint_order_source == "implicit_gmr_qpos_order"
    assert record.root_pos.flags.writeable is False
    assert record.root_quat_xyzw.flags.writeable is False
    assert record.joint_pos.flags.writeable is False
    serialized = result.validation.as_dict()
    assert serialized["root_quaternion_order"] == "xyzw"
    assert serialized["units"] == {
        "root_position": "m",
        "joint_position": "rad",
        "time": "s",
    }
    json.dumps(serialized)


def test_valid_payload_with_explicit_joint_names_records_provenance() -> None:
    payload = valid_payload()
    payload["dof_names"] = [f"joint_{index}" for index in range(29)]

    result = validate_motion_payload(payload, "synthetic.pkl")
    record = result.require_record()

    assert result.validation.warning_codes == ()
    assert record.joint_names == tuple(payload["dof_names"])
    assert record.joint_order_source == "pkl:dof_names"


@pytest.mark.parametrize("bad_fps", [0, -30, np.nan, np.inf, True, [30]])
def test_invalid_fps_is_a_hard_failure(bad_fps: object) -> None:
    payload = valid_payload()
    payload["fps"] = bad_fps

    result = validate_motion_payload(payload, "bad_fps.pkl")

    assert result.record is None
    assert result.validation.hard_fail is True
    assert ValidationCode.FPS_INVALID.value in codes(result)


def test_nan_and_inf_are_counted_and_not_silently_repaired() -> None:
    payload = valid_payload()
    payload["root_pos"][1, 0] = np.nan
    payload["dof_pos"][2, 3] = np.inf

    result = validate_motion_payload(payload, "nonfinite.pkl")

    assert result.record is None
    assert result.validation.nan_count == 1
    assert result.validation.inf_count == 1
    assert ValidationCode.DATA_NAN.value in codes(result)
    assert ValidationCode.DATA_INF.value in codes(result)
    with pytest.raises(InvalidMotionError, match="DATA_NAN"):
        result.require_record()


def test_non_unit_and_zero_quaternions_fail_without_normalization() -> None:
    payload = valid_payload()
    payload["root_rot"][1] *= 1.01
    payload["root_rot"][2] = 0.0
    before = payload["root_rot"].copy()

    result = validate_motion_payload(payload, "bad_quat.pkl")

    assert result.record is None
    assert result.validation.quat_norm_max_error == pytest.approx(1.0)
    assert ValidationCode.QUATERNION_NORM_INVALID.value in codes(result)
    np.testing.assert_array_equal(payload["root_rot"], before)


def test_quaternion_sign_flip_is_valid() -> None:
    payload = valid_payload()
    payload["root_rot"][1] *= -1

    result = validate_motion_payload(payload, "quat_sign.pkl")

    assert result.validation.data_valid is True
    assert ValidationCode.QUATERNION_NORM_INVALID.value not in codes(result)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    [
        ("root_pos", np.zeros((4, 2)), ValidationCode.ROOT_POS_SHAPE_INVALID),
        ("root_rot", np.zeros((4, 3)), ValidationCode.ROOT_QUAT_SHAPE_INVALID),
        ("dof_pos", np.zeros((4, 28)), ValidationCode.DOF_COUNT_MISMATCH),
        ("dof_pos", np.zeros(29), ValidationCode.JOINT_POS_SHAPE_INVALID),
        ("root_pos", np.array([["bad", "data", "x"]] * 4), ValidationCode.ARRAY_NOT_NUMERIC),
    ],
)
def test_shape_dof_and_dtype_errors_have_stable_codes(
    field: str,
    replacement: object,
    expected_code: ValidationCode,
) -> None:
    payload = valid_payload()
    payload[field] = replacement

    result = validate_motion_payload(payload, "bad_shape.pkl")

    assert result.record is None
    assert result.validation.shape_valid is False
    assert expected_code.value in codes(result)


def test_frame_count_mismatch_and_too_short_are_rejected() -> None:
    mismatch = valid_payload()
    mismatch["dof_pos"] = np.zeros((5, 29))
    mismatch_result = validate_motion_payload(mismatch, "mismatch.pkl")
    short_result = validate_motion_payload(valid_payload(num_frames=3), "short.pkl")

    assert ValidationCode.FRAME_COUNT_MISMATCH.value in codes(mismatch_result)
    assert ValidationCode.TOO_FEW_FRAMES.value in codes(short_result)


def test_missing_fields_and_non_mapping_payload_are_rejected() -> None:
    missing = valid_payload()
    del missing["root_rot"]

    missing_result = validate_motion_payload(missing, "missing.pkl")
    mapping_result = validate_motion_payload([1, 2, 3], "list.pkl")

    assert ValidationCode.MISSING_FIELD.value in codes(missing_result)
    assert ValidationCode.PAYLOAD_NOT_MAPPING.value in codes(mapping_result)


def test_missing_and_unreadable_pickle_return_results_instead_of_crashing(tmp_path: Path) -> None:
    missing_result = load_motion_pickle(tmp_path / "missing.pkl")
    broken_path = tmp_path / "broken.pkl"
    broken_path.write_bytes(b"not a pickle")
    broken_result = load_motion_pickle(broken_path)

    assert codes(missing_result) == {ValidationCode.FILE_NOT_FOUND.value}
    assert codes(broken_result) == {ValidationCode.PICKLE_READ_ERROR.value}


def test_pickle_loader_preserves_source_dtype_and_values(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["root_pos"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = dump_payload(tmp_path / "valid.pkl", payload)

    record = load_motion_pickle(path, MotionValidationConfig()).require_record()

    assert record.root_pos.dtype == np.float32
    np.testing.assert_array_equal(record.root_pos, payload["root_pos"])


def test_project_input_conventions_are_explicit_and_consistent() -> None:
    validation = MotionValidationConfig.from_project_config(CONFIG)

    assert validation.input_format == "gmr_pickle_v1"
    assert validation.trusted_pickle_only is True
    assert validation.root_quaternion_order == "xyzw"
    assert validation.expected_dof_count == CONFIG["doctor"]["expected_dof_count"]
