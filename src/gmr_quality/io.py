from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    STRUCTURAL_CODES,
    IssueSeverity,
    MotionLoadResult,
    MotionRecord,
    MotionValidationConfig,
    ValidationCode,
    ValidationIssue,
    ValidationReport,
)


REQUIRED_FIELDS = ("fps", "root_pos", "root_rot", "dof_pos")
ARRAY_FIELDS = ("root_pos", "root_rot", "dof_pos")


def _issue(
    code: ValidationCode,
    field: str,
    message: str,
    severity: IssueSeverity = IssueSeverity.ERROR,
) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, field=field, message=message)


def _empty_report(
    source_path: Path,
    validation_config: MotionValidationConfig,
    issue: ValidationIssue,
) -> ValidationReport:
    return ValidationReport(
        source_path=source_path,
        format_name=validation_config.input_format,
        root_quaternion_order=validation_config.root_quaternion_order,
        units=validation_config.units,
        shape_valid=False,
        data_valid=False,
        hard_fail=True,
        fps=None,
        num_frames=None,
        expected_dof_count=validation_config.expected_dof_count,
        nan_count=0,
        inf_count=0,
        quat_norm_mean_error=None,
        quat_norm_max_error=None,
        array_shapes=tuple((name, None) for name in ARRAY_FIELDS),
        issues=(issue,),
    )


def _as_real_numeric_array(
    value: Any,
    field: str,
    issues: list[ValidationIssue],
) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except Exception as exc:
        issues.append(
            _issue(
                ValidationCode.ARRAY_NOT_NUMERIC,
                field,
                f"cannot convert to an array: {type(exc).__name__}: {exc}",
            )
        )
        return None
    if array.dtype.kind not in "iuf":
        issues.append(
            _issue(
                ValidationCode.ARRAY_NOT_NUMERIC,
                field,
                f"expected a real numeric dtype, got {array.dtype}",
            )
        )
        return None
    return array


def _parse_fps(value: Any, issues: list[ValidationIssue]) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        issues.append(_issue(ValidationCode.FPS_INVALID, "fps", "boolean fps is invalid"))
        return None
    try:
        array = np.asarray(value)
    except Exception as exc:
        issues.append(
            _issue(
                ValidationCode.FPS_INVALID,
                "fps",
                f"cannot convert fps to a scalar: {type(exc).__name__}: {exc}",
            )
        )
        return None
    if array.shape != () or array.dtype.kind not in "iuf":
        issues.append(
            _issue(
                ValidationCode.FPS_INVALID,
                "fps",
                f"fps must be a real numeric scalar, got shape={array.shape}, dtype={array.dtype}",
            )
        )
        return None
    fps = float(array)
    if not np.isfinite(fps) or fps <= 0:
        issues.append(
            _issue(ValidationCode.FPS_INVALID, "fps", f"fps must be finite and positive, got {fps}")
        )
        return None
    return fps


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def validate_motion_payload(
    payload: Any,
    source_path: str | Path,
    validation_config: MotionValidationConfig | None = None,
) -> MotionLoadResult:
    """Validate an already-loaded GMR payload without mutating or normalizing it."""
    config = validation_config or MotionValidationConfig()
    path = Path(source_path).expanduser().resolve()

    if not isinstance(payload, Mapping):
        report = _empty_report(
            path,
            config,
            _issue(
                ValidationCode.PAYLOAD_NOT_MAPPING,
                "payload",
                f"expected a mapping, got {type(payload).__name__}",
            ),
        )
        return MotionLoadResult(record=None, validation=report)

    issues: list[ValidationIssue] = []
    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    for field in missing_fields:
        issues.append(_issue(ValidationCode.MISSING_FIELD, field, f"required field '{field}' is missing"))

    fps = _parse_fps(payload["fps"], issues) if "fps" in payload else None
    arrays: dict[str, np.ndarray | None] = {}
    for field in ARRAY_FIELDS:
        arrays[field] = (
            _as_real_numeric_array(payload[field], field, issues)
            if field in payload
            else None
        )

    root_pos = arrays["root_pos"]
    root_quat = arrays["root_rot"]
    joint_pos = arrays["dof_pos"]

    if root_pos is not None and (root_pos.ndim != 2 or root_pos.shape[1] != 3):
        issues.append(
            _issue(
                ValidationCode.ROOT_POS_SHAPE_INVALID,
                "root_pos",
                f"expected shape (T, 3), got {root_pos.shape}",
            )
        )
    if root_quat is not None and (root_quat.ndim != 2 or root_quat.shape[1] != 4):
        issues.append(
            _issue(
                ValidationCode.ROOT_QUAT_SHAPE_INVALID,
                "root_rot",
                f"expected shape (T, 4), got {root_quat.shape}",
            )
        )
    if joint_pos is not None:
        if joint_pos.ndim != 2:
            issues.append(
                _issue(
                    ValidationCode.JOINT_POS_SHAPE_INVALID,
                    "dof_pos",
                    f"expected shape (T, {config.expected_dof_count}), got {joint_pos.shape}",
                )
            )
        elif joint_pos.shape[1] != config.expected_dof_count:
            issues.append(
                _issue(
                    ValidationCode.DOF_COUNT_MISMATCH,
                    "dof_pos",
                    f"found {joint_pos.shape[1]} DoFs, expected {config.expected_dof_count}",
                )
            )

    frame_counts = {
        field: int(array.shape[0])
        for field, array in arrays.items()
        if array is not None and array.ndim >= 1
    }
    if len(set(frame_counts.values())) > 1:
        issues.append(
            _issue(
                ValidationCode.FRAME_COUNT_MISMATCH,
                "frames",
                f"first dimensions disagree: {frame_counts}",
            )
        )
    num_frames = (
        int(root_pos.shape[0])
        if root_pos is not None and root_pos.ndim >= 1
        else None
    )
    if num_frames is not None and num_frames < config.minimum_frames:
        issues.append(
            _issue(
                ValidationCode.TOO_FEW_FRAMES,
                "frames",
                f"found {num_frames} frames, need at least {config.minimum_frames}",
            )
        )

    nan_count = 0
    inf_count = 0
    for array in arrays.values():
        if array is None:
            continue
        nan_count += int(np.isnan(array).sum())
        inf_count += int(np.isinf(array).sum())
    if nan_count:
        issues.append(_issue(ValidationCode.DATA_NAN, "arrays", f"found {nan_count} NaN value(s)"))
    if inf_count:
        issues.append(_issue(ValidationCode.DATA_INF, "arrays", f"found {inf_count} Inf value(s)"))

    quat_norm_mean_error = None
    quat_norm_max_error = None
    if root_quat is not None and root_quat.ndim == 2 and root_quat.shape[1] == 4:
        finite_rows = np.isfinite(root_quat).all(axis=1)
        if finite_rows.any():
            norms = np.linalg.norm(root_quat[finite_rows], axis=1)
            errors = np.abs(norms - 1.0)
            quat_norm_mean_error = float(np.mean(errors))
            quat_norm_max_error = float(np.max(errors))
            if np.any(errors > config.quaternion_norm_tolerance):
                worst_row = int(np.flatnonzero(finite_rows)[int(np.argmax(errors))])
                issues.append(
                    _issue(
                        ValidationCode.QUATERNION_NORM_INVALID,
                        "root_rot",
                        "quaternions are not unit length within tolerance "
                        f"{config.quaternion_norm_tolerance}; max_error={quat_norm_max_error} "
                        f"at frame {worst_row}. Values were not normalized.",
                    )
                )

    joint_names: tuple[str, ...] | None = None
    joint_order_source = "implicit_gmr_qpos_order"
    if "dof_names" in payload:
        raw_names = payload["dof_names"]
        if (
            not isinstance(raw_names, (list, tuple))
            or len(raw_names) != config.expected_dof_count
            or not all(isinstance(name, str) and name for name in raw_names)
            or len(set(raw_names)) != len(raw_names)
        ):
            issues.append(
                _issue(
                    ValidationCode.JOINT_NAMES_INVALID,
                    "dof_names",
                    f"expected {config.expected_dof_count} unique non-empty joint names",
                )
            )
        else:
            joint_names = tuple(raw_names)
            joint_order_source = "pkl:dof_names"
    else:
        issues.append(
            _issue(
                ValidationCode.JOINT_ORDER_IMPLICIT,
                "dof_pos",
                "PKL has no dof_names; order is treated as GMR qpos[7:] and must be verified against the XML in step 4",
                IssueSeverity.WARNING,
            )
        )

    error_issues = [issue for issue in issues if issue.severity is IssueSeverity.ERROR]
    shape_valid = not any(issue.code in STRUCTURAL_CODES for issue in error_issues)
    data_valid = not error_issues
    report = ValidationReport(
        source_path=path,
        format_name=config.input_format,
        root_quaternion_order=config.root_quaternion_order,
        units=config.units,
        shape_valid=shape_valid,
        data_valid=data_valid,
        hard_fail=not data_valid,
        fps=fps,
        num_frames=num_frames,
        expected_dof_count=config.expected_dof_count,
        nan_count=nan_count,
        inf_count=inf_count,
        quat_norm_mean_error=quat_norm_mean_error,
        quat_norm_max_error=quat_norm_max_error,
        array_shapes=tuple(
            (field, tuple(array.shape) if array is not None else None)
            for field, array in arrays.items()
        ),
        issues=tuple(issues),
    )

    if not data_valid:
        return MotionLoadResult(record=None, validation=report)

    assert fps is not None
    assert root_pos is not None
    assert root_quat is not None
    assert joint_pos is not None
    record = MotionRecord(
        fps=fps,
        root_pos=_readonly_copy(root_pos),
        root_quat_xyzw=_readonly_copy(root_quat),
        joint_pos=_readonly_copy(joint_pos),
        source_path=path,
        format_name=config.input_format,
        units=config.units,
        joint_names=joint_names,
        joint_order_source=joint_order_source,
    )
    return MotionLoadResult(record=record, validation=report)


def load_motion_pickle(
    path: str | Path,
    validation_config: MotionValidationConfig | None = None,
) -> MotionLoadResult:
    """Load and validate a trusted local GMR pickle without silently repairing it."""
    config = validation_config or MotionValidationConfig()
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        report = _empty_report(
            source_path,
            config,
            _issue(ValidationCode.FILE_NOT_FOUND, "source_path", str(source_path)),
        )
        return MotionLoadResult(record=None, validation=report)

    try:
        with source_path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception as exc:
        report = _empty_report(
            source_path,
            config,
            _issue(
                ValidationCode.PICKLE_READ_ERROR,
                "payload",
                f"{type(exc).__name__}: {exc}",
            ),
        )
        return MotionLoadResult(record=None, validation=report)
    return validate_motion_payload(payload, source_path, config)


def load_motion_from_project_config(
    path: str | Path,
    project_config: dict[str, Any],
) -> MotionLoadResult:
    validation_config = MotionValidationConfig.from_project_config(project_config)
    return load_motion_pickle(path, validation_config)
