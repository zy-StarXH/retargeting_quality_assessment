from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationCode(str, Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PICKLE_READ_ERROR = "PICKLE_READ_ERROR"
    PAYLOAD_NOT_MAPPING = "PAYLOAD_NOT_MAPPING"
    MISSING_FIELD = "MISSING_FIELD"
    FPS_INVALID = "FPS_INVALID"
    ARRAY_NOT_NUMERIC = "ARRAY_NOT_NUMERIC"
    ROOT_POS_SHAPE_INVALID = "ROOT_POS_SHAPE_INVALID"
    ROOT_QUAT_SHAPE_INVALID = "ROOT_QUAT_SHAPE_INVALID"
    JOINT_POS_SHAPE_INVALID = "JOINT_POS_SHAPE_INVALID"
    DOF_COUNT_MISMATCH = "DOF_COUNT_MISMATCH"
    FRAME_COUNT_MISMATCH = "FRAME_COUNT_MISMATCH"
    TOO_FEW_FRAMES = "TOO_FEW_FRAMES"
    DATA_NAN = "DATA_NAN"
    DATA_INF = "DATA_INF"
    QUATERNION_NORM_INVALID = "QUATERNION_NORM_INVALID"
    JOINT_NAMES_INVALID = "JOINT_NAMES_INVALID"
    JOINT_ORDER_IMPLICIT = "JOINT_ORDER_IMPLICIT"


STRUCTURAL_CODES = frozenset(
    {
        ValidationCode.FILE_NOT_FOUND,
        ValidationCode.PICKLE_READ_ERROR,
        ValidationCode.PAYLOAD_NOT_MAPPING,
        ValidationCode.MISSING_FIELD,
        ValidationCode.ARRAY_NOT_NUMERIC,
        ValidationCode.ROOT_POS_SHAPE_INVALID,
        ValidationCode.ROOT_QUAT_SHAPE_INVALID,
        ValidationCode.JOINT_POS_SHAPE_INVALID,
        ValidationCode.DOF_COUNT_MISMATCH,
        ValidationCode.FRAME_COUNT_MISMATCH,
        ValidationCode.TOO_FEW_FRAMES,
        ValidationCode.JOINT_NAMES_INVALID,
    }
)


@dataclass(frozen=True)
class MotionUnits:
    root_position: str = "m"
    joint_position: str = "rad"
    time: str = "s"


@dataclass(frozen=True)
class MotionValidationConfig:
    input_format: str = "gmr_pickle_v1"
    trusted_pickle_only: bool = True
    expected_dof_count: int = 29
    minimum_frames: int = 4
    quaternion_norm_tolerance: float = 1e-3
    root_quaternion_order: str = "xyzw"
    units: MotionUnits = field(default_factory=MotionUnits)

    def __post_init__(self) -> None:
        if self.input_format != "gmr_pickle_v1":
            raise ValueError("input_format must be 'gmr_pickle_v1'")
        if self.trusted_pickle_only is not True:
            raise ValueError("GMR pickle loading must remain restricted to trusted input")
        if self.expected_dof_count <= 0:
            raise ValueError("expected_dof_count must be positive")
        if self.minimum_frames < 1:
            raise ValueError("minimum_frames must be at least 1")
        if not np.isfinite(self.quaternion_norm_tolerance) or self.quaternion_norm_tolerance <= 0:
            raise ValueError("quaternion_norm_tolerance must be finite and positive")
        if self.root_quaternion_order != "xyzw":
            raise ValueError("GMR MotionRecord requires root_quaternion_order='xyzw'")
        if self.units != MotionUnits():
            raise ValueError("GMR input units must be meters, radians, and seconds")

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "MotionValidationConfig":
        input_config = config["input"]
        units = input_config["units"]
        return cls(
            input_format=str(input_config["format"]),
            trusted_pickle_only=bool(input_config["trusted_pickle_only"]),
            expected_dof_count=int(input_config["expected_dof_count"]),
            minimum_frames=int(input_config["minimum_frames"]),
            quaternion_norm_tolerance=float(input_config["quaternion_norm_tolerance"]),
            root_quaternion_order=str(input_config["root_quaternion_order"]),
            units=MotionUnits(
                root_position=str(units["root_position"]),
                joint_position=str(units["joint_position"]),
                time=str(units["time"]),
            ),
        )


@dataclass(frozen=True)
class MotionRecord:
    """Validated GMR trajectory using explicit units and xyzw root quaternions."""

    fps: float
    root_pos: NDArray[np.number]
    root_quat_xyzw: NDArray[np.number]
    joint_pos: NDArray[np.number]
    source_path: Path
    format_name: str
    units: MotionUnits
    joint_names: tuple[str, ...] | None
    joint_order_source: str

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps


@dataclass(frozen=True)
class ValidationIssue:
    code: ValidationCode
    severity: IssueSeverity
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    source_path: Path
    format_name: str
    root_quaternion_order: str
    units: MotionUnits
    shape_valid: bool
    data_valid: bool
    hard_fail: bool
    fps: float | None
    num_frames: int | None
    expected_dof_count: int
    nan_count: int
    inf_count: int
    quat_norm_mean_error: float | None
    quat_norm_max_error: float | None
    array_shapes: tuple[tuple[str, tuple[int, ...] | None], ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(
            issue.code.value
            for issue in self.issues
            if issue.severity is IssueSeverity.ERROR
        )

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(
            issue.code.value
            for issue in self.issues
            if issue.severity is IssueSeverity.WARNING
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "format_name": self.format_name,
            "root_quaternion_order": self.root_quaternion_order,
            "units": {
                "root_position": self.units.root_position,
                "joint_position": self.units.joint_position,
                "time": self.units.time,
            },
            "shape_valid": self.shape_valid,
            "data_valid": self.data_valid,
            "hard_fail": self.hard_fail,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "expected_dof_count": self.expected_dof_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "quat_norm_mean_error": self.quat_norm_mean_error,
            "quat_norm_max_error": self.quat_norm_max_error,
            "array_shapes": {
                name: list(shape) if shape is not None else None
                for name, shape in self.array_shapes
            },
            "failure_codes": list(self.failure_codes),
            "warning_codes": list(self.warning_codes),
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class MotionLoadResult:
    record: MotionRecord | None
    validation: ValidationReport

    def require_record(self) -> MotionRecord:
        if self.record is None:
            codes = ", ".join(self.validation.failure_codes) or "unknown validation error"
            raise InvalidMotionError(
                f"Motion is not valid ({codes}): {self.validation.source_path}"
            )
        return self.record


class InvalidMotionError(ValueError):
    """Raised when downstream code explicitly requires a valid MotionRecord."""
