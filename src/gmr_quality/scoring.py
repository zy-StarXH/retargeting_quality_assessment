from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contact_metrics import ContactMetrics
from .events import QualityEvent
from .kinematic_metrics import KinematicMetrics
from .model_info import MotionModelBinding
from .types import IssueSeverity, ValidationReport


SCORE_DIMENSIONS = (
    "smoothness",
    "joint_feasibility",
    "contact",
    "fidelity",
)


@dataclass(frozen=True)
class ScoringConfig:
    version: str
    calibrated: bool
    weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("score version must be non-empty")
        weight_map = dict(self.weights)
        if tuple(weight_map) != SCORE_DIMENSIONS:
            raise ValueError(
                f"score weights must be ordered as {SCORE_DIMENSIONS}"
            )
        values = np.asarray(tuple(weight_map.values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("score weights must be finite, non-negative, and non-zero")

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "ScoringConfig":
        scoring = config["scoring"]
        calibrated = scoring["calibrated"]
        if not isinstance(calibrated, bool):
            raise ValueError("scoring.calibrated must be a boolean")
        raw_weights = scoring["weights"]
        return cls(
            version=str(scoring["version"]),
            calibrated=calibrated,
            weights=tuple(
                (dimension, float(raw_weights[dimension]))
                for dimension in SCORE_DIMENSIONS
            ),
        )


@dataclass(frozen=True)
class HardFailureConfig:
    sustained_collision_frames: int

    def __post_init__(self) -> None:
        if self.sustained_collision_frames < 1:
            raise ValueError("sustained_collision_frames must be at least 1")

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "HardFailureConfig":
        return cls(
            sustained_collision_frames=int(
                config["thresholds"]["sustained_collision_frames"]
            )
        )


@dataclass(frozen=True)
class QualityScores:
    smooth_valid_frame: NDArray[np.bool_]
    joint_valid_frame: NDArray[np.bool_]
    contact_valid_frame: NDArray[np.bool_]
    fidelity_valid_frame: NDArray[np.bool_] | None
    smoothness_score: float
    joint_feasibility_score: float
    contact_score: float
    fidelity_score: float
    quality_score_v0: float
    score_version: str
    score_available_dimensions: tuple[str, ...]
    score_is_calibrated: bool
    normalized_weights: tuple[tuple[str, float], ...]

    @property
    def quality_score(self) -> float:
        """Version-neutral score; quality_score_v0 remains a compatibility alias."""
        return self.quality_score_v0


@dataclass(frozen=True)
class QualityReason:
    code: str
    message: str
    start_frame: int | None = None
    end_frame: int | None = None
    entities: tuple[str, ...] = ()

    @property
    def token(self) -> str:
        token = self.code
        if self.entities:
            token += ":" + "|".join(self.entities)
        if self.start_frame is not None:
            if self.end_frame is None or self.end_frame == self.start_frame:
                token += f"@frame_{self.start_frame}"
            else:
                token += f"@frames_{self.start_frame}-{self.end_frame}"
        return token

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "token": self.token,
            "message": self.message,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "entities": list(self.entities),
        }


@dataclass(frozen=True)
class QualityDecision:
    hard_fail: bool
    failure_reasons: tuple[QualityReason, ...]
    warning_reasons: tuple[QualityReason, ...]


def _readonly_bool(value: np.ndarray) -> NDArray[np.bool_]:
    result = np.asarray(value, dtype=np.bool_).copy()
    result.setflags(write=False)
    return result


def compute_quality_scores(
    kinematic: KinematicMetrics,
    contact: ContactMetrics,
    scoring_config: ScoringConfig,
    *,
    fidelity_valid_frame: np.ndarray | None = None,
) -> QualityScores:
    """Compute the explicit v0 frame-compliance scores and renormalized total."""
    smooth = kinematic.smoothness
    feasibility = kinematic.joint_feasibility
    frame_count = smooth.joint_jump_frame.shape[0]
    if any(
        array.shape != (frame_count,)
        for array in (
            feasibility.velocity_violation_frame,
            feasibility.position_violation_frame,
            feasibility.near_limit_2pct_frame,
            contact.self_collision_nonhand_frame,
            contact.ground_penetration_frame,
            contact.foot_slide_frame,
            contact.undesired_ground_contact_frame,
            contact.airborne_frame,
        )
    ):
        raise ValueError("Score input frame arrays are not aligned")

    smooth_valid = ~(
        smooth.joint_jump_frame | feasibility.velocity_violation_frame
    )
    joint_valid = ~(
        feasibility.position_violation_frame
        | feasibility.near_limit_2pct_frame
    )
    contact_invalid = (
        contact.self_collision_nonhand_frame
        | contact.ground_penetration_frame
        | contact.foot_slide_frame
    )
    if contact.summary.undesired_ground_contact_is_failure:
        contact_invalid = contact_invalid | contact.undesired_ground_contact_frame
    if contact.summary.airborne_is_failure:
        contact_invalid = contact_invalid | contact.airborne_frame
    contact_valid = ~contact_invalid

    fidelity_valid: NDArray[np.bool_] | None = None
    if fidelity_valid_frame is not None:
        raw_fidelity = np.asarray(fidelity_valid_frame)
        if raw_fidelity.shape != (frame_count,) or raw_fidelity.dtype.kind != "b":
            raise ValueError(
                f"fidelity_valid_frame must be boolean shape ({frame_count},)"
            )
        fidelity_valid = _readonly_bool(raw_fidelity)

    smoothness_score = 100.0 * float(np.mean(smooth_valid))
    joint_score = 100.0 * float(np.mean(joint_valid))
    contact_score = 100.0 * float(np.mean(contact_valid))
    fidelity_score = (
        100.0 * float(np.mean(fidelity_valid))
        if fidelity_valid is not None
        else float("nan")
    )
    values = {
        "smoothness": smoothness_score,
        "joint_feasibility": joint_score,
        "contact": contact_score,
        "fidelity": fidelity_score,
    }
    available = tuple(
        dimension
        for dimension in SCORE_DIMENSIONS
        if np.isfinite(values[dimension])
    )
    weight_map = scoring_config.weight_map
    available_weight = sum(weight_map[dimension] for dimension in available)
    if available_weight <= 0:
        raise ValueError("Available score dimensions have zero configured weight")
    normalized_weights = tuple(
        (dimension, weight_map[dimension] / available_weight)
        for dimension in available
    )
    quality_score = sum(
        values[dimension] * weight for dimension, weight in normalized_weights
    )
    return QualityScores(
        smooth_valid_frame=_readonly_bool(smooth_valid),
        joint_valid_frame=_readonly_bool(joint_valid),
        contact_valid_frame=_readonly_bool(contact_valid),
        fidelity_valid_frame=fidelity_valid,
        smoothness_score=smoothness_score,
        joint_feasibility_score=joint_score,
        contact_score=contact_score,
        fidelity_score=fidelity_score,
        quality_score_v0=float(quality_score),
        score_version=scoring_config.version,
        score_available_dimensions=available,
        score_is_calibrated=scoring_config.calibrated,
        normalized_weights=normalized_weights,
    )


def _deduplicate(reasons: list[QualityReason]) -> tuple[QualityReason, ...]:
    result: list[QualityReason] = []
    seen: set[tuple[object, ...]] = set()
    for reason in reasons:
        key = (
            reason.code,
            reason.start_frame,
            reason.end_frame,
            reason.entities,
        )
        if key not in seen:
            seen.add(key)
            result.append(reason)
    return tuple(result)


def compute_quality_decision(
    validation: ValidationReport,
    hard_failure_config: HardFailureConfig,
    *,
    binding: MotionModelBinding | None = None,
    kinematic: KinematicMetrics | None = None,
    contact: ContactMetrics | None = None,
    events: tuple[QualityEvent, ...] = (),
    scores: QualityScores | None = None,
) -> QualityDecision:
    """Apply stable hard-failure codes separately from the continuous score."""
    failures: list[QualityReason] = []
    warnings: list[QualityReason] = []
    for issue in validation.issues:
        reason = QualityReason(
            code=issue.code.value,
            message=f"{issue.field}: {issue.message}",
        )
        if issue.severity is IssueSeverity.ERROR:
            failures.append(reason)
        else:
            warnings.append(reason)

    if validation.data_valid:
        if binding is None or kinematic is None or contact is None or scores is None:
            raise ValueError("Valid data requires binding, metrics, and scores")
        for code in binding.warnings:
            warnings.append(
                QualityReason(code=code, message=code.replace("_", " ").lower())
            )
        for code in contact.warnings:
            warnings.append(
                QualityReason(code=code, message=code.replace("_", " ").lower())
            )

        feasibility = kinematic.joint_feasibility.summary
        if feasibility.joint_limit_violation_frame_count:
            entity = (
                (feasibility.worst_limit_joint,)
                if feasibility.worst_limit_joint
                else ()
            )
            failures.append(
                QualityReason(
                    code="JOINT_LIMIT_EXCEEDED",
                    message=(
                        "At least one joint exceeds its XML limit; "
                        f"maximum excess={feasibility.max_limit_excess_rad} rad"
                    ),
                    start_frame=feasibility.worst_limit_frame,
                    end_frame=feasibility.worst_limit_frame,
                    entities=entity,
                )
            )

        contact_summary = contact.summary
        if (
            contact_summary.max_ground_penetration_m
            > contact_summary.severe_penetration_threshold_m
        ):
            entity = (
                (contact_summary.worst_penetration_body,)
                if contact_summary.worst_penetration_body
                else ()
            )
            failures.append(
                QualityReason(
                    code="SEVERE_PENETRATION",
                    message=(
                        "Maximum ground penetration exceeds the configured severe "
                        f"threshold: {contact_summary.max_ground_penetration_m} m"
                    ),
                    start_frame=contact_summary.worst_penetration_frame,
                    end_frame=contact_summary.worst_penetration_frame,
                    entities=entity,
                )
            )

        for event in events:
            if (
                event.event_type == "SELF_COLLISION_NONHAND"
                and event.frame_count
                >= hard_failure_config.sustained_collision_frames
            ):
                failures.append(
                    QualityReason(
                        code="SUSTAINED_SELF_COLLISION",
                        message=(
                            "Non-hand self collision lasts "
                            f"{event.frame_count} frames"
                        ),
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                        entities=event.entities,
                    )
                )
            elif (
                event.event_type == "UNDESIRED_GROUND_CONTACT"
                and contact_summary.undesired_ground_contact_is_failure
            ):
                failures.append(
                    QualityReason(
                        code="UNDESIRED_GROUND_CONTACT",
                        message="The selected contact policy forbids non-foot ground contact",
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                        entities=event.entities,
                    )
                )
            elif (
                event.event_type == "AIRBORNE_PROXY"
                and contact_summary.airborne_is_failure
            ):
                failures.append(
                    QualityReason(
                        code="AIRBORNE_NOT_ALLOWED",
                        message="The selected contact policy forbids airborne proxy frames",
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                    )
                )

        if not np.isfinite(scores.fidelity_score):
            warnings.append(
                QualityReason(
                    code="FIDELITY_UNAVAILABLE",
                    message="Fidelity is unavailable and its weight was not treated as full score",
                )
            )
        if not scores.score_is_calibrated:
            warnings.append(
                QualityReason(
                    code="SCORE_UNCALIBRATED",
                    message="quality_score is an uncalibrated offline ranking score",
                )
            )

    return QualityDecision(
        hard_fail=bool(failures),
        failure_reasons=_deduplicate(failures),
        warning_reasons=_deduplicate(warnings),
    )
