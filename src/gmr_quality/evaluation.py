from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contact_metrics import (
    ContactEvaluationConfig,
    ContactMetrics,
    compute_contact_metrics,
)
from .events import QualityEvent, extract_quality_events
from .fidelity import FidelityConfig, FidelityMetrics, compute_lafan1_fidelity
from .kinematic_metrics import (
    KinematicMetrics,
    KinematicThresholds,
    compute_kinematic_metrics,
)
from .model_info import BoundMotion
from .mujoco_kinematics import MujocoKinematics
from .scoring import (
    HardFailureConfig,
    QualityDecision,
    QualityScores,
    ScoringConfig,
    compute_quality_decision,
    compute_quality_scores,
)
from .types import MotionLoadResult


@dataclass(frozen=True)
class EvaluationMetadata:
    motion_name: str
    robot_name: str
    motion_category: str
    contact_policy: str
    source_bvh: Path | None = None
    source_pair_status: str = "unpaired"
    expected_pkl_sha256: str = ""
    expected_source_bvh_sha256: str = ""
    frame_alignment: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("motion_name", self.motion_name),
            ("robot_name", self.robot_name),
            ("motion_category", self.motion_category),
            ("contact_policy", self.contact_policy),
            ("source_pair_status", self.source_pair_status),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class QualityEvaluation:
    metadata: EvaluationMetadata
    load_result: MotionLoadResult
    scoring_config: ScoringConfig
    bound_motion: BoundMotion | None
    kinematic_metrics: KinematicMetrics | None
    contact_metrics: ContactMetrics | None
    fidelity_metrics: FidelityMetrics | None
    fidelity_status: str
    fidelity_reason: str
    scores: QualityScores | None
    events: tuple[QualityEvent, ...]
    decision: QualityDecision

    @property
    def data_valid(self) -> bool:
        return self.load_result.validation.data_valid


def evaluate_motion_result(
    load_result: MotionLoadResult,
    kinematics: MujocoKinematics,
    project_config: dict[str, Any],
    *,
    motion_name: str,
    motion_category: str,
    contact_policy: str,
    source_bvh: str | Path | None = None,
    source_pair_status: str | None = None,
    expected_pkl_sha256: str = "",
    expected_source_bvh_sha256: str = "",
    frame_alignment: str = "",
    notes: str = "",
) -> QualityEvaluation:
    """Evaluate one already-loaded motion without adding CLI or batch semantics."""
    resolved_source = (
        Path(source_bvh).expanduser().resolve() if source_bvh else None
    )
    resolved_pair_status = source_pair_status or (
        "provided_unverified" if resolved_source is not None else "unpaired"
    )
    metadata = EvaluationMetadata(
        motion_name=motion_name,
        robot_name=str(project_config["project"]["robot_name"]),
        motion_category=motion_category,
        contact_policy=contact_policy,
        source_bvh=resolved_source,
        source_pair_status=resolved_pair_status,
        expected_pkl_sha256=expected_pkl_sha256,
        expected_source_bvh_sha256=expected_source_bvh_sha256,
        frame_alignment=frame_alignment,
        notes=notes,
    )
    contact_config = ContactEvaluationConfig.from_project_config(
        project_config, contact_policy
    )
    scoring_config = ScoringConfig.from_project_config(project_config)
    hard_failure_config = HardFailureConfig.from_project_config(project_config)
    fidelity_config = FidelityConfig.from_project_config(project_config)
    validation = load_result.validation
    if not validation.data_valid:
        decision = compute_quality_decision(validation, hard_failure_config)
        return QualityEvaluation(
            metadata=metadata,
            load_result=load_result,
            scoring_config=scoring_config,
            bound_motion=None,
            kinematic_metrics=None,
            contact_metrics=None,
            fidelity_metrics=None,
            fidelity_status="disabled_invalid_motion",
            fidelity_reason="Input motion failed strict validation",
            scores=None,
            events=(),
            decision=decision,
        )

    motion = load_result.require_record()
    bound = kinematics.bind_motion(motion)
    kinematic = compute_kinematic_metrics(
        bound.record,
        bound.joint_limits,
        KinematicThresholds.from_project_config(project_config),
    )
    contact = compute_contact_metrics(
        bound,
        kinematics,
        contact_config,
        motion_category=motion_category,
    )
    fidelity: FidelityMetrics | None = None
    if resolved_pair_status == fidelity_config.verified_pair_status:
        if resolved_source is None:
            raise ValueError("Verified fidelity pair is missing source_bvh")
        fidelity = compute_lafan1_fidelity(
            bound,
            kinematics,
            resolved_source,
            fidelity_config,
            pair_status=resolved_pair_status,
            expected_pkl_sha256=expected_pkl_sha256,
            expected_source_bvh_sha256=expected_source_bvh_sha256,
            frame_alignment=frame_alignment,
        )
        fidelity_status = "enabled_verified"
        fidelity_reason = "PKL/BVH hashes and verified replay provenance matched"
    elif resolved_source is None:
        fidelity_status = "disabled_unpaired"
        fidelity_reason = "No reliably paired source BVH is available"
    else:
        fidelity_status = "disabled_unverified_pair"
        fidelity_reason = (
            "Source BVH was provided but pair provenance is not verified"
        )
    scores = compute_quality_scores(
        kinematic,
        contact,
        scoring_config,
        fidelity_valid_frame=(
            fidelity.fidelity_valid_frame if fidelity is not None else None
        ),
    )
    events = extract_quality_events(
        motion_name,
        bound.record,
        kinematic,
        contact,
    )
    decision = compute_quality_decision(
        validation,
        hard_failure_config,
        binding=bound.binding,
        kinematic=kinematic,
        contact=contact,
        events=events,
        scores=scores,
    )
    return QualityEvaluation(
        metadata=metadata,
        load_result=load_result,
        scoring_config=scoring_config,
        bound_motion=bound,
        kinematic_metrics=kinematic,
        contact_metrics=contact,
        fidelity_metrics=fidelity,
        fidelity_status=fidelity_status,
        fidelity_reason=fidelity_reason,
        scores=scores,
        events=events,
        decision=decision,
    )
