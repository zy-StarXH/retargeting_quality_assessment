from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .evaluation import QualityEvaluation, evaluate_motion_result
from .io import load_motion_from_project_config
from .manifest import (
    git_state,
    package_version,
    runtime_info,
    sha256_file,
    write_resolved_config,
    write_run_manifest,
)
from .mujoco_kinematics import MujocoKinematics
from .reporting import BatchReportPaths, write_batch_reports

if TYPE_CHECKING:
    from .plotting import PlotPaths


CATALOG_COLUMNS = (
    "motion_name",
    "pkl_path",
    "source_bvh",
    "motion_category",
    "contact_policy",
    "source_pair_status",
    "pkl_sha256",
    "source_bvh_sha256",
    "frame_alignment",
    "notes",
)

VERIFIED_PAIR_STATUSES = frozenset(
    {"verified_replay_exact", "verified_generated_exact"}
)


@dataclass(frozen=True)
class InputSpecification:
    path: Path
    motion_name: str
    source_bvh: Path | None
    motion_category: str
    contact_policy: str
    source_pair_status: str
    expected_pkl_sha256: str
    expected_source_bvh_sha256: str
    frame_alignment: str
    notes: str


@dataclass(frozen=True)
class InputOutcome:
    specification: InputSpecification
    input_sha256: str
    input_size_bytes: int
    source_bvh_sha256: str | None
    status: str
    hard_fail: bool | None
    failure_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    error: str | None
    fidelity_enabled: bool
    fidelity_status: str
    fidelity_reason: str
    plots: PlotPaths | None


@dataclass(frozen=True)
class EvaluationRunResult:
    output_dir: Path
    evaluations: tuple[QualityEvaluation, ...]
    outcomes: tuple[InputOutcome, ...]
    reports: BatchReportPaths | None
    resolved_config_yaml: Path
    run_manifest_json: Path
    succeeded_count: int
    failed_count: int
    skipped_count: int
    hard_fail_count: int

    @property
    def exit_code(self) -> int:
        return 0 if self.failed_count == 0 else 2


def _catalog_entries(path: Path | None) -> dict[Path, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in CATALOG_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Motion catalog is missing columns: {', '.join(missing)}")
        entries: dict[Path, dict[str, str]] = {}
        for row in reader:
            raw_path = Path(row["pkl_path"]).expanduser()
            if not raw_path.is_absolute():
                raw_path = path.parent / raw_path
            resolved = raw_path.resolve()
            if resolved in entries:
                raise ValueError(f"Duplicate pkl_path in motion catalog: {resolved}")
            entries[resolved] = {key: row.get(key, "") for key in CATALOG_COLUMNS}
        return entries


def _catalog_source_bvh(raw: str, catalog_path: Path | None) -> Path | None:
    if not raw:
        return None
    source = Path(raw).expanduser()
    if not source.is_absolute() and catalog_path is not None:
        source = catalog_path.parent / source
    return source.resolve()


def discover_inputs(
    *,
    input_path: str | Path | None,
    input_dir: str | Path | None,
    pattern: str,
    catalog_path: str | Path | None,
    motion_name: str | None = None,
    source_bvh: str | Path | None = None,
    motion_category: str | None = None,
    contact_policy: str | None = None,
    notes: str | None = None,
) -> tuple[InputSpecification, ...]:
    if (input_path is None) == (input_dir is None):
        raise ValueError("Exactly one of input_path or input_dir is required")
    catalog = (
        Path(catalog_path).expanduser().resolve() if catalog_path is not None else None
    )
    entries = _catalog_entries(catalog)
    directory: Path | None = None
    if input_path is not None:
        source_paths = (Path(input_path).expanduser().resolve(),)
        if not source_paths[0].is_file():
            raise ValueError(f"Input file does not exist: {source_paths[0]}")
    else:
        directory = Path(input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Input directory does not exist: {directory}")
        if not pattern or Path(pattern).is_absolute():
            raise ValueError("Input pattern must be a non-empty relative glob")
        source_paths = tuple(
            sorted(path.resolve() for path in directory.glob(pattern) if path.is_file())
        )
        if not source_paths:
            raise ValueError(
                f"No input files matched pattern {pattern!r} under {directory}"
            )
        if motion_name is not None or source_bvh is not None:
            raise ValueError("motion_name and source_bvh overrides require single-file input")

    explicit_source = (
        Path(source_bvh).expanduser().resolve() if source_bvh is not None else None
    )
    if explicit_source is not None and not explicit_source.is_file():
        raise ValueError(f"Source BVH does not exist: {explicit_source}")

    specifications: list[InputSpecification] = []
    for path in source_paths:
        row = entries.get(path, {})
        if directory is None:
            fallback_name = path.stem
        else:
            relative = path.relative_to(directory).with_suffix("")
            fallback_name = "__".join(relative.parts)
        resolved_name = motion_name or row.get("motion_name") or fallback_name
        resolved_source = explicit_source or _catalog_source_bvh(
            row.get("source_bvh", ""), catalog
        )
        if explicit_source is not None:
            pair_status = "provided_unverified"
            expected_pkl_hash = ""
            expected_source_hash = ""
            alignment = ""
        else:
            pair_status = row.get("source_pair_status") or "unpaired"
            expected_pkl_hash = row.get("pkl_sha256", "")
            expected_source_hash = row.get("source_bvh_sha256", "")
            alignment = row.get("frame_alignment", "")
        if pair_status in VERIFIED_PAIR_STATUSES:
            if resolved_source is None:
                raise ValueError(
                    f"Verified pair {resolved_name!r} has no source BVH"
                )
            if len(expected_pkl_hash) != 64 or len(expected_source_hash) != 64:
                raise ValueError(
                    f"Verified pair {resolved_name!r} requires pinned PKL/BVH hashes"
                )
            if not alignment:
                raise ValueError(
                    f"Verified pair {resolved_name!r} requires frame_alignment"
                )
        elif pair_status == "unpaired" and resolved_source is not None:
            raise ValueError(
                f"Unpaired catalog row {resolved_name!r} cannot name a source BVH"
            )
        specifications.append(
            InputSpecification(
                path=path,
                motion_name=resolved_name,
                source_bvh=resolved_source,
                motion_category=(
                    motion_category or row.get("motion_category") or "unknown"
                ),
                contact_policy=(
                    contact_policy or row.get("contact_policy") or "descriptive_only"
                ),
                source_pair_status=pair_status,
                expected_pkl_sha256=expected_pkl_hash,
                expected_source_bvh_sha256=expected_source_hash,
                frame_alignment=alignment,
                notes=notes if notes is not None else row.get("notes", ""),
            )
        )
    names = tuple(spec.motion_name for spec in specifications)
    if len(set(names)) != len(names):
        raise ValueError("Discovered inputs produce duplicate motion names")
    return tuple(specifications)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_output_dir(
    destination: Path,
    inputs: tuple[InputSpecification, ...],
    protected_input_dirs: Sequence[str | Path],
) -> None:
    for raw_directory in protected_input_dirs:
        directory = Path(raw_directory).expanduser().resolve()
        if _is_within(destination, directory):
            raise ValueError(
                f"Output directory must be outside the input directory: {directory}"
            )
    for specification in inputs:
        if _is_within(destination, specification.path.parent):
            raise ValueError(
                "Output directory must be outside every original input directory: "
                f"{destination} is inside {specification.path.parent}"
            )
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"Output directory must be empty: {destination}")
    else:
        destination.mkdir(parents=True)


def _relative_output_files(destination: Path) -> tuple[dict[str, Any], ...]:
    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if path.name == "run_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(destination)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(files)


def _outcome_dict(outcome: InputOutcome) -> dict[str, Any]:
    specification = outcome.specification
    return {
        "input_path": str(specification.path),
        "input_sha256": outcome.input_sha256,
        "input_size_bytes": outcome.input_size_bytes,
        "motion_name": specification.motion_name,
        "motion_category": specification.motion_category,
        "contact_policy": specification.contact_policy,
        "source_pair_status": specification.source_pair_status,
        "expected_pkl_sha256": specification.expected_pkl_sha256 or None,
        "expected_source_bvh_sha256": (
            specification.expected_source_bvh_sha256 or None
        ),
        "frame_alignment": specification.frame_alignment or None,
        "source_bvh": str(specification.source_bvh) if specification.source_bvh else None,
        "source_bvh_sha256": outcome.source_bvh_sha256,
        "status": outcome.status,
        "hard_fail": outcome.hard_fail,
        "failure_codes": list(outcome.failure_codes),
        "warning_codes": list(outcome.warning_codes),
        "error": outcome.error,
        "fidelity_enabled": outcome.fidelity_enabled,
        "fidelity_status": outcome.fidelity_status,
        "fidelity_reason": outcome.fidelity_reason,
        "plots": (
            {
                "joint_velocity_png": str(outcome.plots.joint_velocity_png),
                "foot_contact_png": str(outcome.plots.foot_contact_png),
                "fidelity_png": (
                    str(outcome.plots.fidelity_png)
                    if outcome.plots.fidelity_png is not None
                    else None
                ),
            }
            if outcome.plots
            else None
        ),
    }


def run_evaluations(
    project_config: dict[str, Any],
    specifications: tuple[InputSpecification, ...],
    output_dir: str | Path,
    *,
    plots_enabled: bool = True,
    execution_command: Sequence[str] = (),
    pattern: str | None = None,
    input_mode: str = "single",
    protected_input_dirs: Sequence[str | Path] = (),
    progress: Callable[[str], None] | None = None,
) -> EvaluationRunResult:
    if not specifications:
        raise ValueError("No input specifications were provided")
    destination = Path(output_dir).expanduser().resolve()
    _prepare_output_dir(destination, specifications, protected_input_dirs)
    kinematics = MujocoKinematics.from_project_config(project_config)
    evaluations: list[QualityEvaluation] = []
    outcomes: list[InputOutcome] = []
    plot_dir = destination / "plots"

    for input_index, specification in enumerate(specifications, start=1):
        if progress:
            progress(
                f"[{input_index}/{len(specifications)}] evaluating "
                f"{specification.motion_name}"
            )
        input_hash = sha256_file(specification.path)
        source_hash = (
            sha256_file(specification.source_bvh)
            if specification.source_bvh is not None
            and specification.source_bvh.is_file()
            else None
        )
        evaluation: QualityEvaluation | None = None
        plots: PlotPaths | None = None
        try:
            load_result = load_motion_from_project_config(
                specification.path, project_config
            )
            evaluation = evaluate_motion_result(
                load_result,
                kinematics,
                project_config,
                motion_name=specification.motion_name,
                motion_category=specification.motion_category,
                contact_policy=specification.contact_policy,
                source_bvh=specification.source_bvh,
                source_pair_status=specification.source_pair_status,
                expected_pkl_sha256=specification.expected_pkl_sha256,
                expected_source_bvh_sha256=(
                    specification.expected_source_bvh_sha256
                ),
                frame_alignment=specification.frame_alignment,
                notes=specification.notes,
            )
            evaluations.append(evaluation)
            if evaluation.data_valid and plots_enabled:
                from .plotting import plot_evaluation

                plots = plot_evaluation(evaluation, plot_dir)
            status = "evaluated" if evaluation.data_valid else "failed_validation"
            error = None
            fidelity_enabled = evaluation.fidelity_metrics is not None
            fidelity_status = evaluation.fidelity_status
            fidelity_reason = evaluation.fidelity_reason
        except Exception as exc:
            status = "failed_exception"
            error = f"{type(exc).__name__}: {exc}"
            fidelity_enabled = False
            fidelity_status = "failed_exception"
            fidelity_reason = error

        decision = evaluation.decision if evaluation is not None else None
        outcomes.append(
            InputOutcome(
                specification=specification,
                input_sha256=input_hash,
                input_size_bytes=specification.path.stat().st_size,
                source_bvh_sha256=source_hash,
                status=status,
                hard_fail=decision.hard_fail if decision else None,
                failure_codes=(
                    tuple(reason.code for reason in decision.failure_reasons)
                    if decision
                    else ()
                ),
                warning_codes=(
                    tuple(reason.code for reason in decision.warning_reasons)
                    if decision
                    else ()
                ),
                error=error,
                fidelity_enabled=fidelity_enabled,
                fidelity_status=fidelity_status,
                fidelity_reason=fidelity_reason,
                plots=plots,
            )
        )
        if progress:
            progress(
                f"[{input_index}/{len(specifications)}] completed "
                f"{specification.motion_name}: status={status}, "
                f"hard_fail={decision.hard_fail if decision else None}"
            )

    reports = (
        write_batch_reports(tuple(evaluations), destination)
        if evaluations
        else None
    )
    resolved_path = write_resolved_config(
        project_config, destination / "resolved_config.yaml"
    )
    succeeded = sum(outcome.status == "evaluated" for outcome in outcomes)
    failed = sum(outcome.status != "evaluated" for outcome in outcomes)
    hard_fail_count = sum(outcome.hard_fail is True for outcome in outcomes)
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(project_config["_config_path"])
    manifest = {
        "schema_version": 1,
        "tool": {
            "name": "gmr-quality-eval",
            "version": package_version(),
            "git": git_state(project_root),
        },
        "gmr": git_state(project_config["paths"]["gmr_root"]),
        "runtime": runtime_info(),
        "robot_model": {
            "xml_path": str(kinematics.info.xml_path),
            "xml_sha256": kinematics.info.xml_sha256,
            "nq": kinematics.info.nq,
            "nv": kinematics.info.nv,
            "evaluated_dofs": kinematics.info.num_dofs,
        },
        "configuration": {
            "source_path": str(config_path.resolve()),
            "source_sha256": sha256_file(config_path),
            "resolved_path": str(resolved_path.relative_to(destination)),
            "resolved_sha256": sha256_file(resolved_path),
            "thresholds": project_config["thresholds"],
            "scoring": project_config["scoring"],
            "contact_policies": project_config["contact_policies"],
            "fidelity": project_config["fidelity"],
            "fidelity_artifacts": {
                "ik_config_path": project_config["paths"][
                    "gmr_lafan1_g1_ik_config"
                ],
                "ik_config_sha256": sha256_file(
                    project_config["paths"]["gmr_lafan1_g1_ik_config"]
                ),
                "gmr_lafan1_loader_path": project_config["paths"][
                    "gmr_lafan1_loader"
                ],
                "gmr_lafan1_loader_sha256": sha256_file(
                    project_config["paths"]["gmr_lafan1_loader"]
                ),
                "pair_replay_evidence_path": project_config["paths"][
                    "pair_replay_evidence"
                ],
                "pair_replay_evidence_sha256": sha256_file(
                    project_config["paths"]["pair_replay_evidence"]
                ),
            },
        },
        "execution": {
            "command": list(execution_command),
            "input_mode": input_mode,
            "pattern": pattern,
            "plots_enabled": plots_enabled,
            "fidelity_requested": any(
                specification.source_bvh is not None
                for specification in specifications
            ),
            "fidelity_enabled": any(
                outcome.fidelity_enabled for outcome in outcomes
            ),
            "fidelity_enabled_count": sum(
                outcome.fidelity_enabled for outcome in outcomes
            ),
        },
        "counts": {
            "discovered": len(specifications),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": 0,
            "hard_fail": hard_fail_count,
        },
        "inputs": [_outcome_dict(outcome) for outcome in outcomes],
        "outputs": _relative_output_files(destination),
    }
    manifest_path = write_run_manifest(
        manifest, destination / "run_manifest.json"
    )
    return EvaluationRunResult(
        output_dir=destination,
        evaluations=tuple(evaluations),
        outcomes=tuple(outcomes),
        reports=reports,
        resolved_config_yaml=resolved_path,
        run_manifest_json=manifest_path,
        succeeded_count=succeeded,
        failed_count=failed,
        skipped_count=0,
        hard_fail_count=hard_fail_count,
    )
