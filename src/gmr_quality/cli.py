from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import load_config
from .doctor import format_human_report, run_doctor
from .runner import discover_inputs, run_evaluations


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "unitree_g1.yaml"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "motion_catalog.csv"
DEFAULT_STEP10_SELECTION = PROJECT_ROOT / "data" / "step10_source_selection.csv"
DEFAULT_V1_CONFIG = PROJECT_ROOT / "configs" / "unitree_g1_v1.yaml"
DEFAULT_STEP10_CATALOG = (
    PROJECT_ROOT / "outputs" / "step10_dataset" / "step10_motion_catalog.csv"
)
DEFAULT_STEP11_CONFIG = PROJECT_ROOT / "configs" / "step11_beyondmimic.yaml"
DEFAULT_STEP12_CONFIG = PROJECT_ROOT / "configs" / "step12_interaction.yaml"
DEFAULT_QUALITY_V2_CONFIG = PROJECT_ROOT / "configs" / "quality_assessment_v2.yaml"
DEFAULT_STEP12_SIMULATION_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "step12_interaction_experiment"
    / "interaction_learning_value.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmr-quality",
        description="Offline quality assessment for GMR-retargeted robot motion.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check the local GMR environment without computing quality metrics.",
    )
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    doctor.add_argument("--json", action="store_true", dest="as_json")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate one trusted GMR pickle or an isolated directory batch.",
    )
    inputs = evaluate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, dest="input_path")
    inputs.add_argument("--input-dir", type=Path)
    evaluate.add_argument("--pattern", default="*.pkl")
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument("--catalog", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--motion-name")
    evaluate.add_argument("--source-bvh", type=Path)
    evaluate.add_argument("--motion-category")
    evaluate.add_argument("--contact-policy")
    evaluate.add_argument("--notes")
    evaluate.add_argument("--no-plots", action="store_true")
    evaluate.add_argument("--fail-on-hard-fail", action="store_true")
    evaluate.add_argument("--json", action="store_true", dest="as_json")

    validate_pipeline = subparsers.add_parser(
        "validate-pipeline",
        help="Inject controlled faults and verify the complete evaluation pipeline.",
    )
    validate_pipeline.add_argument("--input", type=Path, required=True)
    validate_pipeline.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate_pipeline.add_argument("--output-dir", type=Path, required=True)
    validate_pipeline.add_argument("--source-frame", type=int, default=0)
    validate_pipeline.add_argument("--no-plots", action="store_true")
    validate_pipeline.add_argument("--json", action="store_true", dest="as_json")

    prepare_step10 = subparsers.add_parser(
        "prepare-step10-dataset",
        help="Headlessly retarget the frozen stratified LAFAN1 experiment cohort.",
    )
    prepare_step10.add_argument(
        "--selection", type=Path, default=DEFAULT_STEP10_SELECTION
    )
    prepare_step10.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_step10.add_argument("--output-dir", type=Path, required=True)
    prepare_step10.add_argument("--json", action="store_true", dest="as_json")

    run_step10 = subparsers.add_parser(
        "run-step10-experiment",
        help="Evaluate and analyze the frozen stratified step-10 cohort.",
    )
    run_step10.add_argument("--config", type=Path, default=DEFAULT_V1_CONFIG)
    run_step10.add_argument("--catalog", type=Path, default=DEFAULT_STEP10_CATALOG)
    run_step10.add_argument(
        "--selection", type=Path, default=DEFAULT_STEP10_SELECTION
    )
    run_step10.add_argument("--output-dir", type=Path, required=True)
    run_step10.add_argument("--json", action="store_true", dest="as_json")

    refresh_step10 = subparsers.add_parser(
        "refresh-step10-review-packet",
        help="Add deterministic anomaly timelines to an existing step-10 run.",
    )
    refresh_step10.add_argument("--output-dir", type=Path, required=True)
    refresh_step10.add_argument("--json", action="store_true", dest="as_json")

    finalize_step10 = subparsers.add_parser(
        "finalize-step10-experiment",
        help="Validate the frozen manual review and hash the complete step-10 run.",
    )
    finalize_step10.add_argument("--output-dir", type=Path, required=True)
    finalize_step10.add_argument("--review", type=Path, required=True)
    finalize_step10.add_argument("--json", action="store_true", dest="as_json")

    prepare_step11 = subparsers.add_parser(
        "prepare-step11-experiment",
        help="Freeze cohorts and convert Step-10 motions for BeyondMimic training.",
    )
    prepare_step11.add_argument("--config", type=Path, default=DEFAULT_STEP11_CONFIG)
    prepare_step11.add_argument("--output-dir", type=Path, required=True)
    prepare_step11.add_argument("--json", action="store_true", dest="as_json")

    finalize_step11 = subparsers.add_parser(
        "finalize-step11-experiment",
        help="Validate all policy results, analyze learning value, and gate score calibration.",
    )
    finalize_step11.add_argument("--config", type=Path, default=DEFAULT_STEP11_CONFIG)
    finalize_step11.add_argument("--output-dir", type=Path, required=True)
    finalize_step11.add_argument("--json", action="store_true", dest="as_json")

    run_step12 = subparsers.add_parser(
        "run-step12-experiment",
        help="Generate and validate the controlled G1 object-interaction benchmark.",
    )
    run_step12.add_argument("--config", type=Path, default=DEFAULT_STEP12_CONFIG)
    run_step12.add_argument("--output-dir", type=Path, required=True)
    run_step12.add_argument("--json", action="store_true", dest="as_json")

    evaluate_interaction = subparsers.add_parser(
        "evaluate-interaction",
        help=(
            "Evaluate robot motion, hand-object interaction, and task effectiveness "
            "from one interaction NPZ or an isolated directory batch."
        ),
    )
    interaction_inputs = evaluate_interaction.add_mutually_exclusive_group(required=True)
    interaction_inputs.add_argument("--input", type=Path, dest="input_path")
    interaction_inputs.add_argument("--input-dir", type=Path)
    evaluate_interaction.add_argument("--pattern", default="*.npz")
    evaluate_interaction.add_argument("--config", type=Path, default=DEFAULT_QUALITY_V2_CONFIG)
    evaluate_interaction.add_argument("--output-dir", type=Path, required=True)
    evaluate_interaction.add_argument("--no-plots", action="store_true")
    evaluate_interaction.add_argument("--fail-on-hard-fail", action="store_true")
    evaluate_interaction.add_argument("--json", action="store_true", dest="as_json")

    finalize_quality_v2 = subparsers.add_parser(
        "finalize-quality-assessment-v2",
        help=(
            "Relate an existing quality-v2 run to controlled simulation targets "
            "without claiming score calibration."
        ),
    )
    finalize_quality_v2.add_argument(
        "--config", type=Path, default=DEFAULT_QUALITY_V2_CONFIG
    )
    finalize_quality_v2.add_argument("--output-dir", type=Path, required=True)
    finalize_quality_v2.add_argument(
        "--simulation-results", type=Path, default=DEFAULT_STEP12_SIMULATION_RESULTS
    )
    finalize_quality_v2.add_argument("--bootstrap-resamples", type=int, default=10_000)
    finalize_quality_v2.add_argument("--bootstrap-seed", type=int, default=20260807)
    finalize_quality_v2.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    args = build_parser().parse_args(raw_argv)
    try:
        if args.command == "doctor":
            config = load_config(args.config)
            report = run_doctor(config, args.catalog)
            print(report.as_json() if args.as_json else format_human_report(report))
            return 0 if report.ok else 1
        if args.command == "evaluate":
            config = load_config(args.config)
            catalog_path = (
                args.catalog
                if args.catalog is not None
                else Path(config["paths"]["motion_catalog"])
            )
            if args.catalog is not None and not args.catalog.expanduser().is_file():
                raise ValueError(f"Motion catalog does not exist: {args.catalog}")
            specifications = discover_inputs(
                input_path=args.input_path,
                input_dir=args.input_dir,
                pattern=args.pattern,
                catalog_path=catalog_path,
                motion_name=args.motion_name,
                source_bvh=args.source_bvh,
                motion_category=args.motion_category,
                contact_policy=args.contact_policy,
                notes=args.notes,
            )
            result = run_evaluations(
                config,
                specifications,
                args.output_dir,
                plots_enabled=not args.no_plots,
                execution_command=("gmr-quality", *raw_argv),
                pattern=args.pattern if args.input_dir is not None else None,
                input_mode="directory" if args.input_dir is not None else "single",
                protected_input_dirs=(args.input_dir,) if args.input_dir else (),
            )
            payload = {
                "ok": result.failed_count == 0,
                "output_dir": str(result.output_dir),
                "discovered": len(result.outcomes),
                "succeeded": result.succeeded_count,
                "failed": result.failed_count,
                "skipped": result.skipped_count,
                "hard_fail": result.hard_fail_count,
                "fidelity_enabled": sum(
                    outcome.fidelity_enabled for outcome in result.outcomes
                ),
                "manifest": str(result.run_manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Evaluation run complete: "
                    f"discovered={payload['discovered']}, "
                    f"succeeded={payload['succeeded']}, "
                    f"failed={payload['failed']}, "
                    f"hard_fail={payload['hard_fail']}"
                )
                print(f"Output: {result.output_dir}")
                print(f"Manifest: {result.run_manifest_json}")
                if args.source_bvh is not None and not any(
                    outcome.fidelity_enabled for outcome in result.outcomes
                ):
                    print(
                        "Warning: source BVH was recorded as unverified metadata; "
                        "fidelity requires a hash-pinned, replay-verified catalog pair."
                    )
            if result.failed_count:
                return 2
            if args.fail_on_hard_fail and result.hard_fail_count:
                return 3
            return 0
        if args.command == "validate-pipeline":
            from .pipeline_validation import run_pipeline_validation

            config = load_config(args.config)
            result = run_pipeline_validation(
                config,
                args.input,
                args.output_dir,
                source_frame=args.source_frame,
                plots_enabled=not args.no_plots,
                execution_command=("gmr-quality", *raw_argv),
            )
            payload = {
                "ok": result.ok,
                "output_dir": str(result.output_dir),
                "checks_passed": sum(check["passed"] for check in result.checks),
                "checks_total": len(result.checks),
                "scenarios": len(result.evaluation_run.outcomes),
                "runner_succeeded": result.evaluation_run.succeeded_count,
                "runner_failed_as_expected": result.evaluation_run.failed_count,
                "validation_report": str(result.validation_json),
                "manifest": str(result.evaluation_run.run_manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Pipeline validation complete: "
                    f"checks={payload['checks_passed']}/{payload['checks_total']}, "
                    f"scenarios={payload['scenarios']}, "
                    "expected_invalid="
                    f"{payload['runner_failed_as_expected']}"
                )
                print(f"Output: {result.output_dir}")
                print(f"Validation: {result.validation_json}")
                print(f"Manifest: {result.evaluation_run.run_manifest_json}")
            return 0 if result.ok else 1
        if args.command == "prepare-step10-dataset":
            from .step10_dataset import prepare_step10_dataset

            config = load_config(args.config)
            result = prepare_step10_dataset(
                config,
                args.selection,
                args.output_dir,
                progress=(None if args.as_json else print),
            )
            payload = {
                "ok": True,
                "output_dir": str(result.output_dir),
                "motion_count": result.motion_count,
                "frame_count": result.frame_count,
                "catalog": str(result.catalog_csv),
                "manifest": str(result.manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Step-10 dataset complete: "
                    f"motions={result.motion_count}, frames={result.frame_count}"
                )
                print(f"Catalog: {result.catalog_csv}")
                print(f"Manifest: {result.manifest_json}")
            return 0
        if args.command == "run-step10-experiment":
            from .step10_experiment import run_step10_experiment

            config = load_config(args.config)
            result = run_step10_experiment(
                config,
                args.catalog,
                args.selection,
                args.output_dir,
                execution_command=("gmr-quality", *raw_argv),
                progress=(None if args.as_json else print),
            )
            payload = {
                "ok": result.ok,
                "output_dir": str(result.output_dir),
                "motions": len(result.evaluation_run.outcomes),
                "succeeded": result.evaluation_run.succeeded_count,
                "failed": result.evaluation_run.failed_count,
                "fidelity_enabled": sum(
                    outcome.fidelity_enabled
                    for outcome in result.evaluation_run.outcomes
                ),
                "automated_analysis": str(result.automated_analysis_json),
                "review_candidates": str(result.review_candidates_csv),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Step-10 automated experiment complete: "
                    f"motions={payload['motions']}, succeeded={payload['succeeded']}, "
                    f"failed={payload['failed']}, fidelity={payload['fidelity_enabled']}"
                )
                print(f"Automated analysis: {result.automated_analysis_json}")
                print(f"Review candidates: {result.review_candidates_csv}")
            return 0 if result.ok else 1
        if args.command == "finalize-step10-experiment":
            from .step10_experiment import finalize_step10_experiment

            result = finalize_step10_experiment(args.output_dir, args.review)
            payload = {
                "ok": result.ok,
                "output_dir": str(result.output_dir),
                "review_csv": str(result.review_csv),
                "review_summary": str(result.review_summary_json),
                "manifest": str(result.manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(f"Step-10 finalization complete: ok={result.ok}")
                print(f"Review: {result.review_csv}")
                print(f"Manifest: {result.manifest_json}")
            return 0 if result.ok else 1
        if args.command == "refresh-step10-review-packet":
            from .step10_experiment import refresh_step10_review_packet

            candidates = refresh_step10_review_packet(args.output_dir)
            payload = {
                "ok": True,
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "review_candidates": str(candidates),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(f"Step-10 review packet refreshed: {candidates}")
            return 0
        if args.command == "prepare-step11-experiment":
            from .step11_protocol import prepare_step11_experiment

            result = prepare_step11_experiment(
                args.config,
                args.output_dir,
                progress=(None if args.as_json else print),
            )
            payload = {
                "ok": True,
                "preparation_complete": True,
                "ready_for_training": result.ready_for_training,
                "blocker_codes": list(result.blocker_codes),
                "output_dir": str(result.output_dir),
                "cohort_selection": str(result.cohort_csv),
                "training_jobs": str(result.jobs_csv),
                "environment_audit": str(result.environment_audit_json),
                "manifest": str(result.data_manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Step-11 preparation complete: "
                    f"ready_for_training={result.ready_for_training}, "
                    f"blockers={list(result.blocker_codes)}"
                )
                print(f"Manifest: {result.data_manifest_json}")
            return 0
        if args.command == "finalize-step11-experiment":
            from .step11_analysis import finalize_step11_experiment

            result = finalize_step11_experiment(args.config, args.output_dir)
            if args.as_json:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Step-11 finalization complete: "
                    f"jobs={result['policy_result_count']}, "
                    f"score_is_calibrated={result['score_is_calibrated']}"
                )
            return 0
        if args.command == "run-step12-experiment":
            from .step12_experiment import run_step12_experiment

            result = run_step12_experiment(
                args.config,
                args.output_dir,
                progress=(None if args.as_json else print),
            )
            payload = {
                "ok": True,
                "output_dir": str(result.output_dir),
                "demonstration_count": result.demonstration_count,
                "robustness_episode_count": result.robustness_episode_count,
                "score_is_calibrated": result.score_is_calibrated,
                "manifest": str(result.manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Step-12 interaction experiment complete: "
                    f"demos={result.demonstration_count}, "
                    f"episodes={result.robustness_episode_count}, "
                    f"score_is_calibrated={result.score_is_calibrated}"
                )
                print(f"Manifest: {result.manifest_json}")
            return 0
        if args.command == "evaluate-interaction":
            from .interaction_runner import run_interaction_evaluations

            result = run_interaction_evaluations(
                args.config,
                args.output_dir,
                input_path=args.input_path,
                input_dir=args.input_dir,
                pattern=args.pattern,
                plots_enabled=not args.no_plots,
                execution_command=("gmr-quality", *raw_argv),
            )
            payload = {
                "ok": result.failed_count == 0,
                "output_dir": str(result.output_dir),
                "discovered": result.discovered_count,
                "succeeded": result.succeeded_count,
                "failed": result.failed_count,
                "hard_fail": result.hard_fail_count,
                "score_is_calibrated": False,
                "manifest": str(result.run_manifest_json),
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Interaction quality run complete: "
                    f"discovered={result.discovered_count}, "
                    f"succeeded={result.succeeded_count}, "
                    f"failed={result.failed_count}, "
                    f"hard_fail={result.hard_fail_count}"
                )
                print(f"Output: {result.output_dir}")
                print(f"Manifest: {result.run_manifest_json}")
            if result.failed_count:
                return 2
            if args.fail_on_hard_fail and result.hard_fail_count:
                return 3
            return 0
        if args.command == "finalize-quality-assessment-v2":
            from .interaction_runner import finalize_quality_v2_validation

            result = finalize_quality_v2_validation(
                args.config,
                args.output_dir,
                args.simulation_results,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
            payload = {
                "ok": True,
                "output_dir": str(result.output_dir),
                "motions": result.motion_count,
                "correlations": str(result.correlation_csv),
                "group_summary": str(result.group_summary_csv),
                "decision": str(result.decision_json),
                "report": str(result.report_markdown),
                "manifest": str(result.run_manifest_json),
                "score_is_calibrated": False,
            }
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    "Quality-assessment v2 validation complete: "
                    f"motions={result.motion_count}, score_is_calibrated=False"
                )
                print(f"Report: {result.report_markdown}")
                print(f"Manifest: {result.run_manifest_json}")
            return 0
        raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
