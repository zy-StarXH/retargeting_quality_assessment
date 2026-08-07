from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import rankdata, spearmanr, t

from .manifest import sha256_file
from .step11_protocol import load_step11_config


METRICS = (
    "success_rate", "episode_length_s", "mpjpe_m", "weighted_mpjpe_m", "fall_rate"
)


def _finite_or_none(value: Any) -> float | None:
    """Return a strict-JSON-compatible scalar for possibly undefined statistics."""
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return float(values.mean() - half), float(values.mean() + half)


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, count: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(count, len(x)))
    x_rank = rankdata(np.asarray(x)[indices], axis=1)
    y_rank = rankdata(np.asarray(y)[indices], axis=1)
    x_rank -= x_rank.mean(axis=1, keepdims=True)
    y_rank -= y_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x_rank**2, axis=1) * np.sum(y_rank**2, axis=1))
    valid = denominator > 0
    if not np.any(valid):
        return float("nan"), float("nan")
    values = np.sum(x_rank[valid] * y_rank[valid], axis=1) / denominator[valid]
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([train_x, np.ones(len(train_x))])
    lower = np.r_[np.zeros(train_x.shape[1]), -np.inf]
    upper = np.full(train_x.shape[1] + 1, np.inf)
    fit = lsq_linear(design, train_y, bounds=(lower, upper))
    return np.column_stack([test_x, np.ones(len(test_x))]) @ fit.x


def _oof_predictions(table: pd.DataFrame, dimensions: list[str]) -> np.ndarray:
    prediction = np.full(len(table), np.nan)
    x = table[dimensions].to_numpy(dtype=float) / 100.0
    y = table["success_rate"].to_numpy(dtype=float)
    for category in table["motion_category"].unique():
        test = (table["motion_category"] == category).to_numpy()
        prediction[test] = _fit_predict(x[~test], y[~test], x[test])
    return prediction


def _paired_comparisons(run_table: pd.DataFrame) -> pd.DataFrame:
    """Compare cohorts with seed-paired differences and t-based 95% CIs."""
    rows: list[dict[str, Any]] = []
    pairs = (("top_k", "random_k"), ("top_k", "bottom_k"), ("random_k", "bottom_k"))
    for cohort_a, cohort_b in pairs:
        left = run_table.loc[run_table.cohort == cohort_a].set_index("seed")
        right = run_table.loc[run_table.cohort == cohort_b].set_index("seed")
        seeds = left.index.intersection(right.index)
        if len(seeds) != len(left) or len(seeds) != len(right):
            raise ValueError(f"Seed mismatch for {cohort_a} versus {cohort_b}")
        for metric in ("convergence_iteration", *METRICS):
            difference = left.loc[seeds, metric].to_numpy(dtype=float) - right.loc[seeds, metric].to_numpy(dtype=float)
            low, high = _ci95(difference)
            rows.append(
                {
                    "cohort_a": cohort_a,
                    "cohort_b": cohort_b,
                    "metric": metric,
                    "paired_seed_count": len(seeds),
                    "mean_difference_a_minus_b": float(difference.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def finalize_step11_experiment(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_step11_config(config_path)
    output = Path(output_dir).expanduser().resolve()
    jobs = pd.read_csv(output / "training_jobs.csv")
    expected = len(config["cohorts"]["groups"]) * len(config["training"]["policy_seeds"])
    if len(jobs) != expected:
        raise ValueError(f"Expected {expected} frozen training jobs, found {len(jobs)}")
    missing = [path for path in jobs["result_json"] if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            f"STEP11_RESULTS_INCOMPLETE: {len(missing)}/{expected} policy result files are missing; "
            "calibration is forbidden until all controlled runs finish"
        )
    selection = pd.read_csv(output / "cohort_selection.csv")
    run_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    for _, job in jobs.iterrows():
        result = json.loads(Path(job["result_json"]).read_text(encoding="utf-8"))
        if result.get("schema_version") != "step11_policy_result_v1" or int(result["seed"]) != int(job["seed"]):
            raise ValueError(f"Invalid result schema or seed for {job['job_id']}")
        expected_names = set(selection.loc[selection.cohort == job["cohort"], "motion_name"])
        actual_names = {row["motion_name"] for row in result["motions"]}
        if actual_names != expected_names:
            raise ValueError(f"Motion coverage mismatch for {job['job_id']}")
        converged = bool(result.get("converged", result.get("convergence_iteration") is not None))
        convergence_iteration = result.get("convergence_iteration")
        if convergence_iteration is None:
            # Right-censor policies that never reached a meaningful timeout
            # plateau.  This is a valid negative learning outcome, not missing
            # data, and must not be silently dropped from group comparisons.
            convergence_iteration = int(config["training"]["max_iterations"]) + 1
        frame = pd.DataFrame(result["motions"])
        if not all(column in frame for column in METRICS) or frame[list(METRICS)].isna().any().any():
            raise ValueError(f"Missing required learning metrics for {job['job_id']}")
        run_rows.append(
            {
                "job_id": job["job_id"], "cohort": job["cohort"], "seed": int(job["seed"]),
                "convergence_iteration": int(convergence_iteration),
                "converged": converged,
                **{column: float(frame[column].mean()) for column in METRICS},
            }
        )
        for row in result["motions"]:
            motion_rows.append({"job_id": job["job_id"], "cohort": job["cohort"], "seed": int(job["seed"]), **row})
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_table = pd.DataFrame(run_rows)
    motion_table = pd.DataFrame(motion_rows)
    run_table.to_csv(analysis_dir / "run_results.csv", index=False)
    motion_table.to_csv(analysis_dir / "motion_learning_results.csv", index=False)

    summary_rows = []
    for cohort, group in run_table.groupby("cohort", sort=False):
        row: dict[str, Any] = {
            "cohort": cohort,
            "seed_count": len(group),
            "converged_count": int(group["converged"].sum()),
            "converged_rate": float(group["converged"].mean()),
        }
        for metric in ("convergence_iteration", *METRICS):
            values = group[metric].to_numpy(dtype=float)
            low, high = _ci95(values)
            row.update({f"{metric}_mean": values.mean(), f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        summary_rows.append(row)
    cohort_summary = pd.DataFrame(summary_rows)
    cohort_summary.to_csv(analysis_dir / "cohort_summary.csv", index=False)
    _paired_comparisons(run_table).to_csv(analysis_dir / "paired_cohort_comparisons.csv", index=False)

    offline = pd.read_csv(config["inputs"]["step10_motion_results"])
    full = motion_table[motion_table.cohort == "full"].groupby("motion_name", as_index=False)[list(METRICS)].mean()
    joined = offline.merge(full, on="motion_name", validate="one_to_one")
    dimensions = list(config["calibration"]["dimensions"])
    bootstrap_count = int(config["evaluation"]["bootstrap_resamples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    correlation_rows = []
    targets = {
        "success_rate": joined.success_rate.to_numpy(),
        "episode_length": joined.episode_length_s.to_numpy(),
        "negative_mpjpe": -joined.mpjpe_m.to_numpy(),
        "negative_weighted_mpjpe": -joined.weighted_mpjpe_m.to_numpy(),
        "negative_fall_rate": -joined.fall_rate.to_numpy(),
    }
    for metric in [*dimensions, "quality_score"]:
        for target_name, target in targets.items():
            x = joined[metric].to_numpy(dtype=float)
            rho = float(spearmanr(x, target).statistic)
            low, high = _bootstrap_spearman(x, target, bootstrap_count, bootstrap_seed)
            correlation_rows.append({"offline_metric": metric, "learning_target": target_name, "spearman_rho": rho, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high, "n_motions": len(joined)})
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(analysis_dir / "offline_learning_correlations.csv", index=False)

    ablation_rows = []
    for removed in [None, *dimensions]:
        used = [name for name in dimensions if name != removed]
        prediction = _oof_predictions(joined, used)
        rho = float(spearmanr(prediction, joined.success_rate).statistic)
        low, high = _bootstrap_spearman(prediction, joined.success_rate.to_numpy(), bootstrap_count, bootstrap_seed)
        ablation_rows.append({"ablation": "all_dimensions" if removed is None else f"without_{removed}", "dimensions_used": json.dumps(used), "oof_spearman_rho": rho, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    ablations = pd.DataFrame(ablation_rows)
    ablations.to_csv(analysis_dir / "dimension_ablation.csv", index=False)
    primary = ablations.iloc[0]
    primary_rho = _finite_or_none(primary.oof_spearman_rho)
    primary_low = _finite_or_none(primary.bootstrap_ci95_low)
    primary_high = _finite_or_none(primary.bootstrap_ci95_high)
    nonzero_success_motions = int((joined.success_rate > 0).sum())
    distinct_success_rates = int(joined.success_rate.nunique())
    gate = bool(
        primary_rho is not None
        and primary_low is not None
        and primary_rho > 0
        and primary_low > 0
    )
    calibration = {
        "score_is_calibrated": gate,
        "gate": "OOF Spearman > 0 and paired-bootstrap 95% CI lower bound > 0",
        "oof_spearman_rho": primary_rho,
        "bootstrap_ci95": [primary_low, primary_high],
        "full_motion_count": len(joined),
        "nonzero_success_motion_count": nonzero_success_motions,
        "distinct_success_rate_count": distinct_success_rates,
        "sparse_success_outcome": nonzero_success_motions < 3,
        "outcome": "candidate_v2_calibration_supported" if gate else "retain_v1_uncalibrated_negative_result",
    }
    (analysis_dir / "calibration_decision.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Step 11 formal learning results",
        "",
        f"- Controlled policies: {expected} (4 cohorts × {len(config['training']['policy_seeds'])} seeds)",
        f"- Training budget: {config['training']['max_iterations']} iterations, "
        f"{config['training']['transitions_per_policy']:,} transitions/policy",
        f"- Evaluation: {config['evaluation']['episodes_per_motion_per_seed']} episodes/motion/seed",
        f"- Calibration accepted: `{str(gate).lower()}`",
        f"- Decision: `{calibration['outcome']}`",
        "",
        "## Cohort means across policy seeds",
        "",
        "| Cohort | Conv. iteration | Success | Episode length (s) | MPJPE (m) | W-MPJPE (m) | Fall rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cohort_summary.iterrows():
        report_lines.append(
            f"| {row.cohort} | {row.convergence_iteration_mean:.1f} "
            f"[{row.convergence_iteration_ci95_low:.1f}, {row.convergence_iteration_ci95_high:.1f}] | "
            f"{row.success_rate_mean:.4f} [{row.success_rate_ci95_low:.4f}, {row.success_rate_ci95_high:.4f}] | "
            f"{row.episode_length_s_mean:.4f} [{row.episode_length_s_ci95_low:.4f}, {row.episode_length_s_ci95_high:.4f}] | "
            f"{row.mpjpe_m_mean:.4f} [{row.mpjpe_m_ci95_low:.4f}, {row.mpjpe_m_ci95_high:.4f}] | "
            f"{row.weighted_mpjpe_m_mean:.4f} [{row.weighted_mpjpe_m_ci95_low:.4f}, {row.weighted_mpjpe_m_ci95_high:.4f}] | "
            f"{row.fall_rate_mean:.4f} [{row.fall_rate_ci95_low:.4f}, {row.fall_rate_ci95_high:.4f}] |"
        )
    report_lines.extend(
        [
            "",
            "Each mean has a seed-level t-based 95% CI in `cohort_summary.csv`; "
            "seed-paired cohort effects are in `paired_cohort_comparisons.csv`.",
            "Iteration 301 denotes a policy right-censored as not converged within the 300-iteration budget.",
            "",
            "## Offline score versus Full-policy per-motion outcomes",
            "",
            "| Learning target | Spearman rho | Bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    quality_correlations = correlations[correlations.offline_metric == "quality_score"]
    for _, row in quality_correlations.iterrows():
        report_lines.append(
            f"| {row.learning_target} | {row.spearman_rho:.4f} | "
            f"[{row.bootstrap_ci95_low:.4f}, {row.bootstrap_ci95_high:.4f}] |"
        )
    report_lines.extend(
        [
            "",
            "The v1 score is retained as uncalibrated unless the pre-registered OOF gate passes. "
            "A failed gate is a valid negative result and does not trigger post-hoc weight tuning.",
            f"Only {nonzero_success_motions}/{len(joined)} Full-policy motions had non-zero success, "
            "so success-based calibration evidence is sparse and must not be generalized beyond this cohort/budget.",
            "",
        ]
    )
    (analysis_dir / "STEP11_RESULTS.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "manifest_version": "step11_final_v1", "complete": True,
        "training_job_count": expected, "policy_result_count": expected,
        "all_training_jobs_complete": bool((jobs["status"] == "complete").all()),
        "protocol_config": {
            "path": str(Path(config_path).expanduser().resolve()),
            "sha256": sha256_file(Path(config_path).expanduser().resolve()),
        },
        "score_is_calibrated": gate, "calibration": calibration,
        "files": [],
    }
    critical_files = {
        output / "cohort_selection.csv",
        output / "training_jobs.csv",
        output / "resolved_step11_config.yaml",
        output / "gpu_execution_audit.json",
        *analysis_dir.glob("*"),
        *(Path(path) for path in jobs["result_json"]),
        *(Path(path) for path in jobs["checkpoint"]),
    }
    for path in sorted(critical_files):
        if path.is_file():
            manifest["files"].append(
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    final_path = output / "step11_manifest.json"
    final_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
