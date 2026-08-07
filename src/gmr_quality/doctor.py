from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np


DEPENDENCIES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("mujoco", "mujoco"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("pytest", "pytest"),
)

PKL_SHALLOW_KEYS = ("fps", "root_pos", "root_rot", "dof_pos")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    details: str


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in ("pass", "warn", "fail")
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": self.counts,
            "checks": [asdict(check) for check in self.checks],
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


def _check_dependencies() -> Iterable[Check]:
    for module_name, distribution_name in DEPENDENCIES:
        if importlib.util.find_spec(module_name) is None:
            yield Check(
                f"dependency:{module_name}",
                "fail",
                f"missing Python module (distribution: {distribution_name})",
            )
            continue
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = "version unavailable"
        yield Check(f"dependency:{module_name}", "pass", version)


def _run_git(gmr_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(gmr_root), *args),
        check=False,
        capture_output=True,
        text=True,
    )


def _check_gmr(config: dict[str, Any]) -> Iterable[Check]:
    gmr_root = Path(config["paths"]["gmr_root"])
    bvh_script = Path(config["paths"]["gmr_bvh_script"])

    if not gmr_root.is_dir():
        yield Check("gmr_root", "fail", str(gmr_root))
        return
    yield Check("gmr_root", "pass", str(gmr_root))

    revision = _run_git(gmr_root, "rev-parse", "HEAD")
    if revision.returncode:
        yield Check("gmr_git_revision", "fail", revision.stderr.strip())
    else:
        yield Check("gmr_git_revision", "pass", revision.stdout.strip())

    worktree = _run_git(gmr_root, "status", "--porcelain")
    if worktree.returncode:
        yield Check("gmr_worktree", "fail", worktree.stderr.strip())
    elif worktree.stdout.strip():
        changed = len(worktree.stdout.strip().splitlines())
        yield Check(
            "gmr_worktree",
            "warn",
            f"{changed} existing user change(s); this project must not overwrite them",
        )
    else:
        yield Check("gmr_worktree", "pass", "clean")

    yield Check(
        "gmr_bvh_script",
        "pass" if bvh_script.is_file() else "fail",
        str(bvh_script),
    )
    loader_path = Path(config["paths"]["gmr_lafan1_loader"])
    yield Check(
        "gmr_lafan1_loader",
        "pass" if loader_path.is_file() else "fail",
        str(loader_path),
    )
    ik_path = Path(config["paths"]["gmr_lafan1_g1_ik_config"])
    if not ik_path.is_file():
        yield Check("gmr_lafan1_g1_ik_config", "fail", str(ik_path))
    else:
        actual_hash = _sha256(ik_path)
        expected_hash = str(config["fidelity"]["expected_ik_config_sha256"])
        yield Check(
            "gmr_lafan1_g1_ik_config",
            "pass" if actual_hash == expected_hash else "fail",
            f"{ik_path}; sha256={actual_hash}",
        )
    evidence_path = Path(config["paths"]["pair_replay_evidence"])
    yield Check(
        "pair_replay_evidence",
        "pass" if evidence_path.is_file() else "fail",
        (
            f"{evidence_path}; sha256={_sha256(evidence_path)}"
            if evidence_path.is_file()
            else str(evidence_path)
        ),
    )


def _object_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def _check_model(config: dict[str, Any]) -> Iterable[Check]:
    xml_path = Path(config["paths"]["robot_xml"])
    if not xml_path.is_file():
        yield Check("robot_xml", "fail", str(xml_path))
        return
    yield Check("robot_xml", "pass", str(xml_path))

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # MuJoCo exposes several parser exception types.
        yield Check("mujoco_model_load", "fail", f"{type(exc).__name__}: {exc}")
        return

    doctor = config["doctor"]
    expected_nq = int(doctor["expected_nq"])
    expected_nv = int(doctor["expected_nv"])
    dimensions_ok = model.nq == expected_nq and model.nv == expected_nv
    yield Check(
        "mujoco_dimensions",
        "pass" if dimensions_ok else "fail",
        f"nq={model.nq} (expected {expected_nq}), nv={model.nv} (expected {expected_nv})",
    )

    hinge_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    hinge_ids.sort(key=lambda joint_id: int(model.jnt_qposadr[joint_id]))
    expected_dof_count = int(doctor["expected_dof_count"])
    yield Check(
        "hinge_dof_count",
        "pass" if len(hinge_ids) == expected_dof_count else "fail",
        f"found {len(hinge_ids)}, expected {expected_dof_count}",
    )

    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in hinge_ids]
    root_qpos_width = int(doctor["root_qpos_width"])
    expected_addresses = list(range(root_qpos_width, root_qpos_width + expected_dof_count))
    yield Check(
        "hinge_qpos_order",
        "pass" if qpos_addresses == expected_addresses else "fail",
        f"addresses={qpos_addresses}",
    )
    joint_names = [
        _object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in hinge_ids
    ]
    yield Check("hinge_joint_names", "pass", ",".join(joint_names))

    floor_name = config["robot"]["floor_geom_name"]
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, floor_name
    )
    yield Check(
        "floor_geom",
        "pass" if floor_id >= 0 else "fail",
        f"name={floor_name}, id={floor_id}",
    )

    max_radius = float(doctor["sole_sphere_max_radius_m"])
    expected_sole_count = int(doctor["expected_sole_geom_count_per_foot"])
    for side in ("left", "right"):
        body_name = config["robot"][f"{side}_foot_body"]
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            yield Check(f"{side}_foot_body", "fail", f"missing body: {body_name}")
            continue
        sole_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and float(model.geom_size[geom_id, 0]) <= max_radius
        ]
        radii = [float(model.geom_size[geom_id, 0]) for geom_id in sole_geoms]
        yield Check(
            f"{side}_sole_geoms",
            "pass" if len(sole_geoms) == expected_sole_count else "fail",
            f"body={body_name}, ids={sole_geoms}, radii_m={radii}",
        )


def _read_bvh_metadata(path: Path) -> tuple[int | None, float | None]:
    frame_count = None
    frame_time = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("Frames:"):
                frame_count = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":", 1)[1].strip())
                break
    return frame_count, frame_time


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _check_catalog(config: dict[str, Any], catalog_path: Path) -> Iterable[Check]:
    if not catalog_path.is_file():
        yield Check("motion_catalog", "fail", str(catalog_path))
        return
    yield Check("motion_catalog", "pass", str(catalog_path))

    with catalog_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        yield Check("motion_catalog_rows", "fail", "catalog is empty")
        return
    yield Check("motion_catalog_rows", "pass", str(len(rows)))

    allowed_delta = int(config["doctor"]["allowed_source_frame_delta"])
    verified_status = str(config["fidelity"]["verified_pair_status"])
    for row in rows:
        motion_name = row["motion_name"]
        pkl_path = Path(row["pkl_path"])
        if not pkl_path.is_file():
            yield Check(f"motion:{motion_name}:pkl", "fail", str(pkl_path))
            continue

        # This is intentionally a shallow inspection. Strict validation belongs to step 2.
        try:
            with pkl_path.open("rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            yield Check(
                f"motion:{motion_name}:pkl",
                "fail",
                f"{type(exc).__name__}: {exc}",
            )
            continue

        if not isinstance(payload, dict):
            yield Check(f"motion:{motion_name}:pkl", "fail", "payload is not a mapping")
            continue
        missing_keys = [key for key in PKL_SHALLOW_KEYS if key not in payload]
        if missing_keys:
            yield Check(
                f"motion:{motion_name}:pkl",
                "fail",
                f"missing shallow keys: {missing_keys}",
            )
            continue

        shapes = {key: _shape_of(payload[key]) for key in PKL_SHALLOW_KEYS}
        yield Check(
            f"motion:{motion_name}:pkl",
            "pass",
            f"keys={sorted(payload)}, shapes={shapes}",
        )

        pkl_frames = len(np.asarray(payload["root_pos"]))
        source_bvh = row.get("source_bvh", "").strip()
        pair_status = row.get("source_pair_status", "").strip()
        if not source_bvh:
            yield Check(
                f"motion:{motion_name}:source_pair",
                "warn",
                f"{pair_status or 'unpaired'}; fidelity must remain disabled",
            )
            continue

        bvh_path = Path(source_bvh)
        if not bvh_path.is_file():
            yield Check(f"motion:{motion_name}:source_bvh", "fail", str(bvh_path))
            continue
        bvh_frames, frame_time = _read_bvh_metadata(bvh_path)
        if bvh_frames is None:
            yield Check(
                f"motion:{motion_name}:source_frames",
                "fail",
                f"could not read Frames from {bvh_path}",
            )
            continue
        delta = bvh_frames - pkl_frames
        alignment = row.get("frame_alignment", "").strip()
        if pair_status == verified_status:
            expected_pkl_hash = row.get("pkl_sha256", "").strip()
            expected_bvh_hash = row.get("source_bvh_sha256", "").strip()
            actual_pkl_hash = _sha256(pkl_path)
            actual_bvh_hash = _sha256(bvh_path)
            hashes_match = (
                actual_pkl_hash == expected_pkl_hash
                and actual_bvh_hash == expected_bvh_hash
            )
            yield Check(
                f"motion:{motion_name}:source_hashes",
                "pass" if hashes_match else "fail",
                f"pkl={actual_pkl_hash}, bvh={actual_bvh_hash}",
            )
            expected_delta = {
                "exact": 0,
                "source_prefix_drop_last": 1,
            }.get(alignment)
            status = "pass" if expected_delta == delta else "fail"
        elif delta == 0:
            status = "pass"
        elif abs(delta) <= allowed_delta:
            status = "warn"
        else:
            status = "fail"
        yield Check(
            f"motion:{motion_name}:source_frames",
            status,
            f"bvh={bvh_frames}, pkl={pkl_frames}, delta={delta}, "
            f"alignment={alignment or 'unspecified'}, frame_time_s={frame_time}",
        )


def run_doctor(config: dict[str, Any], catalog_path: str | Path | None = None) -> DoctorReport:
    checks: list[Check] = []
    expected_python = str(config["doctor"]["expected_python_major_minor"])
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(
        Check(
            "python_version",
            "pass" if actual_python == expected_python else "warn",
            f"actual={actual_python}, expected={expected_python}",
        )
    )
    checks.extend(_check_dependencies())
    checks.extend(_check_gmr(config))
    checks.extend(_check_model(config))

    resolved_catalog = Path(
        catalog_path if catalog_path is not None else config["paths"]["motion_catalog"]
    ).expanduser().resolve()
    checks.extend(_check_catalog(config, resolved_catalog))
    return DoctorReport(tuple(checks))


def format_human_report(report: DoctorReport) -> str:
    lines = [
        f"GMR quality environment doctor: {'OK' if report.ok else 'FAILED'}",
        "",
    ]
    for check in report.checks:
        lines.append(f"[{check.status.upper():4}] {check.name}: {check.details}")
    counts = report.counts
    lines.extend(
        [
            "",
            f"Summary: {counts['pass']} passed, {counts['warn']} warnings, {counts['fail']} failed",
            "No formal motion-quality metrics were computed.",
        ]
    )
    return "\n".join(lines)
