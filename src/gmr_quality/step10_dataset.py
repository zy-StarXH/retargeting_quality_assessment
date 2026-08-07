from __future__ import annotations

import contextlib
import csv
import io
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .manifest import git_state, sha256_file
from .mujoco_kinematics import MujocoKinematics


STEP10_STRATA = (
    "standing_walking",
    "running_jumping",
    "dance",
    "complex_dynamic",
    "floor_motion",
)
SELECTION_COLUMNS = (
    "motion_name",
    "source_bvh",
    "motion_category",
    "source_theme",
    "contact_policy",
    "selection_rationale",
)
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


@dataclass(frozen=True)
class Step10SourceSelection:
    motion_name: str
    source_bvh: Path
    motion_category: str
    source_theme: str
    contact_policy: str
    selection_rationale: str


@dataclass(frozen=True)
class PreparedStep10Dataset:
    output_dir: Path
    catalog_csv: Path
    manifest_json: Path
    motion_count: int
    frame_count: int


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Unsafe or empty motion_name: {value!r}")
    return value


def load_step10_selection(path: str | Path) -> tuple[Step10SourceSelection, ...]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in SELECTION_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                "Step-10 source selection is missing columns: " + ", ".join(missing)
            )
        rows = list(reader)
    selections: list[Step10SourceSelection] = []
    for row in rows:
        bvh = Path(row["source_bvh"]).expanduser().resolve()
        if not bvh.is_file() or bvh.suffix.lower() != ".bvh":
            raise ValueError(f"Source BVH does not exist: {bvh}")
        category = row["motion_category"].strip()
        if category not in STEP10_STRATA:
            raise ValueError(f"Unsupported step-10 stratum: {category!r}")
        policy = row["contact_policy"].strip()
        if policy not in {"feet_only", "floor_motion"}:
            raise ValueError(f"Unsupported step-10 contact policy: {policy!r}")
        selections.append(
            Step10SourceSelection(
                motion_name=_safe_name(row["motion_name"].strip()),
                source_bvh=bvh,
                motion_category=category,
                source_theme=row["source_theme"].strip(),
                contact_policy=policy,
                selection_rationale=row["selection_rationale"].strip(),
            )
        )
    names = [row.motion_name for row in selections]
    if len(names) != len(set(names)):
        raise ValueError("Step-10 source selection contains duplicate motion names")
    counts = {
        stratum: sum(row.motion_category == stratum for row in selections)
        for stratum in STEP10_STRATA
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError(
            "Step-10 frozen cohort requires exactly three motions per stratum; "
            f"found {counts}"
        )
    return tuple(selections)


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Dataset output is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"Dataset output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _write_pickle_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _source_fps(path: Path) -> float:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip().startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                if np.isfinite(frame_time) and frame_time > 0:
                    return 1.0 / frame_time
                break
    raise ValueError(f"Could not read positive Frame Time from {path}")


def prepare_step10_dataset(
    project_config: dict[str, Any],
    selection_csv: str | Path,
    output_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> PreparedStep10Dataset:
    """Retarget the frozen 15-motion LAFAN1 cohort without opening a viewer."""
    selection_path = Path(selection_csv).expanduser().resolve()
    selections = load_step10_selection(selection_path)
    destination = Path(output_dir).expanduser().resolve()
    _prepare_output(destination)
    retargeted_dir = destination / "retargeted"
    retargeted_dir.mkdir()
    kinematics = MujocoKinematics.from_project_config(project_config)

    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    import general_motion_retargeting.motion_retarget as motion_retarget_module
    import general_motion_retargeting.utils.lafan1 as lafan1_module

    started = time.perf_counter()
    catalog_rows: list[dict[str, str]] = []
    manifest_inputs: list[dict[str, Any]] = []
    total_frames = 0
    for index, selection in enumerate(selections, start=1):
        if progress:
            progress(
                f"[{index}/{len(selections)}] retargeting "
                f"{selection.motion_name} ({selection.motion_category})"
            )
        sequence_start = time.perf_counter()
        frames, human_height = load_bvh_file(
            str(selection.source_bvh), format="lafan1"
        )
        if len(frames) < 4:
            raise ValueError(f"Source has too few frames: {selection.source_bvh}")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            retargeter = GeneralMotionRetargeting(
                src_human="bvh_lafan1",
                tgt_robot="unitree_g1",
                actual_human_height=float(human_height),
                solver="daqp",
                damping=0.5,
                verbose=False,
                use_velocity_limit=False,
            )
        qpos = np.empty((len(frames), 36), dtype=np.float64)
        for frame_index, human_frame in enumerate(frames):
            qpos[frame_index] = retargeter.retarget(human_frame)
        if not np.isfinite(qpos).all():
            raise ValueError(f"GMR produced non-finite qpos for {selection.motion_name}")
        output_path = retargeted_dir / f"{selection.motion_name}.pkl"
        payload = {
            "fps": 30,
            "root_pos": qpos[:, :3],
            "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": qpos[:, 7:],
            "dof_names": kinematics.info.joint_names,
        }
        _write_pickle_atomic(payload, output_path)
        with output_path.open("rb") as stream:
            written = pickle.load(stream)
        np.testing.assert_array_equal(written["root_pos"], payload["root_pos"])
        np.testing.assert_array_equal(written["root_rot"], payload["root_rot"])
        np.testing.assert_array_equal(written["dof_pos"], payload["dof_pos"])
        source_hash = sha256_file(selection.source_bvh)
        output_hash = sha256_file(output_path)
        note = (
            f"theme={selection.source_theme}; {selection.selection_rationale}; "
            "generated headlessly by the frozen step-10 adapter with all source "
            "frames and use_velocity_limit=false"
        )
        catalog_rows.append(
            {
                "motion_name": selection.motion_name,
                "pkl_path": str(output_path),
                "source_bvh": str(selection.source_bvh),
                "motion_category": selection.motion_category,
                "contact_policy": selection.contact_policy,
                "source_pair_status": "verified_generated_exact",
                "pkl_sha256": output_hash,
                "source_bvh_sha256": source_hash,
                "frame_alignment": "exact",
                "notes": note,
            }
        )
        elapsed = time.perf_counter() - sequence_start
        manifest_inputs.append(
            {
                "motion_name": selection.motion_name,
                "motion_category": selection.motion_category,
                "source_theme": selection.source_theme,
                "contact_policy": selection.contact_policy,
                "source_bvh": str(selection.source_bvh),
                "source_bvh_sha256": source_hash,
                "source_frame_count": len(frames),
                "source_fps": _source_fps(selection.source_bvh),
                "actual_human_height_m": float(human_height),
                "output_pkl": str(output_path),
                "output_pkl_sha256": output_hash,
                "output_frame_count": int(qpos.shape[0]),
                "qpos_width": int(qpos.shape[1]),
                "frame_alignment": "exact",
                "serialization_roundtrip_exact": True,
                "retarget_seconds": elapsed,
            }
        )
        total_frames += len(frames)
        if progress:
            progress(
                f"[{index}/{len(selections)}] completed {selection.motion_name}: "
                f"frames={len(frames)}, seconds={elapsed:.2f}"
            )

    catalog_path = destination / "step10_motion_catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(catalog_rows)

    module_path = Path(__file__).resolve()
    motion_retarget_path = Path(motion_retarget_module.__file__).resolve()
    lafan1_path = Path(lafan1_module.__file__).resolve()
    manifest = {
        "schema_version": 1,
        "complete": True,
        "cohort": {
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "strata": list(STEP10_STRATA),
            "motions_per_stratum": 3,
            "motion_count": len(selections),
            "total_frames": total_frames,
        },
        "generator": {
            "module_path": str(module_path),
            "module_sha256": sha256_file(module_path),
            "gmr": git_state(project_config["paths"]["gmr_root"]),
            "motion_retarget_path": str(motion_retarget_path),
            "motion_retarget_sha256": sha256_file(motion_retarget_path),
            "lafan1_loader_path": str(lafan1_path),
            "lafan1_loader_sha256": sha256_file(lafan1_path),
            "ik_config_path": project_config["paths"]["gmr_lafan1_g1_ik_config"],
            "ik_config_sha256": sha256_file(
                project_config["paths"]["gmr_lafan1_g1_ik_config"]
            ),
            "robot_xml_path": project_config["paths"]["robot_xml"],
            "robot_xml_sha256": sha256_file(project_config["paths"]["robot_xml"]),
            "solver": "daqp",
            "damping": 0.5,
            "use_velocity_limit": False,
            "viewer_used": False,
        },
        "catalog": {
            "path": str(catalog_path),
            "sha256": sha256_file(catalog_path),
        },
        "inputs": manifest_inputs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = destination / "retarget_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    if progress:
        progress(
            f"dataset complete: motions={len(selections)}, frames={total_frames}, "
            f"seconds={manifest['elapsed_seconds']:.2f}"
        )
    return PreparedStep10Dataset(
        output_dir=destination,
        catalog_csv=catalog_path,
        manifest_json=manifest_path,
        motion_count=len(selections),
        frame_count=total_frames,
    )
