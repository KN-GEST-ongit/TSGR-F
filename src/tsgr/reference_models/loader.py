"""Load complete feature branches and provenance from processed TSGR-F runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsgr.reference_models.provenance import BranchSignature, signature_from_run
from tsgr.utils.run_paths import latest_run_directory


@dataclass(slots=True)
class LoadedFeatureSession:
    """Feature matrix and metadata for one gesture recording session."""

    run_dir: Path
    branch_name: str
    schema_version: str
    feature_ids: tuple[str, ...]
    schema_payload: dict[str, Any]
    frame_indices: np.ndarray
    relative_times_s: np.ndarray
    values: np.ndarray
    signature: BranchSignature
    frame_status: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])


def resolve_feature_run(path: str | Path) -> Path:
    """Resolve a run directory from a run, session, feature directory, or NPZ path."""
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "feature_arrays.npz":
        return candidate.parent.parent
    if (candidate / "features" / "feature_arrays.npz").is_file():
        return candidate
    if (candidate / "feature_arrays.npz").is_file():
        return candidate.parent
    return latest_run_directory(candidate)


def _load_schema(run_dir: Path) -> dict[str, Any]:
    schema_path = run_dir / "features" / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema does not exist: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError(f"Invalid feature schema JSON: {schema_path}")
    return schema


def load_feature_session(path: str | Path, branch_name: str) -> LoadedFeatureSession:
    """Load complete finite feature rows for one requested branch."""
    run_dir = resolve_feature_run(path)
    npz_path = run_dir / "features" / "feature_arrays.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Feature arrays do not exist: {npz_path}")
    schema = _load_schema(run_dir)
    schema_version = str(schema.get("schema_version", ""))
    feature_ids = tuple(
        str(item["feature_id"])
        for item in schema.get("features", [])
        if isinstance(item, dict) and "feature_id" in item
    )
    with np.load(npz_path, allow_pickle=False) as archive:
        branch_names = [str(value) for value in archive["branch_names"].tolist()]
        if branch_name not in branch_names:
            raise ValueError(
                f"Branch {branch_name!r} is not present in {npz_path}; available: {branch_names}."
            )
        branch_index = branch_names.index(branch_name)
        values = np.asarray(archive["values"][:, branch_index, :], dtype=np.float64)
        frame_indices = np.asarray(archive["frame_index"], dtype=np.int64)
        relative_times = np.asarray(archive["relative_time_s"], dtype=np.float64)
        status = np.asarray(archive["status"]).astype(str)
        npz_feature_ids = tuple(str(value) for value in archive["feature_ids"].tolist())
    if feature_ids != npz_feature_ids:
        raise ValueError(
            f"Feature identifiers in schema and NPZ differ for run {run_dir}."
        )
    if values.shape[1] != len(feature_ids):
        raise ValueError(
            f"Feature dimension mismatch in {npz_path}: {values.shape[1]} vs {len(feature_ids)}."
        )
    complete = np.isfinite(values).all(axis=1)
    values = values[complete]
    frame_indices = frame_indices[complete]
    relative_times = relative_times[complete]
    status = status[complete]
    if values.size == 0:
        raise ValueError(
            f"Run {run_dir} has no complete feature vectors for branch {branch_name}."
        )
    signature = signature_from_run(
        run_dir,
        branch_name=branch_name,
        schema_version=schema_version,
        feature_ids=list(feature_ids),
        schema_payload=schema,
    )
    return LoadedFeatureSession(
        run_dir=run_dir,
        branch_name=branch_name,
        schema_version=schema_version,
        feature_ids=feature_ids,
        schema_payload=schema,
        frame_indices=frame_indices,
        relative_times_s=relative_times,
        values=values,
        signature=signature,
        frame_status=status,
    )
