"""Load persisted reference-model sets for inspection and future classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsgr.reference_models.provenance import (
    BranchSignature,
    feature_ids_hash,
    feature_mask_hash,
    feature_schema_hash,
)


def branch_directory_name(branch_name: str) -> str:
    return branch_name.replace(".", "__")


@dataclass(slots=True)
class ReferenceGestureModel:
    """One gesture prototype and uncertainty description for one feature branch."""

    model_set_dir: Path
    gesture_id: str
    branch_name: str
    signature: BranchSignature
    feature_ids: tuple[str, ...]
    active_mask: np.ndarray
    active_indices: np.ndarray
    mask_sha256: str
    prototype_mean: np.ndarray
    prototype_median: np.ndarray
    standard_deviation: np.ndarray
    q1: np.ndarray
    q3: np.ndarray
    covariance_empirical_active: np.ndarray
    covariance_regularized_active: np.ndarray
    diagonal_variance_active: np.ndarray
    person_ids: tuple[str, ...]
    person_means: np.ndarray
    person_medians: np.ndarray
    metadata: dict[str, Any]


def load_reference_gesture_model(
    model_set_dir: str | Path,
    *,
    gesture_id: str,
    branch_name: str,
) -> ReferenceGestureModel:
    """Load and cross-check one persisted model artifact."""
    root = Path(model_set_dir)
    branch_dir = root / "branches" / branch_directory_name(branch_name)
    gesture_dir = branch_dir / "gestures" / gesture_id
    metadata_path = gesture_dir / "model.json"
    arrays_path = gesture_dir / "model_arrays.npz"
    mask_path = branch_dir / "feature_mask.json"
    schema_path = branch_dir / "feature_schema_snapshot.json"
    if not all(path.is_file() for path in (metadata_path, arrays_path, mask_path, schema_path)):
        raise FileNotFoundError(
            f"Incomplete model artifact for gesture={gesture_id!r}, branch={branch_name!r}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    feature_ids = tuple(str(value) for value in arrays["feature_ids"].tolist())
    active_mask = np.asarray(arrays["active_mask"], dtype=bool)
    active_indices = np.asarray(arrays["active_indices"], dtype=np.int64)
    if active_mask.shape != (len(feature_ids),):
        raise ValueError("Stored active feature mask has an invalid shape.")
    expected_indices = np.flatnonzero(active_mask)
    if not np.array_equal(active_indices, expected_indices):
        raise ValueError("Stored active feature indices do not match the Boolean mask.")

    signature = BranchSignature.from_dict(metadata["branch_signature"])
    if signature.feature_count != len(feature_ids):
        raise ValueError("Stored feature count does not match the model arrays.")
    if signature.feature_ids_sha256 != feature_ids_hash(feature_ids):
        raise ValueError("Stored feature identifiers do not match the branch signature.")
    if signature.feature_schema_sha256 != feature_schema_hash(schema_payload):
        raise ValueError("Stored feature-schema snapshot does not match the branch signature.")

    stored_mask_hash = str(metadata["feature_mask_sha256"])
    if stored_mask_hash != str(mask_payload["feature_mask_sha256"]):
        raise ValueError("Gesture model and branch feature-mask hashes differ.")
    if stored_mask_hash != feature_mask_hash(feature_ids, active_mask.tolist()):
        raise ValueError("Stored feature-mask hash does not match the model arrays.")
    if str(metadata["gesture_id"]) != gesture_id or str(metadata["branch_name"]) != branch_name:
        raise ValueError("Requested gesture/branch does not match persisted model metadata.")

    expected_vector_shape = (len(feature_ids),)
    for array_name in ("prototype_mean", "prototype_median", "standard_deviation", "q1", "q3"):
        if np.asarray(arrays[array_name]).shape != expected_vector_shape:
            raise ValueError(f"Stored {array_name} has an invalid shape.")
    active_count = int(active_mask.sum())
    expected_covariance_shape = (active_count, active_count)
    for array_name in ("covariance_empirical_active", "covariance_regularized_active"):
        if np.asarray(arrays[array_name]).shape != expected_covariance_shape:
            raise ValueError(f"Stored {array_name} has an invalid shape.")
    if np.asarray(arrays["diagonal_variance_active"]).shape != (active_count,):
        raise ValueError("Stored diagonal active variance has an invalid shape.")

    return ReferenceGestureModel(
        model_set_dir=root,
        gesture_id=gesture_id,
        branch_name=branch_name,
        signature=signature,
        feature_ids=feature_ids,
        active_mask=active_mask,
        active_indices=active_indices,
        mask_sha256=str(metadata["feature_mask_sha256"]),
        prototype_mean=np.asarray(arrays["prototype_mean"], dtype=np.float64),
        prototype_median=np.asarray(arrays["prototype_median"], dtype=np.float64),
        standard_deviation=np.asarray(arrays["standard_deviation"], dtype=np.float64),
        q1=np.asarray(arrays["q1"], dtype=np.float64),
        q3=np.asarray(arrays["q3"], dtype=np.float64),
        covariance_empirical_active=np.asarray(
            arrays["covariance_empirical_active"], dtype=np.float64
        ),
        covariance_regularized_active=np.asarray(
            arrays["covariance_regularized_active"], dtype=np.float64
        ),
        diagonal_variance_active=np.asarray(
            arrays["diagonal_variance_active"], dtype=np.float64
        ),
        person_ids=tuple(str(value) for value in arrays.get("person_ids", np.asarray([], dtype=str)).tolist()),
        person_means=np.asarray(arrays.get("person_means", np.empty((0, len(feature_ids)))), dtype=np.float64),
        person_medians=np.asarray(arrays.get("person_medians", np.empty((0, len(feature_ids)))), dtype=np.float64),
        metadata=metadata,
    )
