"""Training-data access helpers required by the final TSGR-F model builder."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsgr.evaluation.io_utils import read_csv_rows
from tsgr.reference_models.loader import load_feature_session
from tsgr.reference_models.manifest import locate_processed_run, read_reference_manifest
from tsgr.utils.natural_sort import natural_key


@dataclass(slots=True)
class TrainingSamples:
    values: np.ndarray
    labels: np.ndarray
    metadata: list[dict[str, Any]]
    gesture_ids: tuple[str, ...]
    gesture_index: np.ndarray


def _training_rows_by_cell(fold_dir: Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(fold_dir / "training_images.csv"):
        key = (
            str(row["gesture_id"]).upper(),
            str(row["public_subject_id"]).upper(),
            str(row["background"]).upper(),
        )
        grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: natural_key(str(row.get("filename") or row.get("relative_path") or row.get("image_id") or "")))
    return grouped


def load_fold_training_samples(
    fold_dir: Path,
    *,
    branch_name: str,
    dataset_root: Path,
    gesture_ids: tuple[str, ...],
) -> TrainingSamples:
    """Load complete TRAIN feature vectors in the fold's public image order."""
    grouped = _training_rows_by_cell(fold_dir)
    exact_runs: dict[tuple[str, str, str], Path] = {}
    manifest_path = fold_dir / "reference_manifest.csv"
    if manifest_path.is_file():
        for entry in read_reference_manifest(manifest_path):
            if entry.include and entry.run_path.is_dir():
                exact_runs[(entry.gesture_id.upper(), entry.person_id.upper(), entry.session_id.upper())] = entry.run_path

    values_parts: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, Any]] = []
    feature_ids: tuple[str, ...] | None = None

    for (gesture_id, subject_id, background), rows in sorted(grouped.items()):
        cell_dir = dataset_root / "training" / subject_id / background / gesture_id
        run = exact_runs.get((gesture_id, subject_id, background)) or locate_processed_run(cell_dir)
        if run is None:
            raise FileNotFoundError(f"No processed training run found for model building: {cell_dir}")
        loaded = load_feature_session(run, branch_name)
        if feature_ids is None:
            feature_ids = loaded.feature_ids
        elif loaded.feature_ids != feature_ids:
            raise ValueError(f"Feature identifiers differ inside fold {fold_dir}.")

        for local_row, frame_index in enumerate(loaded.frame_indices.tolist()):
            frame_index = int(frame_index)
            if frame_index < 0 or frame_index >= len(rows):
                raise ValueError(
                    f"Training feature frame index {frame_index} is outside the public image list for "
                    f"{subject_id}/{background}/{gesture_id}."
                )
            row = rows[frame_index]
            values_parts.append(np.asarray(loaded.values[local_row], dtype=np.float64)[None, :])
            labels.append(gesture_id)
            metadata.append(
                {
                    "image_id": str(row.get("image_id") or ""),
                    "public_subject_id": subject_id,
                    "background": background,
                    "gesture_id": gesture_id,
                    "filename": str(row.get("filename") or ""),
                    "relative_path": str(row.get("relative_path") or ""),
                    "frame_index": frame_index,
                }
            )

    if not values_parts:
        raise ValueError(f"No complete training vectors found for fold {fold_dir}.")
    values = np.vstack(values_parts)
    label_array = np.asarray(labels, dtype=object)
    index_by_gesture = {gesture: index for index, gesture in enumerate(gesture_ids)}
    try:
        gesture_index = np.asarray([index_by_gesture[str(label)] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Training label is missing from the model space: {error}") from error
    return TrainingSamples(
        values=values,
        labels=label_array,
        metadata=metadata,
        gesture_ids=gesture_ids,
        gesture_index=gesture_index,
    )


def fold_run_map(fold_dir: Path, dataset_root: Path) -> dict[tuple[str, str, str], Path]:
    """Resolve processed TRAIN runs for every gesture/subject/background cell."""
    result: dict[tuple[str, str, str], Path] = {}
    manifest_path = fold_dir / "reference_manifest.csv"
    if manifest_path.is_file():
        for entry in read_reference_manifest(manifest_path):
            if entry.include and entry.run_path.is_dir():
                result[(entry.gesture_id.upper(), entry.person_id.upper(), entry.session_id.upper())] = entry.run_path
    for row in read_csv_rows(fold_dir / "training_images.csv"):
        key = (
            str(row.get("gesture_id", "")).upper(),
            str(row.get("public_subject_id", "")).upper(),
            str(row.get("background", "")).upper(),
        )
        if key in result:
            continue
        cell_dir = dataset_root / "training" / key[1] / key[2] / key[0]
        run = locate_processed_run(cell_dir)
        if run is not None:
            result[key] = run
    return result


class LandmarkArchiveCache:
    """Cache persisted detector and normalization arrays used by processed runs."""

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def _load(self, run_dir: Path) -> dict[str, np.ndarray]:
        run = Path(run_dir)
        if run in self._cache:
            return self._cache[run]
        path = run / "normalization_arrays.npz"
        if not path.is_file():
            payload = {"frame_index": np.asarray([], dtype=np.int64)}
            self._cache[run] = payload
            return payload
        with np.load(path, allow_pickle=False) as archive:
            payload = {
                key: np.asarray(archive[key])
                for key in (
                    "frame_index",
                    "raw_image_landmarks",
                    "raw_world_landmarks",
                    "normalized_wrist_middle_mcp",
                )
                if key in archive
            }
        self._cache[run] = payload
        return payload

    def get(self, run_dir: Path, frame_index: int, key: str) -> np.ndarray | None:
        payload = self._load(run_dir)
        indices = np.asarray(payload.get("frame_index", []), dtype=np.int64)
        values = payload.get(key)
        if values is None or indices.size == 0:
            return None
        matches = np.flatnonzero(indices == int(frame_index))
        if matches.size == 0:
            return None
        candidate = np.asarray(values[int(matches[0])], dtype=np.float64)
        if candidate.shape != (21, 3) or not np.isfinite(candidate).all():
            return None
        return candidate
