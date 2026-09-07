"""Fold-local acceptance calibration from training-only TSGRF feature caches.

This module intentionally separates prototype construction from acceptance-radius
calibration. Reference prototypes may remain robust to automatically detected
outliers, while the acceptance calibration can use every complete training
feature vector from the fold. This is useful when manually reviewed samples are
known to represent valid anatomical/execution variability even if they were
flagged by a conservative outlier detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tsgr.reference_models.loader import load_feature_session
from tsgr.evaluation.io_utils import read_csv_rows
from tsgr.reference_models.manifest import locate_processed_run, read_reference_manifest

CALIBRATION_SOURCES = ("prototype_person_means", "all_training_complete")


@dataclass(frozen=True, slots=True)
class GestureCalibration:
    gesture_id: str
    sample_count: int
    standard_deviation: np.ndarray
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class FoldAcceptanceCalibration:
    branch_name: str
    source: str
    feature_ids: tuple[str, ...]
    by_gesture: dict[str, GestureCalibration]


@lru_cache(maxsize=128)
def _load_all_training_complete_cached(
    fold_dir_text: str,
    branch_name: str,
    dataset_root_text: str,
) -> FoldAcceptanceCalibration:
    fold_dir = Path(fold_dir_text)
    dataset_root = Path(dataset_root_text) if dataset_root_text else None

    run_specs: list[tuple[str, Path]] = []
    manifest_path = fold_dir / "reference_manifest.csv"
    entries = [entry for entry in read_reference_manifest(manifest_path) if entry.include] if manifest_path.is_file() else []
    # Prefer the exact processed runs recorded in the fold manifest.
    # Reconstruct them from the portable dataset hierarchy when those paths are unavailable.
    if entries and all(Path(entry.run_path).is_dir() for entry in entries):
        run_specs = [(entry.gesture_id, Path(entry.run_path)) for entry in entries]
    else:
        training_rows = read_csv_rows(fold_dir / "training_images.csv")
        if dataset_root is None or not training_rows:
            raise ValueError(
                f"Fold {fold_dir} has no usable exact reference runs and cannot reconstruct them from the public dataset."
            )
        cells = sorted({
            (str(row["gesture_id"]), str(row["public_subject_id"]), str(row["background"]))
            for row in training_rows
        })
        for gesture_id, subject_id, background in cells:
            cell_dir = dataset_root / "training" / subject_id / background / gesture_id
            run = locate_processed_run(cell_dir)
            if run is None:
                raise FileNotFoundError(
                    f"No processed training run for acceptance calibration: {cell_dir}"
                )
            run_specs.append((gesture_id, run))

    grouped: dict[str, list[np.ndarray]] = {}
    feature_ids: tuple[str, ...] | None = None
    for gesture_id, run_path in run_specs:
        loaded = load_feature_session(run_path, branch_name)
        if feature_ids is None:
            feature_ids = loaded.feature_ids
        elif loaded.feature_ids != feature_ids:
            raise ValueError(
                f"Acceptance-calibration feature identifiers differ inside fold {fold_dir}."
            )
        grouped.setdefault(gesture_id, []).append(np.asarray(loaded.values, dtype=np.float64))

    if feature_ids is None:
        raise ValueError(f"Fold {fold_dir} contains no calibration features.")

    by_gesture: dict[str, GestureCalibration] = {}
    for gesture_id, matrices in sorted(grouped.items()):
        matrix = np.vstack(matrices)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
            raise ValueError(
                f"Invalid all-training calibration matrix for {gesture_id} in fold {fold_dir}."
            )
        by_gesture[gesture_id] = GestureCalibration(
            gesture_id=gesture_id,
            sample_count=int(matrix.shape[0]),
            standard_deviation=np.std(matrix, axis=0),
            values=matrix,
        )

    return FoldAcceptanceCalibration(
        branch_name=branch_name,
        source="all_training_complete",
        feature_ids=feature_ids,
        by_gesture=by_gesture,
    )


def load_fold_acceptance_calibration(
    fold_dir: str | Path,
    *,
    branch_name: str,
    source: str,
    dataset_root: str | Path | None = None,
) -> FoldAcceptanceCalibration | None:
    """Load fold-local calibration data.

    ``prototype_person_means`` returns ``None`` because its uncertainty
    is already persisted in the reference models. ``all_training_complete``
    reads every finite feature vector from the fold training manifest, including
    vectors rejected only by the automatic outlier detector.
    """
    source = str(source).strip().lower()
    if source not in CALIBRATION_SOURCES:
        raise ValueError(f"Unsupported acceptance calibration source: {source}")
    if source == "prototype_person_means":
        return None
    return _load_all_training_complete_cached(
        str(Path(fold_dir).resolve()),
        str(branch_name),
        "" if dataset_root is None else str(Path(dataset_root).resolve()),
    )
