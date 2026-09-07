"""Build balanced versioned reference gesture models from processed sessions."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tsgr import __version__
from tsgr.reference_models.aggregation import (
    GestureStatistics,
    SessionStatistics,
    aggregate_gesture,
    summarize_session,
)
from tsgr.reference_models.loader import LoadedFeatureSession, load_feature_session
from tsgr.reference_models.manifest import ReferenceManifestEntry, write_reference_manifest
from tsgr.reference_models.masking import FeatureMaskResult, build_feature_mask
from tsgr.reference_models.outliers import SessionOutlierResult, detect_session_outliers
from tsgr.reference_models.provenance import BranchSignature
from tsgr.reference_models.repository import branch_directory_name
from tsgr.utils.serialization import environment_report, write_json


@dataclass(slots=True)
class PreparedSession:
    entry: ReferenceManifestEntry
    loaded: LoadedFeatureSession
    outliers: SessionOutlierResult
    statistics: SessionStatistics


@dataclass(slots=True)
class ReferenceModelBuildResult:
    output_dir: Path
    branches: tuple[str, ...]
    gestures: tuple[str, ...]
    model_count: int


def _timestamp_id() -> str:
    return dt.datetime.now().strftime("model_set_%Y%m%d_%H%M%S_%f")


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_branch_signatures(sessions: list[LoadedFeatureSession]) -> BranchSignature:
    if not sessions:
        raise ValueError("No feature sessions were loaded for a reference-model branch.")
    expected = sessions[0].signature
    failures: list[str] = []
    for session in sessions[1:]:
        issues = expected.compatibility_issues(session.signature)
        if issues:
            failures.append(f"{session.run_dir}: " + "; ".join(issues))
    if failures:
        raise ValueError(
            "Incompatible runs cannot be mixed into one reference-model branch:\n"
            + "\n".join(failures)
        )
    return expected


def _prepare_session(
    entry: ReferenceManifestEntry,
    loaded: LoadedFeatureSession,
    outlier_config: dict[str, Any],
) -> PreparedSession:
    result = detect_session_outliers(
        loaded.values,
        feature_z_threshold=float(outlier_config["feature_z_threshold"]),
        robust_rms_threshold=float(outlier_config["robust_rms_threshold"]),
        maximum_extreme_fraction=float(outlier_config["maximum_extreme_fraction"]),
        scale_floor=float(outlier_config["scale_floor"]),
        maximum_reported_features=int(outlier_config["maximum_reported_features"]),
    )
    accepted = loaded.values[result.accepted_mask]
    minimum_frames = int(outlier_config["minimum_accepted_frames_per_session"])
    if accepted.shape[0] < minimum_frames:
        raise ValueError(
            f"Session {entry.gesture_id}/{entry.person_id}/{entry.session_id} retains "
            f"{accepted.shape[0]} frames after outlier detection; minimum is {minimum_frames}."
        )
    statistics = summarize_session(
        gesture_id=entry.gesture_id,
        person_id=entry.person_id,
        session_id=entry.session_id,
        accepted_values=accepted,
        total_frame_count=loaded.frame_count,
    )
    return PreparedSession(entry=entry, loaded=loaded, outliers=result, statistics=statistics)


def _write_long_feature_csv(
    path: Path,
    rows: list[tuple[dict[str, Any], dict[str, np.ndarray]]],
    feature_ids: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    statistic_names = tuple(rows[0][1]) if rows else ()
    metadata_names: list[str] = []
    for metadata, _ in rows:
        for key in metadata:
            if key not in metadata_names:
                metadata_names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*metadata_names, "feature_index", "feature_id", *statistic_names],
        )
        writer.writeheader()
        for metadata, statistics in rows:
            for index, feature_id in enumerate(feature_ids):
                writer.writerow(
                    {
                        **metadata,
                        "feature_index": index,
                        "feature_id": feature_id,
                        **{name: f"{float(values[index]):.12g}" for name, values in statistics.items()},
                    }
                )


def _write_session_outputs(
    branch_dir: Path,
    prepared: list[PreparedSession],
    feature_ids: tuple[str, ...],
    selected_session_keys: set[tuple[str, str, str]],
) -> None:
    summary_path = branch_dir / "session_summaries.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gesture_id",
                "person_id",
                "session_id",
                "run_path",
                "total_frames",
                "accepted_frames",
                "outlier_frames",
                "selected_for_balanced_model",
            ],
        )
        writer.writeheader()
        for item in sorted(
            prepared,
            key=lambda value: (
                value.entry.gesture_id,
                value.entry.person_id,
                value.entry.session_id,
            ),
        ):
            key = (item.entry.gesture_id, item.entry.person_id, item.entry.session_id)
            writer.writerow(
                {
                    "gesture_id": item.entry.gesture_id,
                    "person_id": item.entry.person_id,
                    "session_id": item.entry.session_id,
                    "run_path": Path(
                        __import__("os").path.relpath(item.loaded.run_dir, branch_dir)
                    ).as_posix(),
                    "total_frames": item.statistics.total_frame_count,
                    "accepted_frames": item.statistics.accepted_frame_count,
                    "outlier_frames": item.statistics.outlier_frame_count,
                    "selected_for_balanced_model": int(key in selected_session_keys),
                }
            )
    long_rows = []
    for item in prepared:
        stats = item.statistics
        long_rows.append(
            (
                {
                    "gesture_id": stats.gesture_id,
                    "person_id": stats.person_id,
                    "session_id": stats.session_id,
                },
                {
                    "mean": stats.mean,
                    "median": stats.median,
                    "standard_deviation": stats.standard_deviation,
                    "q1": stats.q1,
                    "q3": stats.q3,
                },
            )
        )
    _write_long_feature_csv(
        branch_dir / "session_feature_statistics.csv", long_rows, feature_ids
    )
    outlier_path = branch_dir / "session_outliers.csv"
    with outlier_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gesture_id",
                "person_id",
                "session_id",
                "frame_index",
                "relative_time_s",
                "accepted",
                "robust_rms_score",
                "extreme_feature_fraction",
                "top_offending_features",
            ],
        )
        writer.writeheader()
        for item in prepared:
            for index in range(item.loaded.frame_count):
                top_ids = [feature_ids[i] for i in item.outliers.top_feature_indices[index]]
                writer.writerow(
                    {
                        "gesture_id": item.entry.gesture_id,
                        "person_id": item.entry.person_id,
                        "session_id": item.entry.session_id,
                        "frame_index": int(item.loaded.frame_indices[index]),
                        "relative_time_s": f"{float(item.loaded.relative_times_s[index]):.9f}",
                        "accepted": int(item.outliers.accepted_mask[index]),
                        "robust_rms_score": f"{float(item.outliers.robust_rms_score[index]):.12g}",
                        "extreme_feature_fraction": f"{float(item.outliers.extreme_feature_fraction[index]):.12g}",
                        "top_offending_features": ";".join(top_ids),
                    }
                )


def _feature_weight_diagnostics(
    gestures: list[GestureStatistics],
    active_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    prototypes = np.vstack([gesture.prototype_mean for gesture in gestures])
    between = np.var(prototypes, axis=0)
    within = np.mean(
        np.vstack([np.maximum(gesture.standard_deviation, 0.0) ** 2 for gesture in gestures]),
        axis=0,
    )
    fisher = between / (within + 1.0e-12)
    inverse_within = 1.0 / (within + 1.0e-8)
    fisher[~active_mask] = 0.0
    inverse_within[~active_mask] = 0.0
    for values in (fisher, inverse_within):
        active_values = values[active_mask]
        if active_values.size and float(active_values.mean()) > 0:
            values[active_mask] = active_values / float(active_values.mean())
    return {
        "between_class_variance": between,
        "within_class_variance": within,
        "fisher_score_weight": fisher,
        "inverse_within_variance_weight": inverse_within,
    }



def _resolve_compact_correlation_threshold(config: dict[str, Any]) -> float:
    """Return a validated fold-local PLCC pruning threshold."""
    section = config.get("compact_feature_mask", {})
    value = float(section.get("correlation_threshold", 0.995))
    if not 0.0 < value <= 1.0:
        raise ValueError("compact feature correlation threshold must be in (0, 1].")
    return value



def _build_compact_redundancy_mask(
    values: np.ndarray,
    feature_ids: tuple[str, ...],
    active_mask: np.ndarray,
    fisher_weights: np.ndarray,
    *,
    correlation_threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]], tuple[str, ...]]:
    """Remove near-duplicate active features using training-fold-only correlation.

    The compact mask is diagnostic and classifier-ready but is derived exclusively
    from the reference-model training fold.  When two active features are highly
    correlated, the feature with the larger Fisher weight is retained; ties keep
    the lower schema index for deterministic reproducibility.
    """
    matrix = np.asarray(values, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool).copy()
    reasons = ["active" if value else "inactive_base_mask" for value in active]
    indices = np.flatnonzero(active)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_ids):
        raise ValueError("Compact-mask matrix shape does not match feature identifiers.")
    if indices.size < 2:
        return active, [], tuple(reasons)
    selected = matrix[:, indices]
    corr = np.corrcoef(selected, rowvar=False)
    pairs: list[tuple[float, int, int, float]] = []
    for left in range(indices.size):
        for right in range(left + 1, indices.size):
            value = float(corr[left, right])
            if np.isfinite(value) and abs(value) >= correlation_threshold:
                pairs.append((abs(value), int(indices[left]), int(indices[right]), value))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    redundant_rows: list[dict[str, Any]] = []
    weights = np.asarray(fisher_weights, dtype=np.float64)
    for absolute, first, second, value in pairs:
        if not active[first] or not active[second]:
            continue
        first_weight = float(weights[first]) if np.isfinite(weights[first]) else 0.0
        second_weight = float(weights[second]) if np.isfinite(weights[second]) else 0.0
        if second_weight > first_weight:
            keep, remove = second, first
        else:
            keep, remove = first, second
        active[remove] = False
        reasons[remove] = (
            f"redundant_with:{feature_ids[keep]}:correlation={value:.12g}:"
            f"kept_by_fisher_weight"
        )
        redundant_rows.append(
            {
                "feature_keep_index": keep,
                "feature_keep_id": feature_ids[keep],
                "feature_remove_index": remove,
                "feature_remove_id": feature_ids[remove],
                "correlation": value,
                "absolute_correlation": absolute,
                "keep_fisher_weight": float(weights[keep]),
                "remove_fisher_weight": float(weights[remove]),
            }
        )
    return active, redundant_rows, tuple(reasons)


def _write_branch_model(
    output_dir: Path,
    branch_name: str,
    prepared: list[PreparedSession],
    signature: BranchSignature,
    feature_ids: tuple[str, ...],
    config: dict[str, Any],
) -> int:
    branch_dir = output_dir / "branches" / branch_directory_name(branch_name)
    branch_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        branch_dir / "feature_schema_snapshot.json",
        prepared[0].loaded.schema_payload,
    )
    accepted_all = np.vstack([item.statistics.accepted_values for item in prepared])
    mask = build_feature_mask(
        accepted_all,
        feature_ids,
        exact_tolerance=float(config["feature_mask"]["exact_tolerance"]),
        absolute_std_threshold=float(config["feature_mask"]["absolute_std_threshold"]),
        relative_std_threshold=float(config["feature_mask"]["relative_std_threshold"]),
    )
    write_json(
        branch_dir / "feature_mask.json",
        {
            "feature_mask_sha256": mask.mask_sha256,
            "feature_count": len(feature_ids),
            "active_feature_count": int(mask.active_mask.sum()),
            "inactive_feature_count": int((~mask.active_mask).sum()),
            "features": [
                {
                    "feature_index": index,
                    "feature_id": feature_id,
                    "active": bool(mask.active_mask[index]),
                    "reason": mask.reasons[index],
                    "standard_deviation": float(mask.standard_deviation[index]),
                    "value_range": float(mask.value_range[index]),
                    "median_absolute_value": float(mask.median_absolute_value[index]),
                }
                for index, feature_id in enumerate(feature_ids)
            ],
        },
    )
    with (branch_dir / "feature_mask.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_index",
                "feature_id",
                "active",
                "reason",
                "standard_deviation",
                "value_range",
                "median_absolute_value",
            ],
        )
        writer.writeheader()
        for index, feature_id in enumerate(feature_ids):
            writer.writerow(
                {
                    "feature_index": index,
                    "feature_id": feature_id,
                    "active": int(mask.active_mask[index]),
                    "reason": mask.reasons[index],
                    "standard_deviation": f"{float(mask.standard_deviation[index]):.12g}",
                    "value_range": f"{float(mask.value_range[index]):.12g}",
                    "median_absolute_value": f"{float(mask.median_absolute_value[index]):.12g}",
                }
            )

    by_gesture: dict[str, list[SessionStatistics]] = {}
    for item in prepared:
        by_gesture.setdefault(item.entry.gesture_id, []).append(item.statistics)
    gestures: list[GestureStatistics] = []
    for gesture_id, sessions in sorted(by_gesture.items()):
        gestures.append(
            aggregate_gesture(
                gesture_id,
                sessions,
                session_balance_mode=str(config["session_balance"]["mode"]),
                fixed_session_count=(
                    None
                    if config["session_balance"].get("fixed_count") in (None, "")
                    else int(config["session_balance"]["fixed_count"])
                ),
                balance_seed=int(config["session_balance"]["seed"]),
                covariance_shrinkage=float(config["covariance"]["shrinkage"]),
                covariance_diagonal_floor=float(config["covariance"]["diagonal_floor"]),
            )
        )

    selected_keys: set[tuple[str, str, str]] = set()
    for gesture in gestures:
        for person in gesture.person_statistics:
            for session_id in person.selected_session_ids:
                selected_keys.add((gesture.gesture_id, person.person_id, session_id))
    _write_session_outputs(branch_dir, prepared, feature_ids, selected_keys)

    person_rows = []
    gesture_rows = []
    for gesture in gestures:
        for person in gesture.person_statistics:
            person_rows.append(
                (
                    {
                        "gesture_id": gesture.gesture_id,
                        "person_id": person.person_id,
                        "selected_session_ids": ";".join(person.selected_session_ids),
                    },
                    {
                        "mean": person.mean,
                        "median": person.median,
                        "standard_deviation": person.standard_deviation,
                        "q1": person.q1,
                        "q3": person.q3,
                    },
                )
            )
        gesture_rows.append(
            (
                {"gesture_id": gesture.gesture_id},
                {
                    "prototype_mean": gesture.prototype_mean,
                    "prototype_median": gesture.prototype_median,
                    "standard_deviation": gesture.standard_deviation,
                    "q1": gesture.q1,
                    "q3": gesture.q3,
                },
            )
        )
    _write_long_feature_csv(branch_dir / "person_feature_statistics.csv", person_rows, feature_ids)
    _write_long_feature_csv(branch_dir / "gesture_feature_statistics.csv", gesture_rows, feature_ids)

    weights = _feature_weight_diagnostics(gestures, mask.active_mask)

    # Build a fold-local compact nonredundant mask from only the sessions that
    # actually contribute to the balanced reference models.
    selected_accepted = np.vstack([
        item.statistics.accepted_values
        for item in prepared
        if (item.entry.gesture_id, item.entry.person_id, item.entry.session_id) in selected_keys
    ])
    compact_correlation_threshold = _resolve_compact_correlation_threshold(config)
    compact_mask, compact_redundant_rows, compact_reasons = _build_compact_redundancy_mask(
        selected_accepted,
        feature_ids,
        mask.active_mask,
        weights["fisher_score_weight"],
        correlation_threshold=compact_correlation_threshold,
    )
    compact_hash = hashlib.sha256(
        json.dumps(
            {"feature_ids": list(feature_ids), "mask": compact_mask.astype(int).tolist()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    compact_payload = {
        "schema_version": "tsgr_reference_compact_feature_mask_v1",
        "source": "selected_reference_training_sessions_only",
        "correlation_threshold": compact_correlation_threshold,
        "feature_count": len(feature_ids),
        "base_active_feature_count": int(mask.active_mask.sum()),
        "compact_active_feature_count": int(compact_mask.sum()),
        "compact_mask_sha256": compact_hash,
        "features": [
            {
                "feature_index": index,
                "feature_id": feature_id,
                "base_active": bool(mask.active_mask[index]),
                "compact_active": bool(compact_mask[index]),
                "reason": compact_reasons[index],
                "fisher_score_weight": float(weights["fisher_score_weight"][index]),
            }
            for index, feature_id in enumerate(feature_ids)
        ],
        "redundant_pairs_used": compact_redundant_rows,
    }
    write_json(branch_dir / "compact_feature_mask.json", compact_payload)
    with (branch_dir / "compact_feature_mask.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_index", "feature_id", "base_active", "compact_active",
                "reason", "fisher_score_weight",
            ],
        )
        writer.writeheader()
        for row in compact_payload["features"]:
            writer.writerow(row)
    with (branch_dir / "compact_redundant_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "feature_keep_index", "feature_keep_id", "feature_remove_index",
            "feature_remove_id", "correlation", "absolute_correlation",
            "keep_fisher_weight", "remove_fisher_weight",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact_redundant_rows)

    with (branch_dir / "feature_weight_diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_index", "feature_id", "active", *weights.keys()],
        )
        writer.writeheader()
        for index, feature_id in enumerate(feature_ids):
            writer.writerow(
                {
                    "feature_index": index,
                    "feature_id": feature_id,
                    "active": int(mask.active_mask[index]),
                    **{
                        name: f"{float(values[index]):.12g}"
                        for name, values in weights.items()
                    },
                }
            )

    active_indices = np.flatnonzero(mask.active_mask)
    for gesture in gestures:
        gesture_dir = branch_dir / "gestures" / gesture.gesture_id
        gesture_dir.mkdir(parents=True, exist_ok=True)
        selected_sessions = sum(
            len(person.selected_session_ids) for person in gesture.person_statistics
        )
        metadata = {
            "gesture_id": gesture.gesture_id,
            "branch_name": branch_name,
            "branch_signature": signature.to_dict(),
            "feature_mask_sha256": mask.mask_sha256,
            "feature_schema_snapshot": "feature_schema_snapshot.json",
            "feature_count": len(feature_ids),
            "active_feature_count": int(active_indices.size),
            "person_count": len(gesture.person_statistics),
            "available_session_count": len(gesture.session_statistics),
            "selected_balanced_session_count": selected_sessions,
            "available_accepted_frame_count": int(
                sum(session.accepted_frame_count for session in gesture.session_statistics)
            ),
            "selected_accepted_frame_count": int(
                sum(
                    session.accepted_frame_count
                    for session in gesture.session_statistics
                    if any(
                        session.person_id == person.person_id
                        and session.session_id in person.selected_session_ids
                        for person in gesture.person_statistics
                    )
                )
            ),
            # This field reports the number available before session balancing.
            "accepted_frame_count": int(
                sum(session.accepted_frame_count for session in gesture.session_statistics)
            ),
            "outlier_frame_count": int(
                sum(session.outlier_frame_count for session in gesture.session_statistics)
            ),
            "session_balance": config["session_balance"],
            "covariance": config["covariance"],
            "aggregation": {
                "session": "accepted-frame mean/median/standard deviation/quartiles",
                "person": "equal-weight mean of selected session means",
                "gesture": "equal-weight mean of person means",
            },
            "persons": [
                {
                    "person_id": person.person_id,
                    "selected_session_ids": list(person.selected_session_ids),
                }
                for person in gesture.person_statistics
            ],
        }
        write_json(gesture_dir / "model.json", metadata)
        np.savez_compressed(
            gesture_dir / "model_arrays.npz",
            feature_ids=np.asarray(feature_ids),
            active_mask=mask.active_mask,
            active_indices=active_indices,
            prototype_mean=gesture.prototype_mean,
            prototype_median=gesture.prototype_median,
            standard_deviation=gesture.standard_deviation,
            q1=gesture.q1,
            q3=gesture.q3,
            covariance_empirical_active=gesture.covariance_empirical[
                np.ix_(active_indices, active_indices)
            ],
            covariance_regularized_active=gesture.covariance_regularized[
                np.ix_(active_indices, active_indices)
            ],
            diagonal_variance_active=gesture.diagonal_variance[active_indices],
            person_ids=np.asarray([person.person_id for person in gesture.person_statistics]),
            person_means=np.vstack([person.mean for person in gesture.person_statistics]),
            person_medians=np.vstack([person.median for person in gesture.person_statistics]),
        )

    write_json(
        branch_dir / "branch_model.json",
        {
            "branch_name": branch_name,
            "branch_signature": signature.to_dict(),
            "feature_mask_sha256": mask.mask_sha256,
            "feature_schema_snapshot": "feature_schema_snapshot.json",
            "feature_count": len(feature_ids),
            "active_feature_count": int(mask.active_mask.sum()),
            "compact_active_feature_count": int(compact_mask.sum()),
            "compact_correlation_threshold": compact_correlation_threshold,
            "compact_feature_mask_sha256": compact_hash,
            "compact_feature_mask": "compact_feature_mask.json",
            "gesture_ids": [gesture.gesture_id for gesture in gestures],
            "gesture_count": len(gestures),
            "session_count": len(prepared),
            "person_count": len({item.entry.person_id for item in prepared}),
            "unique_person_count": len({item.entry.person_id for item in prepared}),
            "person_gesture_count": len(
                {(item.entry.gesture_id, item.entry.person_id) for item in prepared}
            ),
        },
    )
    return len(gestures)


def build_reference_models(
    entries: list[ReferenceManifestEntry],
    *,
    branches: list[str] | tuple[str, ...],
    output_root: str | Path,
    config: dict[str, Any],
    model_set_name: str | None = None,
) -> ReferenceModelBuildResult:
    """Build a complete model set with strict per-branch compatibility checks."""
    included = [entry for entry in entries if entry.include]
    if not included:
        raise ValueError("Reference manifest contains no included sessions.")
    branch_names = tuple(dict.fromkeys(str(branch) for branch in branches))
    if not branch_names:
        raise ValueError("At least one feature branch must be requested.")
    resolved_config = copy.deepcopy(config)
    resolved_config["branches"] = list(branch_names)
    compact_correlation_threshold = _resolve_compact_correlation_threshold(resolved_config)
    resolved_config.setdefault("compact_feature_mask", {})["correlation_threshold"] = compact_correlation_threshold
    output_dir = Path(output_root) / (model_set_name or _timestamp_id())
    if output_dir.exists():
        raise FileExistsError(f"Reference model-set directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_reference_manifest(output_dir / "dataset_manifest_snapshot.csv", entries)
    with (output_dir / "build_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, allow_unicode=True, sort_keys=False)
    write_json(output_dir / "environment.json", environment_report())

    model_count = 0
    branch_summaries: list[dict[str, Any]] = []
    all_gestures = sorted({entry.gesture_id for entry in included})
    try:
        for branch_name in branch_names:
            loaded = [load_feature_session(entry.run_path, branch_name) for entry in included]
            signature = _validate_branch_signatures(loaded)
            feature_ids = loaded[0].feature_ids
            prepared = [
                _prepare_session(entry, session, resolved_config["outlier_detection"])
                for entry, session in zip(included, loaded)
            ]
            gesture_count = _write_branch_model(
                output_dir,
                branch_name,
                prepared,
                signature,
                feature_ids,
                resolved_config,
            )
            model_count += gesture_count
            branch_summaries.append(
                {
                    "branch_name": branch_name,
                    "signature": signature.to_dict(),
                    "gesture_count": gesture_count,
                    "session_count": len(prepared),
                }
            )
        summary = {
            "model_set_name": output_dir.name,
            "model_set_format_version": "tsgr_reference_models_v1",
            "project_version": __version__,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "branches": branch_summaries,
            "branch_names": list(branch_names),
            "gesture_ids": all_gestures,
            "gesture_count": len(all_gestures),
            "model_count": model_count,
            "manifest_entry_count": len(entries),
            "included_manifest_entry_count": len(included),
            "excluded_manifest_entry_count": len(entries) - len(included),
            "compact_correlation_threshold": compact_correlation_threshold,
            "build_config_sha256": _hash_payload(resolved_config),
        }
        write_json(output_dir / "model_set.json", summary)
    except Exception:
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return ReferenceModelBuildResult(
        output_dir=output_dir,
        branches=branch_names,
        gestures=tuple(all_gestures),
        model_count=model_count,
    )
