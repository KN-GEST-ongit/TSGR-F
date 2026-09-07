"""Writers for per-frame JSONL, normalization arrays, and named feature tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from tsgr.features.schema import FEATURE_COUNT, feature_definitions
from tsgr.types import FrameAnalysis
from tsgr.utils.serialization import append_jsonl, write_json


FEATURE_BRANCH_ORDER = (
    "raw.wrist_middle_mcp",
    "raw.middle_finger",
    "filtered.wrist_middle_mcp",
    "filtered.middle_finger",
)


class FrameResultWriter:
    """Write authoritative JSONL and derivative research-friendly feature artifacts."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        scales_csv_path: str | Path | None = None,
        normalization_npz_path: str | Path | None = None,
        save_compact_npz: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

        self.scales_csv_path = (
            Path(scales_csv_path)
            if scales_csv_path is not None
            else self.output_path.parent / "normalization_scales.csv"
        )
        self.normalization_npz_path = (
            Path(normalization_npz_path)
            if normalization_npz_path is not None
            else self.output_path.parent / "normalization_arrays.npz"
        )
        self.save_compact_npz = bool(save_compact_npz)

        self.features_dir = self.output_path.parent / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.classification_features_csv = (
            self.features_dir / "classification_features.csv"
        )
        self.all_feature_branches_csv = (
            self.features_dir / "all_feature_branches.csv"
        )
        self.feature_arrays_npz = self.features_dir / "feature_arrays.npz"
        self.feature_schema_json = self.features_dir / "feature_schema.json"

        for path in (
            self.scales_csv_path,
            self.normalization_npz_path,
            self.classification_features_csv,
            self.all_feature_branches_csv,
            self.feature_arrays_npz,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()

        definitions = feature_definitions()
        write_json(
            self.feature_schema_json,
            {
                "schema_version": "tsgr_full_features_v1",
                "feature_count": FEATURE_COUNT,
                "branch_order": list(FEATURE_BRANCH_ORDER),
                "features": [definition.to_dict() for definition in definitions],
            },
        )
        feature_ids = [definition.feature_id for definition in definitions]

        self._scale_handle = self.scales_csv_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._scale_writer = csv.DictWriter(
            self._scale_handle,
            fieldnames=[
                "frame_index",
                "relative_time_s",
                "status",
                "normalization_valid",
                "scale_wrist_middle_mcp",
                "scale_middle_finger_polyline",
                "scale_middle_finger_chord",
                "scale_ratio_middle_to_palm",
                "basis_determinant",
                "axis_orthogonality_error",
                "maximum_reconstruction_error",
                "temporal_filter_name",
                "classification_feature_branch",
                "quality_status",
                "quality_warning_reasons",
                "quality_hard_rejection_reasons",
                "temporal_geometry_outlier",
                "angular_speed_deg_s",
            ],
        )
        self._scale_writer.writeheader()

        selected_fields = [
            "frame_index",
            "relative_time_s",
            "status",
            "classification_feature_branch",
            "feature_complete",
            "finite_feature_count",
            "feature_warnings",
            *feature_ids,
        ]
        self._classification_handle = self.classification_features_csv.open(
            "w", encoding="utf-8", newline=""
        )
        self._classification_writer = csv.DictWriter(
            self._classification_handle,
            fieldnames=selected_fields,
        )
        self._classification_writer.writeheader()

        branch_fields = [
            "frame_index",
            "relative_time_s",
            "status",
            "branch",
            "landmark_source",
            "scale_name",
            "is_classification_branch",
            "feature_complete",
            "finite_feature_count",
            "feature_warnings",
            *feature_ids,
        ]
        self._branches_handle = self.all_feature_branches_csv.open(
            "w", encoding="utf-8", newline=""
        )
        self._branches_writer = csv.DictWriter(
            self._branches_handle,
            fieldnames=branch_fields,
        )
        self._branches_writer.writeheader()

        self._records: list[dict[str, Any]] = []
        self._feature_records: list[dict[str, Any]] = []
        self._closed = False

    @staticmethod
    def _format_feature_values(values: np.ndarray) -> dict[str, str]:
        return {
            definition.feature_id: (
                f"{float(value):.12g}" if np.isfinite(value) else ""
            )
            for definition, value in zip(feature_definitions(), values)
        }

    def _write_features(self, analysis: FrameAnalysis) -> None:
        selected_vector = analysis.classification_feature_vector
        selected_row: dict[str, Any] = {
            "frame_index": analysis.frame_index,
            "relative_time_s": f"{analysis.relative_time_s:.9f}",
            "status": analysis.status,
            "classification_feature_branch": (
                analysis.classification_feature_branch or ""
            ),
            "feature_complete": int(
                selected_vector.complete if selected_vector is not None else False
            ),
            "finite_feature_count": (
                selected_vector.finite_count if selected_vector is not None else 0
            ),
            "feature_warnings": (
                ";".join(selected_vector.warnings)
                if selected_vector is not None
                else ""
            ),
        }
        if selected_vector is not None:
            selected_row.update(self._format_feature_values(selected_vector.values))
        self._classification_writer.writerow(selected_row)

        for branch in FEATURE_BRANCH_ORDER:
            vector = analysis.feature_vectors.get(branch)
            if vector is None:
                continue
            row: dict[str, Any] = {
                "frame_index": analysis.frame_index,
                "relative_time_s": f"{analysis.relative_time_s:.9f}",
                "status": analysis.status,
                "branch": branch,
                "landmark_source": vector.landmark_source,
                "scale_name": vector.scale_name,
                "is_classification_branch": int(
                    branch == analysis.classification_feature_branch
                ),
                "feature_complete": int(vector.complete),
                "finite_feature_count": vector.finite_count,
                "feature_warnings": ";".join(vector.warnings),
            }
            row.update(self._format_feature_values(vector.values))
            self._branches_writer.writerow(row)

        if self.save_compact_npz:
            branch_values = np.full(
                (len(FEATURE_BRANCH_ORDER), FEATURE_COUNT),
                np.nan,
                dtype=np.float64,
            )
            for index, branch in enumerate(FEATURE_BRANCH_ORDER):
                vector = analysis.feature_vectors.get(branch)
                if vector is not None:
                    branch_values[index] = vector.values
            self._feature_records.append(
                {
                    "frame_index": analysis.frame_index,
                    "relative_time_s": analysis.relative_time_s,
                    "status": analysis.status,
                    "classification_feature_branch": (
                        analysis.classification_feature_branch or ""
                    ),
                    "values": branch_values,
                }
            )

    def write(self, analysis: FrameAnalysis) -> None:
        if self._closed:
            raise RuntimeError("Cannot write after FrameResultWriter.close().")
        append_jsonl(self.output_path, analysis.to_dict())

        normalization = analysis.normalization
        angular_speed = analysis.temporal_diagnostics.get("angular_speed_deg_s")
        row: dict[str, Any] = {
            "frame_index": analysis.frame_index,
            "relative_time_s": f"{analysis.relative_time_s:.9f}",
            "status": analysis.status,
            "normalization_valid": int(normalization is not None),
            "scale_wrist_middle_mcp": "",
            "scale_middle_finger_polyline": "",
            "scale_middle_finger_chord": "",
            "scale_ratio_middle_to_palm": "",
            "basis_determinant": "",
            "axis_orthogonality_error": "",
            "maximum_reconstruction_error": "",
            "temporal_filter_name": analysis.temporal_filter_name,
            "classification_feature_branch": (
                analysis.classification_feature_branch or ""
            ),
            "quality_status": analysis.quality_metrics.get("quality_status", ""),
            "quality_warning_reasons": ";".join(
                analysis.quality_metrics.get("warning_reasons", [])
            ),
            "quality_hard_rejection_reasons": ";".join(
                analysis.quality_metrics.get("hard_rejection_reasons", [])
            ),
            "temporal_geometry_outlier": int(
                bool(
                    analysis.temporal_diagnostics.get(
                        "temporal_geometry_outlier", False
                    )
                )
            ),
            "angular_speed_deg_s": (
                "" if angular_speed is None else f"{float(angular_speed):.12g}"
            ),
        }
        if normalization is not None:
            row.update(
                {
                    "scale_wrist_middle_mcp": (
                        f"{normalization.scale_wrist_middle_mcp:.12g}"
                    ),
                    "scale_middle_finger_polyline": (
                        f"{normalization.scale_middle_finger:.12g}"
                    ),
                    "scale_middle_finger_chord": (
                        f"{normalization.scale_middle_finger_chord:.12g}"
                    ),
                    "scale_ratio_middle_to_palm": (
                        f"{normalization.scale_ratio_middle_to_palm:.12g}"
                    ),
                    "basis_determinant": (
                        f"{float(normalization.diagnostics['basis_determinant']):.12g}"
                    ),
                    "axis_orthogonality_error": (
                        f"{float(normalization.diagnostics['axis_orthogonality_error']):.12g}"
                    ),
                    "maximum_reconstruction_error": (
                        f"{float(normalization.diagnostics['maximum_reconstruction_error']):.12g}"
                    ),
                }
            )
        self._scale_writer.writerow(row)
        self._write_features(analysis)

        self._scale_handle.flush()
        self._classification_handle.flush()
        self._branches_handle.flush()
        if self.save_compact_npz:
            self._records.append(self._array_record(analysis))

    @staticmethod
    def _array_record(analysis: FrameAnalysis) -> dict[str, Any]:
        nan21 = np.full((21, 3), np.nan, dtype=np.float64)
        nan3 = np.full((3,), np.nan, dtype=np.float64)
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)
        normalization = analysis.normalization
        raw_hand = analysis.selected_right_hand
        filtered_hand = analysis.filtered_right_hand
        raw_world = nan21.copy() if raw_hand is None else raw_hand.world_landmarks
        raw_image = nan21.copy() if raw_hand is None else raw_hand.image_landmarks
        filtered_world = (
            nan21.copy() if filtered_hand is None else filtered_hand.world_landmarks
        )
        filtered_image = (
            nan21.copy() if filtered_hand is None else filtered_hand.image_landmarks
        )
        if normalization is None:
            return {
                "frame_index": analysis.frame_index,
                "relative_time_s": analysis.relative_time_s,
                "valid": False,
                "raw_world_landmarks": raw_world,
                "raw_image_landmarks": raw_image,
                "filtered_world_landmarks": filtered_world,
                "filtered_image_landmarks": filtered_image,
                "temporal_filter_name": analysis.temporal_filter_name,
                "classification_feature_branch": (
                    analysis.classification_feature_branch or ""
                ),
                "origin_world": nan3,
                "axes_world": nan33,
                "centered_world": nan21,
                "aligned_world": nan21.copy(),
                "normalized_wrist_middle_mcp": nan21.copy(),
                "normalized_middle_finger": nan21.copy(),
                "scale_wrist_middle_mcp": np.nan,
                "scale_middle_finger": np.nan,
                "scale_middle_finger_chord": np.nan,
                "scale_ratio_middle_to_palm": np.nan,
            }
        return {
            "frame_index": analysis.frame_index,
            "relative_time_s": analysis.relative_time_s,
            "valid": True,
            "raw_world_landmarks": raw_world,
            "raw_image_landmarks": raw_image,
            "filtered_world_landmarks": filtered_world,
            "filtered_image_landmarks": filtered_image,
            "temporal_filter_name": analysis.temporal_filter_name,
            "classification_feature_branch": (
                analysis.classification_feature_branch or ""
            ),
            "origin_world": normalization.origin_world,
            "axes_world": normalization.axes_world,
            "centered_world": normalization.centered_world,
            "aligned_world": normalization.aligned_world,
            "normalized_wrist_middle_mcp": (
                normalization.normalized_wrist_middle_mcp
            ),
            "normalized_middle_finger": normalization.normalized_middle_finger,
            "scale_wrist_middle_mcp": normalization.scale_wrist_middle_mcp,
            "scale_middle_finger": normalization.scale_middle_finger,
            "scale_middle_finger_chord": normalization.scale_middle_finger_chord,
            "scale_ratio_middle_to_palm": normalization.scale_ratio_middle_to_palm,
        }

    @staticmethod
    def _records_to_npz(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not records:
            return {}
        arrays: dict[str, np.ndarray] = {}
        for key in records[0]:
            arrays[key] = np.asarray([record[key] for record in records])
        return arrays

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._scale_handle.close()
        self._classification_handle.close()
        self._branches_handle.close()

        if self.save_compact_npz and self._records:
            np.savez_compressed(
                self.normalization_npz_path,
                **self._records_to_npz(self._records),
            )
        if self.save_compact_npz and self._feature_records:
            payload = self._records_to_npz(self._feature_records)
            payload["branch_names"] = np.asarray(FEATURE_BRANCH_ORDER)
            payload["feature_ids"] = np.asarray(
                [definition.feature_id for definition in feature_definitions()]
            )
            np.savez_compressed(self.feature_arrays_npz, **payload)

        self._records.clear()
        self._feature_records.clear()

    def __enter__(self) -> "FrameResultWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
