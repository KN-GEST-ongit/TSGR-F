"""Frame pipeline: detection, quality, dual branches, normalization, and features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tsgr.detection.mediapipe_hand_landmarker import MediaPipeHandLandmarker
from tsgr.detection.right_hand_selector import (
    contains_effective_left_hand,
    effective_handedness,
    select_hand_ignoring_handedness,
    select_right_hand,
)
from tsgr.features.extractor import FeatureExtractionError, extract_feature_vector
from tsgr.filtering.factory import create_landmark_filter
from tsgr.preprocessing.roi import ROITracker, crop_image, roi_from_normalized_landmarks
from tsgr.preprocessing.spatial_normalization import (
    SpatialNormalizationError,
    SpatialNormalizationResult,
    normalize_world_landmarks,
)
from tsgr.quality.frame_quality import evaluate_frame_quality
from tsgr.quality.temporal_geometry import TemporalGeometryMonitor
from tsgr.types import FrameAnalysis, FramePacket, HandCandidate


class FramePipeline:
    """Process frames while preserving independent raw and filtered feature branches."""

    def __init__(self, config: dict[str, Any], model_path: str | Path) -> None:
        self.config = config
        self.detector = MediaPipeHandLandmarker(config["mediapipe"], model_path)

        roi = config["roi"]
        self.roi_tracker = ROITracker(
            float(roi["smoothing_alpha"]),
            int(roi["full_frame_interval"]),
            int(roi["maximum_missed_frames"]),
        )

        temporal_filter = config.get("temporal_filter", {})
        self.filter_name = str(temporal_filter.get("active", "none"))
        filter_params = dict(
            temporal_filter.get("filters", {}).get(self.filter_name, {})
        )
        filter_params["maximum_gap_s"] = float(
            temporal_filter.get("maximum_gap_s", 0.25)
        )
        self.world_filter = create_landmark_filter(self.filter_name, filter_params)
        self.image_filter = create_landmark_filter(self.filter_name, filter_params)
        self.reset_on_missing = bool(
            temporal_filter.get("reset_on_missing_hand", True)
        )
        self.geometry_monitor = TemporalGeometryMonitor(
            config.get("temporal_geometry", {})
        )

        classification_input = config.get("classification_input", {})
        self.classification_landmark_source = str(
            classification_input.get("landmark_source", "raw")
        )
        self.classification_scale = str(
            classification_input.get("scale", "wrist_middle_mcp")
        )
        self.export_all_feature_branches = bool(
            classification_input.get("export_all_feature_branches", True)
        )
        self.features_enabled = bool(config.get("features", {}).get("enabled", True))

    def _normalize(
        self,
        hand: HandCandidate,
        notes: list[str],
        branch_name: str,
    ) -> SpatialNormalizationResult | None:
        if not bool(self.config.get("normalization", {}).get("enabled", True)):
            return None
        normalization_config = self.config["normalization"]
        try:
            return normalize_world_landmarks(
                hand.world_landmarks,
                minimum_axis_norm=float(
                    normalization_config.get("minimum_axis_norm", 1e-8)
                ),
                minimum_scale=float(
                    normalization_config.get("minimum_scale", 1e-6)
                ),
            )
        except SpatialNormalizationError as error:
            notes.append(f"SpatialNormalizationError[{branch_name}]: {error}")
            return None

    @staticmethod
    def _branch_key(landmark_source: str, scale_name: str) -> str:
        return f"{landmark_source}.{scale_name}"

    def _extract_features(
        self,
        raw_normalization: SpatialNormalizationResult | None,
        filtered_normalization: SpatialNormalizationResult | None,
        notes: list[str],
    ) -> dict[str, Any]:
        if not self.features_enabled:
            return {}

        requested: list[tuple[str, str]] = []
        if self.export_all_feature_branches:
            requested.extend(
                (source, scale)
                for source in ("raw", "filtered")
                for scale in ("wrist_middle_mcp", "middle_finger")
            )
        else:
            requested.append(
                (
                    self.classification_landmark_source,
                    self.classification_scale,
                )
            )

        results: dict[str, Any] = {}
        normalizations = {
            "raw": raw_normalization,
            "filtered": filtered_normalization,
        }
        for source, scale in requested:
            normalization = normalizations[source]
            if normalization is None:
                continue
            branch_key = self._branch_key(source, scale)
            try:
                results[branch_key] = extract_feature_vector(
                    normalization,
                    landmark_source=source,
                    scale_name=scale,
                )
            except FeatureExtractionError as error:
                notes.append(f"FeatureExtractionError[{branch_key}]: {error}")
        return results

    def process(self, packet: FramePacket) -> FrameAnalysis:
        total_start = time.perf_counter_ns()
        image = packet.image_bgr
        height, width = image.shape[:2]
        roi_config = self.config["roi"]

        inference_roi = None
        detection_image = image
        if (
            bool(roi_config.get("enabled", True))
            and bool(roi_config.get("inference_enabled", False))
            and self.roi_tracker.should_use_roi(packet.frame_index)
            and self.roi_tracker.current is not None
        ):
            inference_roi = self.roi_tracker.current
            detection_image = crop_image(image, inference_roi)

        detection_start = time.perf_counter_ns()
        candidates = self.detector.detect(
            detection_image,
            int(round(packet.relative_time_s * 1000)),
            (width, height) if inference_roi is not None else None,
            inference_roi,
        )
        detection_end = time.perf_counter_ns()

        invert_handedness = bool(self.config["mediapipe"]["invert_handedness_labels"])
        handedness_policy = str(
            self.config.get("mediapipe", {}).get("handedness_policy", "reject_left")
        ).strip().lower()
        if handedness_policy not in {"reject_left", "ignore_handedness"}:
            raise ValueError(
                "mediapipe.handedness_policy must be 'reject_left' or 'ignore_handedness'."
            )
        if handedness_policy == "ignore_handedness":
            selected = select_hand_ignoring_handedness(candidates)
        else:
            selected = select_right_hand(
                candidates,
                invert_labels=invert_handedness,
            )
        detection_diagnostics = dict(getattr(self.detector, "last_diagnostics", {}) or {})
        left_observed = bool(
            contains_effective_left_hand(
                candidates,
                invert_labels=invert_handedness,
            )
            or detection_diagnostics.get("left_observed_in_any_attempt", False)
        )
        critical_left_hand = bool(
            handedness_policy == "reject_left" and left_observed
        )
        selected_effective_handedness = (
            effective_handedness(selected.handedness_label, invert_handedness)
            if selected is not None
            else ""
        )
        detection_diagnostics["handedness_policy"] = handedness_policy
        detection_diagnostics["left_observed"] = left_observed
        detection_diagnostics["selected_raw_handedness"] = (
            selected.handedness_label if selected is not None else ""
        )
        detection_diagnostics["selected_effective_handedness"] = selected_effective_handedness
        detection_diagnostics["handedness_mismatch"] = bool(
            selected is not None and selected_effective_handedness == "left"
        )
        detection_diagnostics["handedness_policy_action"] = (
            "reject_frame"
            if critical_left_hand
            else (
                "accept_landmarks_ignore_label"
                if handedness_policy == "ignore_handedness" and left_observed
                else "accept"
            )
        )

        display_roi = None
        if selected is not None and bool(roi_config.get("enabled", True)):
            observed = roi_from_normalized_landmarks(
                selected.image_landmarks,
                width,
                height,
                float(roi_config["margin_ratio"]),
                int(roi_config["minimum_size_px"]),
            )
            display_roi = self.roi_tracker.update(observed)
        else:
            self.roi_tracker.mark_missed()
            display_roi = self.roi_tracker.current

        quality = evaluate_frame_quality(
            image,
            selected,
            self.config["quality"],
        )

        branch_start = time.perf_counter_ns()
        filtered = None
        raw_normalization = None
        filtered_normalization = None
        selected_normalization = None
        temporal_diagnostics: dict[str, Any] = {}
        feature_vectors: dict[str, Any] = {}
        notes: list[str] = []
        if detection_diagnostics.get("recovered"):
            notes.append(
                f"MediaPipeRecovery: {detection_diagnostics.get('selected_attempt', 'unknown')}"
            )
        if detection_diagnostics.get("handedness_retry_observation"):
            notes.append("MediaPipeHandednessRetryObservation: an earlier IMAGE pass returned effective left handedness.")
        if detection_diagnostics.get("left_observed_in_any_attempt"):
            notes.append("MediaPipeHandednessObservation: at least one detector attempt returned effective left handedness.")
        if handedness_policy == "ignore_handedness" and left_observed:
            notes.append(
                "HandednessPolicy[ignore_handedness]: raw/effective Left was retained diagnostically, "
                "but detected landmarks were accepted because the acquisition protocol guarantees one right hand."
            )

        if selected is not None:
            filtered = HandCandidate(
                selected.handedness_label,
                selected.handedness_score,
                self.image_filter.update(
                    selected.image_landmarks,
                    packet.relative_time_s,
                ),
                self.world_filter.update(
                    selected.world_landmarks,
                    packet.relative_time_s,
                ),
                selected.source_hand_index,
            )
            raw_normalization = self._normalize(selected, notes, "raw")
            filtered_normalization = self._normalize(filtered, notes, "filtered")

            if filtered_normalization is not None:
                temporal_diagnostics = self.geometry_monitor.update(
                    filtered.world_landmarks,
                    filtered_normalization,
                    packet.relative_time_s,
                )
                if temporal_diagnostics.get("temporal_geometry_outlier"):
                    reasons = ", ".join(
                        temporal_diagnostics.get("warning_reasons", [])
                    )
                    notes.append(f"TemporalGeometryWarning: {reasons}")

            feature_vectors = self._extract_features(
                raw_normalization,
                filtered_normalization,
                notes,
            )
            selected_normalization = (
                raw_normalization
                if self.classification_landmark_source == "raw"
                else filtered_normalization
            )
        elif self.reset_on_missing:
            self.world_filter.reset()
            self.image_filter.reset()
            self.geometry_monitor.reset()

        branch_end = time.perf_counter_ns()
        classification_branch = self._branch_key(
            self.classification_landmark_source,
            self.classification_scale,
        )

        if critical_left_hand:
            status = "critical_left_hand_detected"
            notes.append(
                "CriticalHandednessError: HandednessPolicy[reject_left]: MediaPipe returned effective left handedness; "
                "the frame is rejected from gesture classification and should map to NO_GESTURE/invalid input."
            )
        elif selected is None:
            status = "no_right_hand" if candidates else "no_hand"
        elif (
            selected_normalization is None
            and bool(self.config.get("normalization", {}).get("enabled", True))
        ):
            status = "degenerate_geometry"
        elif bool(quality["quality_pass"]):
            status = "valid"
        else:
            status = "low_quality"

        total_end = time.perf_counter_ns()
        return FrameAnalysis(
            frame_index=packet.frame_index,
            capture_timestamp_ns=packet.capture_timestamp_ns,
            relative_time_s=packet.relative_time_s,
            status=status,
            candidates=candidates,
            selected_right_hand=selected,
            filtered_right_hand=filtered,
            temporal_filter_name=self.filter_name,
            detection_diagnostics=detection_diagnostics,
            quality_metrics=quality,
            roi_xyxy=display_roi.as_tuple() if display_roi else None,
            inference_roi_xyxy=(
                inference_roi.as_tuple() if inference_roi else None
            ),
            timing_ms={
                "detection": (detection_end - detection_start) / 1e6,
                "branches_normalization_features": (
                    branch_end - branch_start
                )
                / 1e6,
                "total": (total_end - total_start) / 1e6,
            },
            notes=notes,
            normalization=selected_normalization,
            raw_normalization=raw_normalization,
            filtered_normalization=filtered_normalization,
            temporal_diagnostics=temporal_diagnostics,
            feature_vectors=feature_vectors,
            classification_feature_branch=classification_branch,
        )


    def reset_sequence(self) -> None:
        """Reset all sequence-local state before processing an unrelated recording."""
        self.roi_tracker.reset()
        self.world_filter.reset()
        self.image_filter.reset()
        self.geometry_monitor.reset()
        self.detector.start_new_sequence()

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "FramePipeline":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
