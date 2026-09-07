"""Typed data structures used by the capture and landmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsgr.features.schema import FeatureVector
from tsgr.preprocessing.spatial_normalization import SpatialNormalizationResult


@dataclass(slots=True)
class FramePacket:
    """Single source frame with acquisition timing metadata."""

    frame_index: int
    capture_timestamp_ns: int
    relative_time_s: float
    image_bgr: np.ndarray
    source_id: str
    scheduled_time_s: float | None = None


@dataclass(slots=True)
class HandCandidate:
    """One hand candidate returned by the landmark detector."""

    handedness_label: str
    handedness_score: float
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    source_hand_index: int

    def validate(self) -> None:
        if self.image_landmarks.shape != (21, 3):
            raise ValueError(
                f"Expected image landmarks with shape (21, 3), got {self.image_landmarks.shape}."
            )
        if self.world_landmarks.shape != (21, 3):
            raise ValueError(
                f"Expected world landmarks with shape (21, 3), got {self.world_landmarks.shape}."
            )


@dataclass(slots=True)
class FrameAnalysis:
    """Complete result of one frame, including all feature-extraction branches."""

    frame_index: int
    capture_timestamp_ns: int
    relative_time_s: float
    status: str
    candidates: list[HandCandidate] = field(default_factory=list)
    selected_right_hand: HandCandidate | None = None
    filtered_right_hand: HandCandidate | None = None
    temporal_filter_name: str = "none"
    detection_diagnostics: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, float | bool | str | list[str]] = field(default_factory=dict)
    roi_xyxy: tuple[int, int, int, int] | None = None
    inference_roi_xyxy: tuple[int, int, int, int] | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    normalization: SpatialNormalizationResult | None = None
    raw_normalization: SpatialNormalizationResult | None = None
    filtered_normalization: SpatialNormalizationResult | None = None
    temporal_diagnostics: dict[str, Any] = field(default_factory=dict)
    feature_vectors: dict[str, FeatureVector] = field(default_factory=dict)
    classification_feature_branch: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def classification_feature_vector(self) -> FeatureVector | None:
        if self.classification_feature_branch is None:
            return None
        return self.feature_vectors.get(self.classification_feature_branch)

    def to_dict(self) -> dict[str, Any]:
        def candidate_to_dict(candidate: HandCandidate) -> dict[str, Any]:
            return {
                "handedness_label": candidate.handedness_label,
                "handedness_score": float(candidate.handedness_score),
                "source_hand_index": int(candidate.source_hand_index),
                "image_landmarks": candidate.image_landmarks.tolist(),
                "world_landmarks": candidate.world_landmarks.tolist(),
            }

        classification_vector = self.classification_feature_vector
        return {
            "frame_index": self.frame_index,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "relative_time_s": self.relative_time_s,
            "status": self.status,
            "candidates": [candidate_to_dict(item) for item in self.candidates],
            "selected_right_hand": (
                candidate_to_dict(self.selected_right_hand)
                if self.selected_right_hand is not None
                else None
            ),
            "filtered_right_hand": (
                candidate_to_dict(self.filtered_right_hand)
                if self.filtered_right_hand is not None
                else None
            ),
            "temporal_filter_name": self.temporal_filter_name,
            "detection_diagnostics": self.detection_diagnostics,
            "quality_metrics": self.quality_metrics,
            "roi_xyxy": list(self.roi_xyxy) if self.roi_xyxy else None,
            "inference_roi_xyxy": (
                list(self.inference_roi_xyxy) if self.inference_roi_xyxy else None
            ),
            "timing_ms": self.timing_ms,
            "notes": self.notes,
            "temporal_diagnostics": self.temporal_diagnostics,
            "normalization": (
                self.normalization.to_dict() if self.normalization is not None else None
            ),
            "raw_normalization": (
                self.raw_normalization.to_dict()
                if self.raw_normalization is not None
                else None
            ),
            "filtered_normalization": (
                self.filtered_normalization.to_dict()
                if self.filtered_normalization is not None
                else None
            ),
            "classification_feature_branch": self.classification_feature_branch,
            "classification_feature_vector": (
                classification_vector.to_dict(include_named_values=False)
                if classification_vector is not None
                else None
            ),
            "feature_vectors": {
                key: value.to_dict(include_named_values=False)
                for key, value in self.feature_vectors.items()
            },
        }
