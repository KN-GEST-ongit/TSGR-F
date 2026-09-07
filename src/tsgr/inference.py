"""High-level single-image inference API for the final TSGR-F pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tsgr.config import load_config
from tsgr.evaluation.experiment_evaluator import _prediction_from_missing
from tsgr.evaluation.frozen_end_to_end_v42 import (
    _acceptance_thresholds_for_predictions,
    _load_acceptance,
    predict_frozen_detailed,
)
from tsgr.evaluation.frozen_routing_v40 import _load_model
from tsgr.evaluation.image_feature_support import image_features
from tsgr.pipeline.frame_pipeline import FramePipeline
from tsgr.types import FramePacket


@dataclass(frozen=True, slots=True)
class TSGRFImagePrediction:
    """One operational TSGR-F decision for a single image."""

    predicted_state: str
    candidate_gesture: str | None
    accepted: bool
    route: str | None
    acceptance_score: float | None
    acceptance_threshold: float | None
    status: str
    reason: str
    top_gestures: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        prefix = "GESTURE_"
        if self.predicted_state.startswith(prefix):
            return self.predicted_state[len(prefix) :]
        return self.predicted_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "predicted_state": self.predicted_state,
            "candidate_gesture": self.candidate_gesture,
            "accepted": self.accepted,
            "route": self.route,
            "acceptance_score": self.acceptance_score,
            "acceptance_threshold": self.acceptance_threshold,
            "status": self.status,
            "reason": self.reason,
            "top_gestures": list(self.top_gestures),
        }


class TSGRFImageRecognizer:
    """Run the final fixed-routing and TRAIN-only acceptance pipeline on images."""

    def __init__(
        self,
        *,
        mediapipe_model: str | Path,
        routing_model_root: str | Path,
        acceptance_root: str | Path,
        scenario: str,
        fold_id: str,
        config_path: str | Path | None = None,
        detection_profile: str = "high_recall",
        handedness_policy: str = "ignore_handedness",
    ) -> None:
        if detection_profile not in {"fast", "balanced", "high_recall"}:
            raise ValueError("detection_profile must be fast, balanced, or high_recall.")
        if handedness_policy not in {"reject_left", "ignore_handedness"}:
            raise ValueError("handedness_policy must be reject_left or ignore_handedness.")

        self.scenario = str(scenario)
        self.fold_id = str(fold_id)
        self.routing_model_root = Path(routing_model_root)
        self.acceptance_root = Path(acceptance_root)

        self.model, self.arrays = _load_model(
            self.routing_model_root,
            self.scenario,
            self.fold_id,
        )
        model_path = self.routing_model_root / self.scenario / self.fold_id / "model.json"
        self.acceptance = _load_acceptance(
            self.acceptance_root,
            self.scenario,
            self.fold_id,
            model_path,
        )
        self.gestures = tuple(str(value) for value in self.model["gesture_ids"])
        self.expected_image_feature_ids = tuple(
            str(value) for value in np.asarray(self.arrays["image_feature_ids"]).tolist()
        )

        config = load_config(
            config_path,
            overrides={
                "mediapipe": {
                    "running_mode": "image",
                    "handedness_policy": handedness_policy,
                    "image_detection": {"active_profile": detection_profile},
                    "video_recovery": {
                        "enabled": False,
                        "after_consecutive_failures": 1,
                        "image_profile": "high_recall",
                    },
                },
                "roi": {"inference_enabled": False},
                "temporal_filter": {"active": "none"},
            },
        )
        self.pipeline = FramePipeline(config, mediapipe_model)
        self._closed = False

    def predict(self, image_path: str | Path) -> TSGRFImagePrediction:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode input image: {path}")
        return self.predict_bgr(image, source_id=str(path))

    def predict_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        source_id: str = "image",
    ) -> TSGRFImagePrediction:
        if self._closed:
            raise RuntimeError("TSGRFImageRecognizer is closed.")
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must have shape (height, width, 3).")
        if image.size == 0:
            raise ValueError("image_bgr cannot be empty.")

        self.pipeline.reset_sequence()
        analysis = self.pipeline.process(
            FramePacket(
                frame_index=0,
                capture_timestamp_ns=time.time_ns(),
                relative_time_s=0.0,
                image_bgr=image,
                source_id=str(source_id),
            )
        )

        feature_vector = analysis.feature_vectors.get(str(self.model["branch_name"]))
        q2_values: list[float] = []
        q2_ids: list[str] = []
        if analysis.selected_right_hand is not None:
            q2_values, q2_ids = image_features(analysis.selected_right_hand.image_landmarks)

        input_ok = bool(
            feature_vector is not None
            and np.isfinite(feature_vector.values).all()
            and q2_values
            and tuple(q2_ids) == self.expected_image_feature_ids
            and np.isfinite(np.asarray(q2_values, dtype=np.float64)).all()
        )
        if not input_ok:
            predicted_state, reason = _prediction_from_missing(analysis.status)
            return TSGRFImagePrediction(
                predicted_state=predicted_state,
                candidate_gesture=None,
                accepted=False,
                route=None,
                acceptance_score=None,
                acceptance_threshold=None,
                status=str(analysis.status),
                reason=reason,
            )

        qfull = np.asarray(feature_vector.values, dtype=np.float64)[None, :]
        q2 = np.asarray(q2_values, dtype=np.float64)[None, :]
        detailed = predict_frozen_detailed(self.model, self.arrays, qfull, q2)
        thresholds = _acceptance_thresholds_for_predictions(
            self.acceptance,
            self.gestures,
            detailed["route"],
            detailed["final"],
        )

        final_index = int(detailed["final"][0])
        candidate = self.gestures[final_index]
        route = str(detailed["route"][0])
        score = float(detailed["acceptance_score"][0])
        threshold = float(thresholds[0])
        accepted = bool(score <= threshold)
        order = np.asarray(detailed["global_order"][0], dtype=np.int64)
        top = tuple(self.gestures[int(index)] for index in order[:3])

        return TSGRFImagePrediction(
            predicted_state=f"GESTURE_{candidate}" if accepted else "NO_GESTURE",
            candidate_gesture=candidate,
            accepted=accepted,
            route=route,
            acceptance_score=score,
            acceptance_threshold=threshold,
            status=str(analysis.status),
            reason="accepted_fixed_route" if accepted else "rejected_by_train_only_acceptance",
            top_gestures=top,
        )

    def close(self) -> None:
        if not self._closed:
            self.pipeline.close()
            self._closed = True

    def __enter__(self) -> "TSGRFImageRecognizer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["TSGRFImagePrediction", "TSGRFImageRecognizer"]
