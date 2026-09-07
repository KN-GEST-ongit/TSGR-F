"""MediaPipe Tasks adapter with separate IMAGE and VIDEO detection policies."""

from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Any

import cv2
import numpy as np

from tsgr.detection.policy import DetectionPass, image_detection_passes, video_detection_pass, video_recovery_passes
from tsgr.detection.right_hand_selector import effective_handedness
from tsgr.preprocessing.roi import PixelROI, remap_normalized_landmarks_to_full_frame
from tsgr.types import HandCandidate


class MediaPipeUnavailableError(RuntimeError):
    """Raised when MediaPipe cannot be imported in the active environment."""


def _border_median_bgr(image_bgr: np.ndarray) -> tuple[int, int, int]:
    """Estimate a neutral padding color from the outer image border."""
    height, width = image_bgr.shape[:2]
    band = max(1, int(round(min(height, width) * 0.05)))
    samples = np.concatenate(
        (
            image_bgr[:band, :, :].reshape(-1, 3),
            image_bgr[-band:, :, :].reshape(-1, 3),
            image_bgr[:, :band, :].reshape(-1, 3),
            image_bgr[:, -band:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    median = np.median(samples, axis=0)
    return tuple(int(np.clip(round(value), 0, 255)) for value in median)


def pad_for_detection(image_bgr: np.ndarray, padding_ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad an image without resizing the hand and return (left, top, right, bottom)."""
    ratio = max(0.0, float(padding_ratio))
    if ratio <= 0.0:
        return image_bgr, (0, 0, 0, 0)
    height, width = image_bgr.shape[:2]
    pad_x = max(1, int(round(width * ratio)))
    pad_y = max(1, int(round(height * ratio)))
    color = _border_median_bgr(image_bgr)
    padded = cv2.copyMakeBorder(
        image_bgr,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, (pad_x, pad_y, pad_x, pad_y)


def remap_padded_landmarks_to_original(
    landmarks: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    padded_width: int,
    padded_height: int,
    padding: tuple[int, int, int, int],
) -> np.ndarray:
    """Map normalized IMAGE landmarks from a padded canvas to the original crop."""
    left, top, _, _ = padding
    output = np.asarray(landmarks, dtype=np.float64).copy()
    output[:, 0] = (output[:, 0] * padded_width - left) / max(1, original_width)
    output[:, 1] = (output[:, 1] * padded_height - top) / max(1, original_height)
    output[:, 2] = output[:, 2] * (padded_width / max(1, original_width))
    return output


class MediaPipeHandLandmarker:
    """Blocking MediaPipe Hand Landmarker with strict right-hand research diagnostics.

    IMAGE mode can use an ordered high-recall chain. VIDEO mode remains stateful and
    fast by default, with an explicitly enabled stateless IMAGE fallback after loss.
    """

    def __init__(self, config: dict[str, Any], model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {path}. Run 'tsgr-download-model'."
            )
        if bool(config.get("suppress_native_warnings", True)):
            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import mediapipe as mp
            try:
                from absl import logging as absl_logging
                absl_logging.set_verbosity(absl_logging.ERROR)
            except Exception:
                pass
        except ModuleNotFoundError as error:
            raise MediaPipeUnavailableError(
                "MediaPipe is not installed. Install requirements.txt in Python 3.12.10."
            ) from error

        self.mp = mp
        self.config = dict(config)
        self.model_path = path.resolve()
        self.running_mode = str(config.get("running_mode", "video")).strip().lower()
        if self.running_mode not in {"image", "video"}:
            raise ValueError("mediapipe.running_mode must be 'image' or 'video'.")
        self.invert_handedness_labels = bool(config.get("invert_handedness_labels", False))
        self.handedness_policy = str(config.get("handedness_policy", "reject_left")).strip().lower()
        if self.handedness_policy not in {"reject_left", "ignore_handedness"}:
            raise ValueError(
                "mediapipe.handedness_policy must be 'reject_left' or 'ignore_handedness'."
            )
        self.num_hands = int(config.get("num_hands", 2))
        self._landmarkers: dict[tuple[Any, ...], Any] = {}
        self._last_timestamp_ms = -1
        self._sequence_timestamp_offset_ms = 0
        self._consecutive_video_misses = 0
        self.last_diagnostics: dict[str, Any] = {}

        if self.running_mode == "video":
            profile = video_detection_pass(self.config)
            self._get_landmarker("video", profile)
        else:
            passes = image_detection_passes(self.config)
            self._get_landmarker("image", passes[0])

    def _landmarker_key(self, mode: str, spec: DetectionPass) -> tuple[Any, ...]:
        return (
            mode,
            round(spec.min_hand_detection_confidence, 6),
            round(spec.min_hand_presence_confidence, 6),
            round(spec.min_tracking_confidence, 6),
        )

    def _get_landmarker(self, mode: str, spec: DetectionPass) -> Any:
        key = self._landmarker_key(mode, spec)
        existing = self._landmarkers.get(key)
        if existing is not None:
            return existing
        running_mode = (
            self.mp.tasks.vision.RunningMode.IMAGE
            if mode == "image"
            else self.mp.tasks.vision.RunningMode.VIDEO
        )
        options = self.mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running_mode,
            num_hands=self.num_hands,
            min_hand_detection_confidence=float(spec.min_hand_detection_confidence),
            min_hand_presence_confidence=float(spec.min_hand_presence_confidence),
            min_tracking_confidence=float(spec.min_tracking_confidence),
        )
        landmarker = self.mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._landmarkers[key] = landmarker
        return landmarker

    def _result_to_candidates(
        self,
        result: Any,
        *,
        original_detection_size: tuple[int, int],
        padded_size: tuple[int, int] | None,
        padding: tuple[int, int, int, int],
        full_image_size: tuple[int, int] | None,
        roi: PixelROI | None,
    ) -> list[HandCandidate]:
        candidates: list[HandCandidate] = []
        count = min(
            len(result.hand_landmarks),
            len(result.hand_world_landmarks),
            len(result.handedness),
        )
        original_width, original_height = original_detection_size
        for hand_index in range(count):
            categories = result.handedness[hand_index]
            category = categories[0] if categories else None
            label = category.category_name if category is not None else "unknown"
            score = float(category.score) if category is not None else 0.0
            image_landmarks = np.asarray(
                [[point.x, point.y, point.z] for point in result.hand_landmarks[hand_index]],
                dtype=np.float64,
            )
            world_landmarks = np.asarray(
                [[point.x, point.y, point.z] for point in result.hand_world_landmarks[hand_index]],
                dtype=np.float64,
            )
            if padded_size is not None and any(padding):
                image_landmarks = remap_padded_landmarks_to_original(
                    image_landmarks,
                    original_width=original_width,
                    original_height=original_height,
                    padded_width=padded_size[0],
                    padded_height=padded_size[1],
                    padding=padding,
                )
            if roi is not None and full_image_size is not None:
                full_width, full_height = full_image_size
                image_landmarks = remap_normalized_landmarks_to_full_frame(
                    image_landmarks,
                    roi,
                    full_width,
                    full_height,
                )
            candidate = HandCandidate(
                handedness_label=label,
                handedness_score=score,
                image_landmarks=image_landmarks,
                world_landmarks=world_landmarks,
                source_hand_index=hand_index,
            )
            candidate.validate()
            candidates.append(candidate)
        return candidates

    def _effective_labels(self, candidates: list[HandCandidate]) -> list[str]:
        return [
            effective_handedness(candidate.handedness_label, self.invert_handedness_labels)
            for candidate in candidates
        ]

    def _has_right(self, candidates: list[HandCandidate]) -> bool:
        return "right" in self._effective_labels(candidates)

    def _has_acceptable_hand(self, candidates: list[HandCandidate]) -> bool:
        """Return whether the current handedness policy accepts any candidate."""
        if self.handedness_policy == "ignore_handedness":
            return bool(candidates)
        return self._has_right(candidates)

    def _attempt_record(
        self,
        spec: DetectionPass,
        candidates: list[HandCandidate],
        elapsed_ms: float,
        *,
        mode: str,
    ) -> dict[str, Any]:
        return {
            "name": spec.name,
            "mode": mode,
            "min_hand_detection_confidence": spec.min_hand_detection_confidence,
            "min_hand_presence_confidence": spec.min_hand_presence_confidence,
            "min_tracking_confidence": spec.min_tracking_confidence,
            "padding_ratio": spec.padding_ratio,
            "candidate_count": len(candidates),
            "raw_handedness_labels": [item.handedness_label for item in candidates],
            "effective_handedness_labels": self._effective_labels(candidates),
            "handedness_scores": [float(item.handedness_score) for item in candidates],
            "right_detected": self._has_right(candidates),
            "acceptable_hand_detected": self._has_acceptable_hand(candidates),
            "handedness_policy": self.handedness_policy,
            "elapsed_ms": float(elapsed_ms),
        }

    def _detect_image_chain(
        self,
        image_bgr: np.ndarray,
        *,
        full_image_size: tuple[int, int] | None,
        roi: PixelROI | None,
        profile_name: str | None = None,
        context: str = "image",
    ) -> tuple[list[HandCandidate], dict[str, Any]]:
        passes = image_detection_passes(self.config, profile_name=profile_name)
        original_height, original_width = image_bgr.shape[:2]
        attempts: list[dict[str, Any]] = []
        best_nonright: list[HandCandidate] = []
        best_nonright_score = -1.0
        selected_candidates: list[HandCandidate] = []
        selected_name = ""

        for spec in passes:
            padded, padding = pad_for_detection(image_bgr, spec.padding_ratio)
            padded_height, padded_width = padded.shape[:2]
            image_rgb = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
            landmarker = self._get_landmarker("image", spec)
            started = time.perf_counter_ns()
            result = landmarker.detect(mp_image)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            candidates = self._result_to_candidates(
                result,
                original_detection_size=(original_width, original_height),
                padded_size=(padded_width, padded_height),
                padding=padding,
                full_image_size=full_image_size,
                roi=roi,
            )
            attempts.append(self._attempt_record(spec, candidates, elapsed_ms, mode="image"))
            if self._has_acceptable_hand(candidates):
                selected_candidates = candidates
                selected_name = spec.name
                break
            score = max((item.handedness_score for item in candidates), default=-1.0)
            if score > best_nonright_score:
                best_nonright = candidates
                best_nonright_score = score

        if not selected_candidates:
            selected_candidates = best_nonright
        effective = self._effective_labels(selected_candidates)
        earlier_left = any(
            "left" in attempt.get("effective_handedness_labels", [])
            for attempt in attempts[:-1]
        )
        any_left = any(
            "left" in attempt.get("effective_handedness_labels", [])
            for attempt in attempts
        )
        if "right" in effective and "left" in effective:
            outcome = "right_with_left_candidate"
        elif "right" in effective:
            outcome = "right_detected"
        elif "left" in effective and self.handedness_policy == "ignore_handedness":
            outcome = "left_accepted_by_ignore_handedness"
        elif "left" in effective:
            outcome = "left_only"
        elif selected_candidates and self.handedness_policy == "ignore_handedness":
            outcome = "hand_accepted_by_ignore_handedness"
        elif selected_candidates:
            outcome = "nonright_unknown"
        else:
            outcome = "no_hand"
        diagnostics = {
            "running_mode": "image",
            "context": context,
            "profile": str(profile_name or self.config.get("image_detection", {}).get("active_profile", "high_recall")),
            "strategy": "ordered_retry_chain",
            "attempt_count": len(attempts),
            "attempts": attempts,
            "selected_attempt": selected_name,
            "recovered": bool(selected_name and selected_name != passes[0].name),
            "final_outcome": outcome,
            "handedness_retry_observation": bool(earlier_left),
            "left_observed_in_any_attempt": bool(any_left),
            "handedness_policy": self.handedness_policy,
            "accepted_by_handedness_policy": bool(self._has_acceptable_hand(selected_candidates)),
        }
        return selected_candidates, diagnostics

    def _detect_video_once(
        self,
        image_bgr: np.ndarray,
        timestamp_ms: int,
        *,
        full_image_size: tuple[int, int] | None,
        roi: PixelROI | None,
    ) -> tuple[list[HandCandidate], dict[str, Any]]:
        spec = video_detection_pass(self.config)
        adjusted_timestamp = int(timestamp_ms) + self._sequence_timestamp_offset_ms
        if adjusted_timestamp <= self._last_timestamp_ms:
            adjusted_timestamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = adjusted_timestamp
        image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        landmarker = self._get_landmarker("video", spec)
        started = time.perf_counter_ns()
        result = landmarker.detect_for_video(mp_image, adjusted_timestamp)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        height, width = image_bgr.shape[:2]
        candidates = self._result_to_candidates(
            result,
            original_detection_size=(width, height),
            padded_size=None,
            padding=(0, 0, 0, 0),
            full_image_size=full_image_size,
            roi=roi,
        )
        return candidates, self._attempt_record(spec, candidates, elapsed_ms, mode="video")

    def detect(
        self,
        image_bgr: np.ndarray,
        timestamp_ms: int,
        full_image_size: tuple[int, int] | None = None,
        roi: PixelROI | None = None,
    ) -> list[HandCandidate]:
        if self.running_mode == "image":
            candidates, diagnostics = self._detect_image_chain(
                image_bgr,
                full_image_size=full_image_size,
                roi=roi,
            )
            self.last_diagnostics = diagnostics
            return candidates

        video_candidates, video_attempt = self._detect_video_once(
            image_bgr,
            timestamp_ms,
            full_image_size=full_image_size,
            roi=roi,
        )
        right_detected = self._has_right(video_candidates)
        acceptable_detected = self._has_acceptable_hand(video_candidates)
        if acceptable_detected:
            self._consecutive_video_misses = 0
        else:
            self._consecutive_video_misses += 1

        recovery = self.config.get("video_recovery", {})
        recovery_enabled = bool(recovery.get("enabled", False))
        recovery_after = max(1, int(recovery.get("after_consecutive_failures", 1)))
        diagnostics: dict[str, Any] = {
            "running_mode": "video",
            "context": "video",
            "profile": str(self.config.get("video_detection", {}).get("active_profile", "balanced")),
            "strategy": "stateful_video_tracking",
            "attempt_count": 1,
            "attempts": [video_attempt],
            "selected_attempt": video_attempt["name"] if acceptable_detected else "",
            "recovered": False,
            "video_recovery_enabled": recovery_enabled,
            "video_recovery_used": False,
            "consecutive_video_misses": self._consecutive_video_misses,
            "final_outcome": (
                "right_detected"
                if right_detected
                else (
                    "left_accepted_by_ignore_handedness"
                    if acceptable_detected and "left" in self._effective_labels(video_candidates)
                    else ("left_only" if "left" in self._effective_labels(video_candidates) else "no_hand")
                )
            ),
            "left_observed_in_any_attempt": bool(
                "left" in video_attempt.get("effective_handedness_labels", [])
            ),
            "handedness_policy": self.handedness_policy,
            "accepted_by_handedness_policy": bool(acceptable_detected),
        }
        if acceptable_detected or not recovery_enabled or self._consecutive_video_misses < recovery_after:
            self.last_diagnostics = diagnostics
            return video_candidates

        recovery_profile = str(recovery.get("image_profile", "high_recall"))
        recovery_candidates, recovery_diag = self._detect_image_chain(
            image_bgr,
            full_image_size=full_image_size,
            roi=roi,
            profile_name=recovery_profile,
            context="video_recovery",
        )
        diagnostics["attempts"].extend(recovery_diag["attempts"])
        diagnostics["attempt_count"] = len(diagnostics["attempts"])
        diagnostics["video_recovery_used"] = True
        diagnostics["video_recovery_profile"] = recovery_profile
        diagnostics["video_recovery_diagnostics"] = recovery_diag
        diagnostics["left_observed_in_any_attempt"] = bool(
            diagnostics.get("left_observed_in_any_attempt", False)
            or recovery_diag.get("left_observed_in_any_attempt", False)
        )
        if self._has_acceptable_hand(recovery_candidates):
            diagnostics["selected_attempt"] = f"video_recovery:{recovery_diag.get('selected_attempt', '')}"
            diagnostics["recovered"] = True
            diagnostics["final_outcome"] = (
                "right_recovered_after_video_loss"
                if self._has_right(recovery_candidates)
                else "hand_recovered_ignore_handedness"
            )
            self._consecutive_video_misses = 0
            self.last_diagnostics = diagnostics
            return recovery_candidates

        if recovery_candidates and not video_candidates:
            final_candidates = recovery_candidates
        else:
            final_candidates = video_candidates
        effective = self._effective_labels(final_candidates)
        if final_candidates and self.handedness_policy == "ignore_handedness":
            diagnostics["final_outcome"] = (
                "left_accepted_by_ignore_handedness" if "left" in effective else "hand_accepted_by_ignore_handedness"
            )
            diagnostics["accepted_by_handedness_policy"] = True
        else:
            diagnostics["final_outcome"] = "left_only" if "left" in effective else "no_hand"
        self.last_diagnostics = diagnostics
        return final_candidates

    def update_runtime_config(self, config: dict[str, Any]) -> None:
        """Update per-sequence settings that do not require rebuilding the model graph."""
        requested_mode = str(config.get("running_mode", self.running_mode)).strip().lower()
        if requested_mode != self.running_mode:
            raise ValueError(
                f"Cannot change MediaPipe running mode from {self.running_mode!r} to {requested_mode!r} "
                "inside a reused pipeline. Create a new FramePipeline instead."
            )
        self.config = dict(config)
        self.invert_handedness_labels = bool(config.get("invert_handedness_labels", False))
        self.handedness_policy = str(config.get("handedness_policy", "reject_left")).strip().lower()
        if self.handedness_policy not in {"reject_left", "ignore_handedness"}:
            raise ValueError(
                "mediapipe.handedness_policy must be 'reject_left' or 'ignore_handedness'."
            )

    def start_new_sequence(self, gap_ms: int = 1000) -> None:
        """Reset sequence-local diagnostics and preserve VIDEO timestamp monotonicity."""
        self._consecutive_video_misses = 0
        self.last_diagnostics = {}
        if getattr(self, "running_mode", "video") != "video":
            return
        gap = max(1, int(gap_ms))
        if self._last_timestamp_ms < 0:
            self._sequence_timestamp_offset_ms = 0
        else:
            self._sequence_timestamp_offset_ms = self._last_timestamp_ms + gap

    def close(self) -> None:
        """Close all native MediaPipe tasks exactly once.

        Clear the registry before invoking native ``close()`` so a repeated wrapper
        close cannot dispatch the same native handle twice.  Worker processes call
        this method explicitly before Python begins shutting down concurrent-futures
        infrastructure.
        """
        landmarkers = list(self._landmarkers.values())
        self._landmarkers.clear()
        first_error: BaseException | None = None
        for landmarker in landmarkers:
            try:
                landmarker.close()
            except BaseException as error:  # Close any remaining native handles.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "MediaPipeHandLandmarker":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
