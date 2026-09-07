from __future__ import annotations

from pathlib import Path

import numpy as np

from tsgr.analysis.audit_utils import mediapipe_failure_reason
from tsgr.config import load_config
from tsgr.detection.mediapipe_hand_landmarker import (
    pad_for_detection,
    remap_padded_landmarks_to_original,
)
from tsgr.detection.policy import image_detection_passes, video_detection_pass
from tsgr.detection.right_hand_selector import contains_effective_left_hand
from tsgr.pipeline import frame_pipeline as frame_pipeline_module
from tsgr.types import FramePacket, HandCandidate


def _candidate(label: str, score: float = 0.9) -> HandCandidate:
    image = np.zeros((21, 3), dtype=float)
    world = np.zeros((21, 3), dtype=float)
    return HandCandidate(label, score, image, world, 0)


def test_default_image_high_recall_chain_has_four_ordered_passes() -> None:
    config = load_config()["mediapipe"]
    passes = image_detection_passes(config)
    assert [item.name for item in passes] == [
        "standard",
        "relaxed",
        "padded_relaxed",
        "padded_max_recall",
    ]
    assert passes[0].min_hand_detection_confidence == 0.5
    assert passes[-1].min_hand_detection_confidence == 0.25
    assert passes[-1].padding_ratio == 0.25


def test_video_profiles_expose_balanced_and_high_recall_thresholds() -> None:
    config = load_config()["mediapipe"]
    balanced = video_detection_pass(config, "balanced")
    high_recall = video_detection_pass(config, "high_recall")
    assert balanced.min_hand_detection_confidence == 0.5
    assert high_recall.min_hand_detection_confidence == 0.35
    assert high_recall.min_hand_presence_confidence == 0.65
    assert high_recall.min_tracking_confidence == 0.65


def test_padding_keeps_original_hand_geometry_coordinate_mapping() -> None:
    image = np.full((200, 200, 3), 127, dtype=np.uint8)
    padded, padding = pad_for_detection(image, 0.25)
    assert padded.shape[:2] == (300, 300)
    points = np.zeros((21, 3), dtype=float)
    points[:, 0] = (50 + 100) / 300
    points[:, 1] = (50 + 60) / 300
    remapped = remap_padded_landmarks_to_original(
        points,
        original_width=200,
        original_height=200,
        padded_width=300,
        padded_height=300,
        padding=padding,
    )
    assert np.allclose(remapped[:, 0], 0.5)
    assert np.allclose(remapped[:, 1], 0.3)


def test_effective_left_hand_is_explicitly_detected() -> None:
    assert contains_effective_left_hand([_candidate("Left")], invert_labels=False)
    assert not contains_effective_left_hand([_candidate("Left")], invert_labels=True)


def test_critical_left_hand_status_is_reported_by_frame_pipeline(monkeypatch, tmp_path: Path) -> None:
    class FakeDetector:
        def __init__(self, config, model_path):
            self.last_diagnostics = {
                "running_mode": "image",
                "profile": "high_recall",
                "selected_attempt": "padded_max_recall",
                "final_outcome": "left_only",
            }

        def detect(self, *args, **kwargs):
            return [_candidate("Left")]

        def start_new_sequence(self, gap_ms: int = 1000):
            return None

        def update_runtime_config(self, config):
            return None

        def close(self):
            return None

    monkeypatch.setattr(frame_pipeline_module, "MediaPipeHandLandmarker", FakeDetector)
    config = load_config(overrides={"mediapipe": {"running_mode": "image"}})
    pipeline = frame_pipeline_module.FramePipeline(config, tmp_path / "unused.task")
    packet = FramePacket(0, 0, 0.0, np.zeros((200, 200, 3), dtype=np.uint8), "test")
    result = pipeline.process(packet)
    assert result.status == "critical_left_hand_detected"
    assert mediapipe_failure_reason(result.to_dict()) == "critical_left_hand_detected"


def test_default_video_recovery_is_explicitly_disabled() -> None:
    config = load_config()["mediapipe"]
    assert config["running_mode"] == "video"
    assert config["video_recovery"]["enabled"] is False


def test_image_and_video_modes_remain_distinct_in_configuration() -> None:
    image = load_config(overrides={"mediapipe": {"running_mode": "image"}})
    video = load_config(overrides={"mediapipe": {"running_mode": "video"}})
    assert image["mediapipe"]["running_mode"] == "image"
    assert video["mediapipe"]["running_mode"] == "video"


def test_any_left_handedness_observation_is_critical_even_if_final_retry_recovers_right(monkeypatch, tmp_path: Path) -> None:
    class FakeDetector:
        def __init__(self, config, model_path):
            self.last_diagnostics = {
                "running_mode": "image",
                "profile": "high_recall",
                "selected_attempt": "padded_relaxed",
                "final_outcome": "right_detected",
                "recovered": True,
                "left_observed_in_any_attempt": True,
                "handedness_retry_observation": True,
            }

        def detect(self, *args, **kwargs):
            return [_candidate("Right")]

        def start_new_sequence(self, gap_ms: int = 1000):
            return None

        def update_runtime_config(self, config):
            return None

        def close(self):
            return None

    monkeypatch.setattr(frame_pipeline_module, "MediaPipeHandLandmarker", FakeDetector)
    config = load_config(overrides={"mediapipe": {"running_mode": "image"}})
    pipeline = frame_pipeline_module.FramePipeline(config, tmp_path / "unused.task")
    packet = FramePacket(0, 0, 0.0, np.zeros((200, 200, 3), dtype=np.uint8), "test")
    result = pipeline.process(packet)
    assert result.selected_right_hand is not None
    assert result.status == "critical_left_hand_detected"
    assert any("CriticalHandednessError" in note for note in result.notes)
