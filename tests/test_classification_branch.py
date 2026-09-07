from __future__ import annotations

from pathlib import Path

import numpy as np

from tsgr.config import load_default_config
from tsgr.types import FramePacket, HandCandidate


def synthetic_world() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    points[1:5] = [[-0.02, 0.02, 0.01], [-0.04, 0.04, 0.01], [-0.06, 0.06, 0.015], [-0.08, 0.08, 0.02]]
    for base, x in ((5, -0.03), (9, 0.0), (13, 0.025), (17, 0.05)):
        points[base] = [x, 0.05, 0.0]
        points[base + 1] = [x, 0.09, 0.002]
        points[base + 2] = [x, 0.125, 0.004]
        points[base + 3] = [x, 0.155, 0.006]
    return points


def image_landmarks() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[:, 0] = np.linspace(0.35, 0.65, 21)
    points[:, 1] = np.linspace(0.70, 0.25, 21)
    return points


class FakeDetector:
    def __init__(self, config: dict, model_path: Path) -> None:
        self.calls = 0

    def detect(self, image, timestamp_ms, image_dimensions=None, inference_roi=None):
        world = synthetic_world()
        if self.calls:
            world[4] += np.asarray([0.04, -0.02, 0.03])
        self.calls += 1
        return [
            HandCandidate(
                handedness_label="Right",
                handedness_score=0.99,
                image_landmarks=image_landmarks(),
                world_landmarks=world,
                source_hand_index=0,
            )
        ]

    def close(self) -> None:
        pass


def test_filtered_classifier_branch_is_selectable_and_distinct(monkeypatch) -> None:
    import tsgr.pipeline.frame_pipeline as frame_pipeline

    monkeypatch.setattr(frame_pipeline, "MediaPipeHandLandmarker", FakeDetector)
    config = load_default_config()
    config["temporal_filter"]["active"] = "ema"
    config["temporal_filter"]["filters"]["ema"]["alpha"] = 0.2
    config["classification_input"]["landmark_source"] = "filtered"
    config["classification_input"]["scale"] = "wrist_middle_mcp"

    image = np.full((240, 320, 3), 127, dtype=np.uint8)
    with frame_pipeline.FramePipeline(config, Path("unused.task")) as pipeline:
        pipeline.process(FramePacket(0, 0, 0.0, image, "test"))
        analysis = pipeline.process(FramePacket(1, 1, 1.0 / 30.0, image, "test"))

    assert analysis.classification_feature_branch == "filtered.wrist_middle_mcp"
    assert analysis.classification_feature_vector is analysis.feature_vectors["filtered.wrist_middle_mcp"]
    assert not np.allclose(
        analysis.feature_vectors["raw.wrist_middle_mcp"].values,
        analysis.feature_vectors["filtered.wrist_middle_mcp"].values,
    )
