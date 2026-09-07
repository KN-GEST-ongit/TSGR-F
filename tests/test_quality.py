from __future__ import annotations

import numpy as np

from tsgr.config import load_config
from tsgr.quality.frame_quality import evaluate_frame_quality
from tsgr.types import HandCandidate


def test_missing_hand_fails_quality() -> None:
    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    quality = evaluate_frame_quality(image, None, load_config()["quality"])
    assert quality["hand_present"] is False
    assert quality["quality_pass"] is False


def test_valid_synthetic_hand_produces_finite_score() -> None:
    image = np.indices((200, 200)).sum(axis=0).astype(np.uint8)
    image = np.repeat(image[:, :, None], 3, axis=2)
    landmarks = np.zeros((21, 3), dtype=float)
    landmarks[:, 0] = np.linspace(0.3, 0.7, 21)
    landmarks[:, 1] = np.linspace(0.2, 0.8, 21)
    hand = HandCandidate("Right", 0.95, landmarks, np.zeros((21, 3)), 0)
    quality = evaluate_frame_quality(image, hand, load_config()["quality"])
    assert 0.0 <= float(quality["quality_score"]) <= 1.0
