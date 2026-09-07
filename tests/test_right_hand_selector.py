from __future__ import annotations

import numpy as np

from tsgr.detection.right_hand_selector import select_right_hand
from tsgr.types import HandCandidate


def candidate(label: str, score: float, index: int) -> HandCandidate:
    return HandCandidate(
        handedness_label=label,
        handedness_score=score,
        image_landmarks=np.zeros((21, 3)),
        world_landmarks=np.zeros((21, 3)),
        source_hand_index=index,
    )


def test_selects_highest_scoring_right_hand() -> None:
    selected = select_right_hand(
        [candidate("Left", 0.99, 0), candidate("Right", 0.70, 1), candidate("Right", 0.92, 2)]
    )
    assert selected is not None
    assert selected.source_hand_index == 2


def test_can_invert_handedness_labels() -> None:
    selected = select_right_hand([candidate("Left", 0.88, 0)], invert_labels=True)
    assert selected is not None
    assert selected.source_hand_index == 0
