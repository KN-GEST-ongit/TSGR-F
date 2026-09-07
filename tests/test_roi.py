from __future__ import annotations

import numpy as np

from tsgr.preprocessing.roi import (
    PixelROI,
    remap_normalized_landmarks_to_full_frame,
    roi_from_normalized_landmarks,
)


def test_roi_contains_all_landmarks() -> None:
    landmarks = np.zeros((21, 3), dtype=float)
    landmarks[:, 0] = np.linspace(0.4, 0.6, 21)
    landmarks[:, 1] = np.linspace(0.3, 0.7, 21)
    roi = roi_from_normalized_landmarks(landmarks, 1000, 500, 0.2, 40)
    assert roi.x1 <= 400
    assert roi.x2 >= 600
    assert roi.y1 <= 150
    assert roi.y2 >= 350


def test_crop_landmarks_are_remapped_to_full_frame() -> None:
    crop_landmarks = np.zeros((21, 3), dtype=float)
    crop_landmarks[:, 0] = 0.5
    crop_landmarks[:, 1] = 0.5
    roi = PixelROI(100, 50, 300, 250)
    remapped = remap_normalized_landmarks_to_full_frame(
        crop_landmarks,
        roi,
        full_width=400,
        full_height=400,
    )
    assert np.allclose(remapped[:, 0], 0.5)
    assert np.allclose(remapped[:, 1], 0.375)
