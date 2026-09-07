from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tsgr.processing.image_outputs import save_roi_crop


def test_roi_crop_is_saved_with_expected_shape(tmp_path: Path) -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[20:80, 30:90] = 200
    output = save_roi_crop(image, (30, 20, 90, 80), tmp_path, frame_index=7)
    assert output is not None
    loaded = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == (60, 60, 3)
    assert int(loaded.mean()) == 200
