"""Helpers for saving derivative frame images without changing source archives."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tsgr.preprocessing.roi import PixelROI, crop_image


def save_roi_crop(
    image_bgr: np.ndarray,
    roi_xyxy: tuple[int, int, int, int] | None,
    output_dir: str | Path,
    frame_index: int,
    png_compression: int = 3,
) -> Path | None:
    """Save the current hand ROI as a derivative PNG and return its path."""
    if roi_xyxy is None:
        return None
    roi = PixelROI(*roi_xyxy)
    if roi.width <= 0 or roi.height <= 0:
        return None
    output_path = Path(output_dir) / f"frame_{frame_index:08d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped = crop_image(image_bgr, roi)
    success = cv2.imwrite(
        str(output_path),
        cropped,
        [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)],
    )
    if not success:
        raise OSError(f"Could not save cropped frame: {output_path}")
    return output_path
