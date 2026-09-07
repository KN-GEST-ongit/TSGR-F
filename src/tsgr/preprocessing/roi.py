"""Region-of-interest geometry and temporal tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PixelROI:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def roi_from_normalized_landmarks(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    margin_ratio: float,
    minimum_size_px: int,
) -> PixelROI:
    if landmarks.shape != (21, 3):
        raise ValueError("landmarks must have shape (21, 3).")
    x_values = landmarks[:, 0] * image_width
    y_values = landmarks[:, 1] * image_height
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    width = max(x_max - x_min, float(minimum_size_px))
    height = max(y_max - y_min, float(minimum_size_px))
    side = max(width, height)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half_size = side * (0.5 + margin_ratio)
    x1 = max(0, int(np.floor(center_x - half_size)))
    y1 = max(0, int(np.floor(center_y - half_size)))
    x2 = min(image_width, int(np.ceil(center_x + half_size)))
    y2 = min(image_height, int(np.ceil(center_y + half_size)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Computed ROI is empty.")
    return PixelROI(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_image(image_bgr: np.ndarray, roi: PixelROI) -> np.ndarray:
    return image_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2].copy()


def remap_normalized_landmarks_to_full_frame(
    landmarks: np.ndarray,
    roi: PixelROI,
    full_width: int,
    full_height: int,
) -> np.ndarray:
    if landmarks.shape != (21, 3):
        raise ValueError("landmarks must have shape (21, 3).")
    remapped = landmarks.astype(np.float64, copy=True)
    remapped[:, 0] = (roi.x1 + landmarks[:, 0] * roi.width) / full_width
    remapped[:, 1] = (roi.y1 + landmarks[:, 1] * roi.height) / full_height
    # MediaPipe image-space z is scaled similarly to x, so compensate for crop width.
    remapped[:, 2] = landmarks[:, 2] * (roi.width / full_width)
    return remapped


class ROITracker:
    """Maintain a smoothed hand ROI and periodically request full-frame reacquisition."""

    def __init__(
        self,
        smoothing_alpha: float,
        full_frame_interval: int,
        maximum_missed_frames: int,
    ) -> None:
        self.alpha = float(smoothing_alpha)
        self.full_frame_interval = max(1, int(full_frame_interval))
        self.maximum_missed_frames = max(0, int(maximum_missed_frames))
        self.current: PixelROI | None = None
        self.missed_frames = 0

    def should_use_roi(self, frame_index: int) -> bool:
        if self.current is None or self.missed_frames > 0:
            return False
        return frame_index % self.full_frame_interval != 0

    def update(self, observed: PixelROI) -> PixelROI:
        if self.current is None:
            self.current = observed
        else:
            alpha = self.alpha
            self.current = PixelROI(
                x1=round((1 - alpha) * self.current.x1 + alpha * observed.x1),
                y1=round((1 - alpha) * self.current.y1 + alpha * observed.y1),
                x2=round((1 - alpha) * self.current.x2 + alpha * observed.x2),
                y2=round((1 - alpha) * self.current.y2 + alpha * observed.y2),
            )
        self.missed_frames = 0
        return self.current

    def mark_missed(self) -> None:
        self.missed_frames += 1
        if self.missed_frames > self.maximum_missed_frames:
            self.current = None
            self.missed_frames = 0

    def reset(self) -> None:
        self.current = None
        self.missed_frames = 0
