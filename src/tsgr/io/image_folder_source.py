"""Frame source for naturally sorted image folders."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from tsgr.types import FramePacket
from tsgr.utils.natural_sort import natural_key


class ImageFolderSource:
    """Yield images from a directory using a synthetic frame rate."""

    def __init__(
        self,
        folder: str | Path,
        fps: float,
        extensions: list[str] | tuple[str, ...],
        recursive: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {self.folder}")
        normalized_extensions = {item.lower() for item in extensions}
        iterator = self.folder.rglob("*") if recursive else self.folder.glob("*")
        self.files = sorted(
            (path for path in iterator if path.is_file() and path.suffix.lower() in normalized_extensions),
            key=natural_key,
        )
        if not self.files:
            raise FileNotFoundError(f"No supported images found in: {self.folder}")
        self.fps = float(fps)
        self.source_id = f"image_folder:{self.folder.resolve()}"

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[FramePacket]:
        start_ns = time.perf_counter_ns()
        for frame_index, path in enumerate(self.files):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"OpenCV could not read image: {path}")
            relative_time_s = frame_index / self.fps
            yield FramePacket(
                frame_index=frame_index,
                capture_timestamp_ns=start_ns + int(relative_time_s * 1_000_000_000),
                relative_time_s=relative_time_s,
                image_bgr=image,
                source_id=self.source_id,
                scheduled_time_s=relative_time_s,
            )
