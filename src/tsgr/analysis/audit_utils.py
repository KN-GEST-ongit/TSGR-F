"""Shared helpers for exact MediaPipe-failure and outlier review artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
from tsgr.visualization.matplotlib_backend import configure_headless_backend

configure_headless_backend()

import matplotlib.pyplot as plt
import numpy as np

from tsgr.constants import HAND_CONNECTIONS


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file while ignoring empty lines."""
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {error}") from error
            if isinstance(item, dict):
                rows.append(item)
    return rows


def result_map(run_dir: str | Path | None) -> dict[int, dict[str, Any]]:
    """Return per-frame results indexed by canonical frame index."""
    if run_dir is None:
        return {}
    rows = read_jsonl(Path(run_dir) / "frame_results.jsonl")
    return {int(row.get("frame_index", -1)): row for row in rows if "frame_index" in row}


def mediapipe_failure_reason(result: dict[str, Any] | None) -> str:
    """Describe why one image/frame did not yield a usable right-hand feature vector."""
    if result is None:
        return "not_processed"
    status = str(result.get("status", "unknown"))
    if status in {"no_hand", "no_right_hand", "critical_left_hand_detected", "degenerate_geometry"}:
        return status
    vector = result.get("classification_feature_vector")
    if not isinstance(vector, dict):
        return "missing_feature_vector"
    if not bool(vector.get("complete", False)):
        warnings = vector.get("warnings") or []
        detail = ";".join(str(item) for item in warnings)
        return "incomplete_feature_vector" + (f":{detail}" if detail else "")
    return ""


def _selected_landmarks(result: dict[str, Any] | None) -> np.ndarray | None:
    if result is None:
        return None
    selected = result.get("selected_right_hand")
    if not isinstance(selected, dict):
        return None
    landmarks = selected.get("image_landmarks")
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    return points


def draw_result_overlay(image_bgr: np.ndarray, result: dict[str, Any] | None) -> np.ndarray:
    """Draw the selected MediaPipe hand and concise status information."""
    output = image_bgr.copy()
    height, width = output.shape[:2]
    points = _selected_landmarks(result)
    if points is not None:
        pixels = np.column_stack(
            (
                np.clip(points[:, 0] * width, 0, width - 1),
                np.clip(points[:, 1] * height, 0, height - 1),
            )
        ).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, tuple(pixels[start]), tuple(pixels[end]), (70, 220, 70), 2)
        for index, point in enumerate(pixels):
            cv2.circle(output, tuple(point), 3, (40, 40, 240), -1)
            cv2.putText(
                output,
                str(index),
                (int(point[0] + 3), int(point[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    status = "not_processed" if result is None else str(result.get("status", "unknown"))
    reason = mediapipe_failure_reason(result)
    lines = [f"status={status}", f"failure={reason or 'none'}"]
    quality = result.get("quality_metrics", {}) if result else {}
    if isinstance(quality, dict):
        warnings = quality.get("warning_reasons") or []
        if warnings:
            lines.append("quality=" + ",".join(str(value) for value in warnings))
    box_height = 10 + 22 * len(lines)
    cv2.rectangle(output, (0, 0), (min(width, 900), box_height), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(output, text, (8, 20 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save_review_pair(
    source_image: str | Path,
    destination: str | Path,
    *,
    result: dict[str, Any] | None = None,
    header_lines: Iterable[str] = (),
) -> Path | None:
    """Save a side-by-side original/overlay image with textual review metadata."""
    source = Path(source_image)
    if not source.is_file():
        return None
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        return None
    overlay = draw_result_overlay(image, result)
    height = max(image.shape[0], overlay.shape[0])
    left = cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    right = cv2.copyMakeBorder(overlay, 0, height - overlay.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    combined = np.hstack((left, right))
    lines = [str(line) for line in header_lines if str(line).strip()]
    if lines:
        header_height = 12 + 22 * len(lines)
        header = np.full((header_height, combined.shape[1], 3), 255, dtype=np.uint8)
        for index, text in enumerate(lines):
            cv2.putText(header, text[:180], (8, 20 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        combined = np.vstack((header, combined))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), combined):
        return None
    return target


def copy_source_file(source: str | Path, destination: str | Path) -> Path:
    """Copy a source artifact for manual inspection without altering its bytes."""
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return target


def make_contact_sheets(
    records: list[dict[str, Any]],
    *,
    image_key: str,
    title_key: str,
    output_dir: str | Path,
    prefix: str,
    per_page: int = 12,
) -> list[Path]:
    """Create deterministic contact sheets from review-image paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    for page_index, start in enumerate(range(0, len(records), per_page), start=1):
        chunk = records[start : start + per_page]
        columns = 3
        rows = int(np.ceil(len(chunk) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, record in zip(axes.flat, chunk):
            image = cv2.imread(str(record[image_key]), cv2.IMREAD_COLOR)
            if image is not None:
                axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title(str(record.get(title_key, "")), fontsize=8, wrap=True)
            axis.axis("off")
        fig.suptitle(prefix.replace("_", " "), fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = output / f"{prefix}_page_{page_index:03d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pages.append(path)
    return pages
