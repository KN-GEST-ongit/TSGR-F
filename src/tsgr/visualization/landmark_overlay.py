"""Landmark overlays used for optional error-review exports."""
from __future__ import annotations

import re
from typing import Iterable

import cv2
import numpy as np

from tsgr.constants import HAND_CONNECTIONS, LANDMARK_NAMES


DERIVED_FEATURE_IDS = (
    "derived.thumb_tip_to_index_pip_distance",
    "derived.thumb_tip_to_index_dip_distance",
)


def _feature_landmark_indices(feature_id: str) -> tuple[int, ...]:
    name_to_index = {name: index for index, name in enumerate(LANDMARK_NAMES)}
    if feature_id.startswith("landmark."):
        parts = feature_id.split(".")
        if len(parts) >= 3 and parts[1] in name_to_index:
            return (name_to_index[parts[1]],)
    bend = {
        "bend.thumb.cmc": (0, 1, 2), "bend.thumb.mcp": (1, 2, 3), "bend.thumb.ip": (2, 3, 4),
        "bend.index.mcp": (0, 5, 6), "bend.index.pip": (5, 6, 7), "bend.index.dip": (6, 7, 8),
        "bend.middle.mcp": (0, 9, 10), "bend.middle.pip": (9, 10, 11), "bend.middle.dip": (10, 11, 12),
        "bend.ring.mcp": (0, 13, 14), "bend.ring.pip": (13, 14, 15), "bend.ring.dip": (14, 15, 16),
        "bend.pinky.mcp": (0, 17, 18), "bend.pinky.pip": (17, 18, 19), "bend.pinky.dip": (18, 19, 20),
    }
    if feature_id in bend:
        return bend[feature_id]
    if feature_id == "distance.tip.thumb_index":
        return (4, 8)
    if feature_id == "thumb_index.distal_joint_distance":
        return (3, 7)
    if feature_id == "thumb_index.proximal_joint_distance":
        return (2, 6)
    if feature_id == "thumb.tip_to_mcp_distance.index":
        return (4, 5)
    if feature_id == DERIVED_FEATURE_IDS[0]:
        return (4, 6)
    if feature_id == DERIVED_FEATURE_IDS[1]:
        return (4, 7)
    if feature_id.startswith("straightness."):
        finger = feature_id.split(".")[-1]
        endpoints = {"thumb": (1, 4), "index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}
        return endpoints.get(finger, ())
    match = re.match(r"thumb\.tip_to_mcp_distance\.(index|middle|ring|pinky)$", feature_id)
    if match:
        base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}[match.group(1)]
        return (4, base)
    return ()


def draw_landmarks_overlay(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    *,
    highlighted_features: Iterable[str] = (),
    highlighted_bone: tuple[int, int] | None = None,
    title: str = "",
) -> np.ndarray:
    """Draw the persisted hand skeleton and optional feature highlights."""
    output = image.copy()
    height, width = output.shape[:2]
    if landmarks is None:
        cv2.putText(
            output,
            "persisted raw image landmarks unavailable",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output
    points = np.column_stack(
        (
            np.clip(landmarks[:, 0] * width, 0, width - 1),
            np.clip(landmarks[:, 1] * height, 0, height - 1),
        )
    ).astype(int)
    highlighted: set[int] = set()
    highlight_pairs: list[tuple[int, int]] = []
    for feature_id in highlighted_features:
        indices = _feature_landmark_indices(str(feature_id))
        highlighted.update(indices)
        if len(indices) == 2:
            highlight_pairs.append((indices[0], indices[1]))
    for start, end in HAND_CONNECTIONS:
        cv2.line(output, tuple(points[start]), tuple(points[end]), (70, 220, 70), 2)
    for start, end in highlight_pairs:
        cv2.line(output, tuple(points[start]), tuple(points[end]), (0, 210, 255), 3)
    if highlighted_bone is not None:
        cv2.line(output, tuple(points[highlighted_bone[0]]), tuple(points[highlighted_bone[1]]), (0, 0, 255), 4)
        highlighted.update(highlighted_bone)
    for index, point in enumerate(points):
        color = (0, 210, 255) if index in highlighted else (40, 40, 240)
        radius = 6 if index in highlighted else 4
        cv2.circle(output, tuple(point), radius, color, -1)
        cv2.putText(
            output,
            str(index),
            (int(point[0] + 4), int(point[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if title:
        cv2.rectangle(output, (0, 0), (min(width, 1200), 38), (0, 0, 0), -1)
        cv2.putText(output, title[:150], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output
