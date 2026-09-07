"""IMAGE-space geometric features used by the final selective routing model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsgr.evaluation.training_support import LandmarkArchiveCache, TrainingSamples


FINGERS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("thumb", (0, 1, 2, 3, 4)),
    ("index", (0, 5, 6, 7, 8)),
    ("middle", (0, 9, 10, 11, 12)),
    ("ring", (0, 13, 14, 15, 16)),
    ("pinky", (0, 17, 18, 19, 20)),
)
TIP_IDS = (4, 8, 12, 16, 20)
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CRITICAL_THUMB_INDEX = ((4, 8, "tip"), (4, 7, "dip"), (4, 6, "pip"), (4, 5, "mcp"))


@dataclass(slots=True)
class ImageFeatureSet:
    values: np.ndarray
    feature_ids: tuple[str, ...]
    availability: np.ndarray


def _angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u <= 1e-12 or norm_v <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))
    return math.acos(cosine) / math.pi


def image_features(landmarks: np.ndarray | None) -> tuple[list[float], list[str]]:
    """Extract the fixed IMAGE-2D feature schema from 21 hand landmarks."""
    feature_ids: list[str] = []
    if landmarks is None:
        return [], feature_ids
    xy = np.asarray(landmarks, dtype=np.float64)[:, :2]
    scale = float(np.linalg.norm(xy[0] - xy[9]))
    if not np.isfinite(scale) or scale <= 1e-12:
        return [], feature_ids
    values: list[float] = []

    for start, end, name in CRITICAL_THUMB_INDEX:
        feature_ids.append(f"img2d.distance.thumb_tip.index_{name}")
        values.append(float(np.linalg.norm(xy[start] - xy[end]) / scale))

    for i in range(len(TIP_IDS)):
        for j in range(i + 1, len(TIP_IDS)):
            feature_ids.append(f"img2d.distance.tip.{TIP_NAMES[i]}_{TIP_NAMES[j]}")
            values.append(float(np.linalg.norm(xy[TIP_IDS[i]] - xy[TIP_IDS[j]]) / scale))

    for tip, name in zip(TIP_IDS, TIP_NAMES):
        feature_ids.append(f"img2d.distance.wrist_tip.{name}")
        values.append(float(np.linalg.norm(xy[0] - xy[tip]) / scale))

    for name, chain in FINGERS[1:]:
        _, mcp, pip, dip, tip = chain
        for joint_name, a, b, c in (
            ("mcp", 0, mcp, pip),
            ("pip", mcp, pip, dip),
            ("dip", pip, dip, tip),
        ):
            feature_ids.append(f"img2d.bend.{name}.{joint_name}")
            values.append(_angle_2d(xy[a], xy[b], xy[c]))

    for joint_name, a, b, c in (("cmc", 0, 1, 2), ("mcp", 1, 2, 3), ("ip", 2, 3, 4)):
        feature_ids.append(f"img2d.bend.thumb.{joint_name}")
        values.append(_angle_2d(xy[a], xy[b], xy[c]))

    for name, chain in FINGERS:
        points = xy[np.asarray(chain, dtype=np.int64)]
        path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        endpoint = float(np.linalg.norm(points[-1] - points[0]))
        feature_ids.append(f"img2d.straightness.{name}")
        values.append(endpoint / path if path > 1e-12 else 0.0)

    return values, feature_ids


def build_image_feature_matrix(
    samples: TrainingSamples,
    *,
    run_map: dict[tuple[str, str, str], Path],
    cache: LandmarkArchiveCache,
) -> ImageFeatureSet:
    """Build the complete IMAGE-2D TRAIN matrix from persisted landmarks."""
    rows: list[list[float] | None] = []
    feature_ids: tuple[str, ...] | None = None
    available = np.zeros(len(samples.metadata), dtype=bool)
    for index, metadata in enumerate(samples.metadata):
        key = (
            str(metadata.get("gesture_id", "")).upper(),
            str(metadata.get("public_subject_id", "")).upper(),
            str(metadata.get("background", "")).upper(),
        )
        run = run_map.get(key)
        landmarks = cache.get(run, int(metadata.get("frame_index", -1)), "raw_image_landmarks") if run is not None else None
        values, ids = image_features(landmarks)
        if values:
            if feature_ids is None:
                feature_ids = tuple(ids)
            elif tuple(ids) != feature_ids:
                raise RuntimeError("Inconsistent image feature schema.")
            rows.append(values)
            available[index] = True
        else:
            rows.append(None)
    if feature_ids is None:
        raise RuntimeError("No persisted image landmarks were available for 2D fusion.")
    matrix = np.full((len(rows), len(feature_ids)), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if row is not None:
            matrix[index] = np.asarray(row, dtype=np.float64)
    if not np.all(available):
        missing = int(np.sum(~available))
        raise RuntimeError(f"Missing persisted IMAGE landmarks for {missing} training samples.")
    return ImageFeatureSet(matrix, feature_ids, available)
