"""Compute the complete TSGR-F single-frame feature vector from normalized landmarks."""

from __future__ import annotations

import math

import numpy as np

from tsgr.features.schema import FEATURE_COUNT, FINGER_NAMES, FINGER_TIPS, PAIR_NAMES, FeatureVector
from tsgr.preprocessing.spatial_normalization import SpatialNormalizationResult


class FeatureExtractionError(ValueError):
    """Raised when a feature vector cannot be constructed from a normalization result."""


def _angle_between(first: np.ndarray, second: np.ndarray, warnings: list[str], name: str) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        warnings.append(f"degenerate_vector:{name}")
        return float("nan")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def _joint_angle(points: np.ndarray, a: int, b: int, c: int, warnings: list[str], name: str) -> float:
    return _angle_between(points[a] - points[b], points[c] - points[b], warnings, name)


def _distance(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def _unit(vector: np.ndarray, warnings: list[str], name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        warnings.append(f"degenerate_vector:{name}")
        return np.full(3, np.nan, dtype=np.float64)
    return vector / norm


def _straightness(points: np.ndarray, indices: tuple[int, ...], warnings: list[str], name: str) -> float:
    polyline = float(
        sum(np.linalg.norm(points[end] - points[start]) for start, end in zip(indices[:-1], indices[1:]))
    )
    if polyline <= 1e-12:
        warnings.append(f"degenerate_polyline:{name}")
        return float("nan")
    return float(np.linalg.norm(points[indices[-1]] - points[indices[0]]) / polyline)


def _thumb_opposition_angle(points: np.ndarray, warnings: list[str]) -> float:
    palm_normal = np.cross(points[5] - points[0], points[17] - points[0])
    thumb_normal = np.cross(points[1] - points[0], points[2] - points[1])
    return _angle_between(palm_normal, thumb_normal, warnings, "thumb_opposition")


def _segment_mean_spread(
    points: np.ndarray,
    first_segments: tuple[tuple[int, int], ...],
    second_segments: tuple[tuple[int, int], ...],
    warnings: list[str],
    name: str,
) -> float:
    angles = [
        _angle_between(
            points[first_end] - points[first_start],
            points[second_end] - points[second_start],
            warnings,
            f"{name}.{index}",
        )
        for index, ((first_start, first_end), (second_start, second_end)) in enumerate(
            zip(first_segments, second_segments)
        )
    ]
    finite = np.asarray([value for value in angles if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def extract_feature_vector(
    normalization: SpatialNormalizationResult,
    *,
    landmark_source: str,
    scale_name: str,
) -> FeatureVector:
    """Extract all 159 features for one raw/filtered and scale branch.

    Shape features are calculated from canonical points. The last six global
    orientation features intentionally use the preserved world-space basis.
    """
    if landmark_source not in {"raw", "filtered"}:
        raise FeatureExtractionError("landmark_source must be 'raw' or 'filtered'.")
    if scale_name == "wrist_middle_mcp":
        points = np.asarray(normalization.normalized_wrist_middle_mcp, dtype=np.float64)
    elif scale_name == "middle_finger":
        points = np.asarray(normalization.normalized_middle_finger, dtype=np.float64)
    else:
        raise FeatureExtractionError(
            "scale_name must be 'wrist_middle_mcp' or 'middle_finger'."
        )
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise FeatureExtractionError("Normalized landmarks must be a finite (21, 3) array.")

    warnings: list[str] = []
    values: list[float] = points.reshape(-1).astype(np.float64).tolist()

    joint_features = (
        (0, 1, 2, "thumb.cmc"),
        (1, 2, 3, "thumb.mcp"),
        (2, 3, 4, "thumb.ip"),
        (0, 5, 6, "index.mcp"),
        (5, 6, 7, "index.pip"),
        (6, 7, 8, "index.dip"),
        (0, 9, 10, "middle.mcp"),
        (9, 10, 11, "middle.pip"),
        (10, 11, 12, "middle.dip"),
        (0, 13, 14, "ring.mcp"),
        (13, 14, 15, "ring.pip"),
        (14, 15, 16, "ring.dip"),
        (0, 17, 18, "pinky.mcp"),
        (17, 18, 19, "pinky.pip"),
        (18, 19, 20, "pinky.dip"),
    )
    values.extend(
        _joint_angle(points, a, b, c, warnings, f"bend.{name}")
        for a, b, c, name in joint_features
    )

    bases = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
    axes = {"thumb": (2, 4), "index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}
    tips = dict(zip(FINGER_NAMES, FINGER_TIPS))
    for first, second in PAIR_NAMES:
        values.append(_joint_angle(points, bases[first], 0, bases[second], warnings, f"spread.mcp.{first}_{second}"))
    for first, second in PAIR_NAMES:
        first_start, first_end = axes[first]
        second_start, second_end = axes[second]
        values.append(
            _angle_between(
                points[first_end] - points[first_start],
                points[second_end] - points[second_start],
                warnings,
                f"spread.axis.{first}_{second}",
            )
        )

    segments = {
        "thumb": ((1, 2), (2, 3), (3, 4)),
        "index": ((5, 6), (6, 7), (7, 8)),
        "middle": ((9, 10), (10, 11), (11, 12)),
        "ring": ((13, 14), (14, 15), (15, 16)),
        "pinky": ((17, 18), (18, 19), (19, 20)),
    }
    for first, second in (("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        values.append(
            _segment_mean_spread(points, segments[first], segments[second], warnings, f"spread.segment_mean.{first}_{second}")
        )

    for first, second in PAIR_NAMES:
        values.append(_distance(points, tips[first], tips[second]))
    for finger in FINGER_NAMES:
        values.append(_distance(points, tips[finger], 0))

    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    for finger in FINGER_NAMES:
        values.append(float(np.linalg.norm(points[tips[finger]] - palm_center)))

    straightness_indices = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    for finger in FINGER_NAMES:
        values.append(_straightness(points, straightness_indices[finger], warnings, f"straightness.{finger}"))

    for finger in FINGER_NAMES:
        start, end = axes[finger]
        unit_axis = _unit(points[end] - points[start], warnings, f"out_of_plane.{finger}")
        values.append(float(math.asin(np.clip(unit_axis[2], -1.0, 1.0))) if np.isfinite(unit_axis).all() else float("nan"))

    thumb_axis = points[4] - points[2]
    index_axis = points[8] - points[5]
    values.append(_angle_between(thumb_axis, index_axis, warnings, "thumb_index.grip_angle"))
    values.append(_distance(points, 3, 7))
    values.append(_distance(points, 2, 6))
    values.append(_thumb_opposition_angle(points, warnings))
    thumb_axis_unit = _unit(thumb_axis, warnings, "thumb_axis")
    values.append(float(math.atan2(thumb_axis_unit[1], thumb_axis_unit[0])) if np.isfinite(thumb_axis_unit).all() else float("nan"))
    values.extend(float(points[index, 2]) for index in (1, 2, 3, 4))
    values.extend(_distance(points, 4, base) for base in (5, 9, 13, 17))
    values.extend(float(component) for component in thumb_axis_unit)

    palm_width = _distance(points, 5, 17)
    values.extend(
        (
            palm_width,
            _distance(points, 5, 9),
            _distance(points, 9, 13),
            _distance(points, 13, 17),
        )
    )
    palm_length = float(np.linalg.norm(normalization.aligned_world[9] - normalization.aligned_world[0]))
    world_palm_width = float(np.linalg.norm(normalization.aligned_world[5] - normalization.aligned_world[17]))
    values.append(world_palm_width / palm_length if palm_length > 1e-12 else float("nan"))

    axes_world = np.asarray(normalization.axes_world, dtype=np.float64)
    values.extend(float(value) for value in axes_world[:, 2])
    values.extend(float(value) for value in axes_world[:, 1])

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (FEATURE_COUNT,):
        raise FeatureExtractionError(
            f"Extractor produced {array.size} values instead of {FEATURE_COUNT}."
        )
    return FeatureVector(
        values=array,
        landmark_source=landmark_source,
        scale_name=scale_name,
        warnings=sorted(set(warnings)),
    )
