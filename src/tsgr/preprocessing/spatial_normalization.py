"""Spatial canonicalization and dual-scale normalization for MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
PINKY_MCP = 17


class SpatialNormalizationError(ValueError):
    """Raised when a stable local hand coordinate system cannot be constructed."""


@dataclass(slots=True)
class SpatialNormalizationResult:
    """All intermediate arrays and diagnostics produced by spatial normalization."""

    origin_world: np.ndarray
    axes_world: np.ndarray
    rotation_world_to_canonical: np.ndarray
    centered_world: np.ndarray
    aligned_world: np.ndarray
    normalized_wrist_middle_mcp: np.ndarray
    normalized_middle_finger: np.ndarray
    scale_wrist_middle_mcp: float
    scale_middle_finger: float
    scale_middle_finger_chord: float
    scale_ratio_middle_to_palm: float
    diagnostics: dict[str, float | bool | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_world": self.origin_world.tolist(),
            "axes_world": self.axes_world.tolist(),
            "rotation_world_to_canonical": self.rotation_world_to_canonical.tolist(),
            "centered_world": self.centered_world.tolist(),
            "aligned_world": self.aligned_world.tolist(),
            "normalized_wrist_middle_mcp": self.normalized_wrist_middle_mcp.tolist(),
            "normalized_middle_finger": self.normalized_middle_finger.tolist(),
            "scales": {
                "wrist_middle_mcp": float(self.scale_wrist_middle_mcp),
                "middle_finger_polyline": float(self.scale_middle_finger),
                "middle_finger_chord": float(self.scale_middle_finger_chord),
                "middle_to_palm_ratio": float(self.scale_ratio_middle_to_palm),
            },
            "diagnostics": self.diagnostics,
        }


def _as_landmarks(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (21, 3):
        raise SpatialNormalizationError(
            f"Expected world landmarks with shape (21, 3), got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise SpatialNormalizationError("World landmarks contain NaN or infinite values.")
    return array


def _safe_unit(vector: np.ndarray, minimum_norm: float, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= minimum_norm:
        raise SpatialNormalizationError(
            f"Cannot construct {name}: vector norm {norm!r} is not greater than {minimum_norm}."
        )
    return vector / norm


def middle_finger_polyline_length(points: np.ndarray) -> float:
    """Return the P9-P10-P11-P12 polyline length in the input coordinate unit."""
    array = _as_landmarks(points)
    indices = (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    return float(
        sum(
            np.linalg.norm(array[end] - array[start])
            for start, end in zip(indices[:-1], indices[1:])
        )
    )


def build_canonical_axes(
    points: np.ndarray,
    minimum_axis_norm: float = 1e-8,
) -> np.ndarray:
    """Build a right-handed orthonormal basis attached to a right hand.

    Columns contain the canonical X, Y, and Z axes expressed in MediaPipe world
    coordinates. +Y points from the wrist to the middle MCP. +X points from the
    pinky side toward the index/thumb side. +Z follows X cross Y and therefore
    remains tied to the palm plane instead of the camera frame.
    """
    array = _as_landmarks(points)
    y_axis = _safe_unit(
        array[MIDDLE_MCP] - array[WRIST], minimum_axis_norm, "canonical Y axis"
    )
    lateral_hint = array[INDEX_MCP] - array[PINKY_MCP]
    lateral_orthogonal = lateral_hint - np.dot(lateral_hint, y_axis) * y_axis
    x_axis = _safe_unit(lateral_orthogonal, minimum_axis_norm, "canonical X axis")
    z_axis = _safe_unit(
        np.cross(x_axis, y_axis), minimum_axis_norm, "canonical Z axis"
    )
    # Recompute X to remove accumulated floating-point non-orthogonality.
    x_axis = _safe_unit(
        np.cross(y_axis, z_axis), minimum_axis_norm, "re-orthogonalized X axis"
    )
    axes = np.column_stack((x_axis, y_axis, z_axis))
    determinant = float(np.linalg.det(axes))
    if determinant <= 0.0:
        raise SpatialNormalizationError(
            f"Canonical basis is not right-handed; determinant={determinant:.6g}."
        )
    return axes


def normalize_world_landmarks(
    points: np.ndarray,
    *,
    minimum_axis_norm: float = 1e-8,
    minimum_scale: float = 1e-6,
) -> SpatialNormalizationResult:
    """Center, rotate, and normalize 21 MediaPipe world landmarks by two scales."""
    array = _as_landmarks(points)
    origin = array[WRIST].copy()
    centered = array - origin
    axes = build_canonical_axes(array, minimum_axis_norm=minimum_axis_norm)

    # For row-vector points, multiplying by axes returns dot products with each axis.
    aligned = centered @ axes
    rotation_world_to_canonical = axes.T

    scale_palm = float(np.linalg.norm(array[MIDDLE_MCP] - array[WRIST]))
    scale_middle = middle_finger_polyline_length(array)
    scale_middle_chord = float(np.linalg.norm(array[MIDDLE_TIP] - array[MIDDLE_MCP]))
    if scale_palm <= minimum_scale:
        raise SpatialNormalizationError(
            f"Wrist-middle MCP scale {scale_palm:.6g} is not greater than {minimum_scale}."
        )
    if scale_middle <= minimum_scale:
        raise SpatialNormalizationError(
            f"Middle-finger polyline scale {scale_middle:.6g} is not greater than {minimum_scale}."
        )

    normalized_palm = aligned / scale_palm
    normalized_middle = aligned / scale_middle
    reconstructed = aligned @ axes.T + origin
    reconstruction_error = float(np.max(np.linalg.norm(reconstructed - array, axis=1)))
    gram = axes.T @ axes
    orthogonality_error = float(np.max(np.abs(gram - np.eye(3))))
    determinant = float(np.linalg.det(axes))

    diagnostics: dict[str, float | bool | str] = {
        "valid": True,
        "axis_orthogonality_error": orthogonality_error,
        "basis_determinant": determinant,
        "maximum_reconstruction_error": reconstruction_error,
        "canonical_wrist_norm": float(np.linalg.norm(aligned[WRIST])),
        "canonical_middle_mcp_x_abs": float(abs(aligned[MIDDLE_MCP, 0])),
        "canonical_middle_mcp_z_abs": float(abs(aligned[MIDDLE_MCP, 2])),
        "palm_width_world": float(np.linalg.norm(array[INDEX_MCP] - array[PINKY_MCP])),
        "palm_normal_camera_z": float(axes[2, 2]),
        "hand_axis_camera_z": float(axes[2, 1]),
    }

    return SpatialNormalizationResult(
        origin_world=origin,
        axes_world=axes,
        rotation_world_to_canonical=rotation_world_to_canonical,
        centered_world=centered,
        aligned_world=aligned,
        normalized_wrist_middle_mcp=normalized_palm,
        normalized_middle_finger=normalized_middle,
        scale_wrist_middle_mcp=scale_palm,
        scale_middle_finger=scale_middle,
        scale_middle_finger_chord=scale_middle_chord,
        scale_ratio_middle_to_palm=scale_middle / scale_palm,
        diagnostics=diagnostics,
    )


def apply_similarity_transform(
    points: np.ndarray,
    *,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply a synthetic row-vector similarity transform used by tests and benchmarks."""
    array = _as_landmarks(points)
    rotation_array = np.asarray(rotation, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3).")
    if translation_array.shape != (3,):
        raise ValueError("translation must have shape (3,).")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    return scale * (array @ rotation_array.T) + translation_array
