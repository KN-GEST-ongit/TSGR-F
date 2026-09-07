"""Robust within-session feature outlier detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SessionOutlierResult:
    """Per-frame robust scores, acceptance mask, and diagnostic details."""

    accepted_mask: np.ndarray
    robust_rms_score: np.ndarray
    extreme_feature_fraction: np.ndarray
    robust_center: np.ndarray
    robust_scale: np.ndarray
    variable_feature_mask: np.ndarray
    top_feature_indices: list[list[int]]


def detect_session_outliers(
    values: np.ndarray,
    *,
    feature_z_threshold: float = 6.0,
    robust_rms_threshold: float = 4.0,
    maximum_extreme_fraction: float = 0.05,
    scale_floor: float = 1.0e-6,
    maximum_reported_features: int = 8,
) -> SessionOutlierResult:
    """Detect gross feature vectors using median/MAD standardized deviations.

    The detector intentionally targets clear landmark failures rather than removing a
    fixed percentage of valid frames. A frame is rejected only when its aggregate
    robust deviation or the fraction of extreme features exceeds an absolute limit.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("Session feature values must be a non-empty 2-D matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Session feature values must be finite before outlier detection.")
    center = np.median(matrix, axis=0)
    absolute_deviation = np.abs(matrix - center)
    mad = np.median(absolute_deviation, axis=0)
    q1 = np.quantile(matrix, 0.25, axis=0)
    q3 = np.quantile(matrix, 0.75, axis=0)
    robust_scale = np.maximum(1.4826 * mad, 0.05 * (q3 - q1))
    value_range = np.max(matrix, axis=0) - np.min(matrix, axis=0)
    # A single severe landmark failure can leave MAD and IQR equal to zero.
    # Treat any range above the floor as variable and standardize it with the floor.
    variable = (robust_scale > scale_floor) | (value_range > scale_floor)
    robust_scale = np.where(variable, np.maximum(robust_scale, scale_floor), 1.0)
    z = absolute_deviation / robust_scale
    if variable.any():
        selected = z[:, variable]
        clipped = np.minimum(selected, 12.0)
        rms = np.sqrt(np.mean(np.square(clipped), axis=1))
        extreme_fraction = np.mean(selected > feature_z_threshold, axis=1)
    else:
        rms = np.zeros(matrix.shape[0], dtype=np.float64)
        extreme_fraction = np.zeros(matrix.shape[0], dtype=np.float64)
    accepted = (rms <= robust_rms_threshold) & (
        extreme_fraction <= maximum_extreme_fraction
    )
    top_features: list[list[int]] = []
    for row in z:
        ordered = np.argsort(row)[::-1]
        top_features.append(
            [
                int(index)
                for index in ordered[:maximum_reported_features]
                if variable[index] and row[index] > feature_z_threshold
            ]
        )
    return SessionOutlierResult(
        accepted_mask=accepted,
        robust_rms_score=rms,
        extreme_feature_fraction=extreme_fraction,
        robust_center=center,
        robust_scale=np.where(variable, robust_scale, 0.0),
        variable_feature_mask=variable,
        top_feature_indices=top_features,
    )
