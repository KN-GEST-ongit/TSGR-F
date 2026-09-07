"""Detect constant and near-constant features for reference-model distance spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsgr.reference_models.provenance import feature_mask_hash


@dataclass(slots=True)
class FeatureMaskResult:
    """Global active mask shared by every gesture model in one branch."""

    active_mask: np.ndarray
    reasons: tuple[str, ...]
    standard_deviation: np.ndarray
    value_range: np.ndarray
    median_absolute_value: np.ndarray
    mask_sha256: str


def build_feature_mask(
    values: np.ndarray,
    feature_ids: tuple[str, ...] | list[str],
    *,
    exact_tolerance: float = 1.0e-12,
    absolute_std_threshold: float = 1.0e-8,
    relative_std_threshold: float = 1.0e-6,
) -> FeatureMaskResult:
    """Mask features that carry no measurable variation in the complete model set."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_ids):
        raise ValueError("Feature-mask matrix shape does not match feature identifiers.")
    if matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Feature-mask values must be a non-empty finite matrix.")
    std = np.std(matrix, axis=0)
    ranges = np.max(matrix, axis=0) - np.min(matrix, axis=0)
    median_abs = np.median(np.abs(matrix), axis=0)
    relative_limit = relative_std_threshold * np.maximum(1.0, median_abs)
    exact = (std <= exact_tolerance) | (ranges <= exact_tolerance)
    near = (~exact) & (std <= np.maximum(absolute_std_threshold, relative_limit))
    active = ~(exact | near)
    reasons = tuple(
        "active" if active[index] else ("constant" if exact[index] else "near_constant")
        for index in range(matrix.shape[1])
    )
    return FeatureMaskResult(
        active_mask=active,
        reasons=reasons,
        standard_deviation=std,
        value_range=ranges,
        median_absolute_value=median_abs,
        mask_sha256=feature_mask_hash(list(feature_ids), active.tolist()),
    )
