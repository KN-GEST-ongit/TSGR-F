from __future__ import annotations

import numpy as np
import pytest

from tsgr.preprocessing.spatial_normalization import (
    SpatialNormalizationError,
    apply_similarity_transform,
    normalize_world_landmarks,
)


def synthetic_right_hand() -> np.ndarray:
    return np.asarray(
        [
            [0.000, 0.000, 0.000],
            [-0.018, 0.015, 0.002],
            [-0.035, 0.031, 0.005],
            [-0.052, 0.047, 0.008],
            [-0.070, 0.061, 0.010],
            [-0.026, 0.041, 0.000],
            [-0.029, 0.076, 0.002],
            [-0.031, 0.106, 0.004],
            [-0.032, 0.132, 0.005],
            [0.000, 0.046, 0.000],
            [0.000, 0.087, 0.001],
            [0.001, 0.122, 0.002],
            [0.001, 0.153, 0.003],
            [0.023, 0.043, 0.000],
            [0.026, 0.081, 0.001],
            [0.028, 0.112, 0.002],
            [0.030, 0.139, 0.003],
            [0.043, 0.036, 0.000],
            [0.049, 0.068, 0.001],
            [0.053, 0.094, 0.002],
            [0.057, 0.116, 0.003],
        ],
        dtype=np.float64,
    )


def rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return z @ y @ x


def test_canonicalization_is_similarity_invariant() -> None:
    source = synthetic_right_hand()
    baseline = normalize_world_landmarks(source)
    transformed = apply_similarity_transform(
        source,
        rotation=rotation_xyz(0.44, -0.62, 1.14),
        scale=2.75,
        translation=np.asarray([0.34, -0.12, 0.81]),
    )
    result = normalize_world_landmarks(transformed)

    np.testing.assert_allclose(
        result.normalized_wrist_middle_mcp,
        baseline.normalized_wrist_middle_mcp,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.normalized_middle_finger,
        baseline.normalized_middle_finger,
        atol=1e-10,
    )
    assert result.scale_wrist_middle_mcp == pytest.approx(
        baseline.scale_wrist_middle_mcp * 2.75
    )
    assert result.scale_middle_finger == pytest.approx(
        baseline.scale_middle_finger * 2.75
    )


def test_axes_are_orthonormal_and_canonical_middle_axis_is_vertical() -> None:
    result = normalize_world_landmarks(synthetic_right_hand())
    np.testing.assert_allclose(result.axes_world.T @ result.axes_world, np.eye(3), atol=1e-12)
    assert np.linalg.det(result.axes_world) == pytest.approx(1.0, abs=1e-12)
    assert result.aligned_world[9, 0] == pytest.approx(0.0, abs=1e-12)
    assert result.aligned_world[9, 2] == pytest.approx(0.0, abs=1e-12)
    assert result.normalized_wrist_middle_mcp[9, 1] == pytest.approx(1.0, abs=1e-12)


def test_two_scales_are_preserved_separately() -> None:
    result = normalize_world_landmarks(synthetic_right_hand())
    expected_middle = sum(
        np.linalg.norm(synthetic_right_hand()[end] - synthetic_right_hand()[start])
        for start, end in ((9, 10), (10, 11), (11, 12))
    )
    assert result.scale_middle_finger == pytest.approx(expected_middle)
    assert result.scale_ratio_middle_to_palm == pytest.approx(
        result.scale_middle_finger / result.scale_wrist_middle_mcp
    )
    assert not np.allclose(
        result.normalized_wrist_middle_mcp,
        result.normalized_middle_finger,
    )


def test_degenerate_palm_geometry_is_rejected() -> None:
    points = synthetic_right_hand()
    points[5] = points[17]
    with pytest.raises(SpatialNormalizationError):
        normalize_world_landmarks(points)


def test_nonfinite_landmarks_are_rejected() -> None:
    points = synthetic_right_hand()
    points[4, 2] = np.nan
    with pytest.raises(SpatialNormalizationError):
        normalize_world_landmarks(points)
