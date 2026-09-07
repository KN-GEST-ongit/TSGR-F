from __future__ import annotations

import numpy as np

from tsgr.features import FEATURE_COUNT, extract_feature_vector
from tsgr.preprocessing.spatial_normalization import apply_similarity_transform, normalize_world_landmarks


def synthetic_right_hand() -> np.ndarray:
    return np.asarray(
        [
            [0.000, 0.000, 0.000], [-0.018, 0.015, 0.002], [-0.035, 0.031, 0.005],
            [-0.052, 0.047, 0.008], [-0.070, 0.061, 0.010], [-0.026, 0.041, 0.000],
            [-0.029, 0.076, 0.002], [-0.031, 0.106, 0.004], [-0.032, 0.132, 0.005],
            [0.000, 0.046, 0.000], [0.000, 0.087, 0.001], [0.001, 0.122, 0.002],
            [0.001, 0.153, 0.003], [0.023, 0.043, 0.000], [0.026, 0.081, 0.001],
            [0.028, 0.112, 0.002], [0.030, 0.139, 0.003], [0.043, 0.036, 0.000],
            [0.049, 0.068, 0.001], [0.053, 0.094, 0.002], [0.057, 0.116, 0.003],
        ],
        dtype=np.float64,
    )


def rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_feature_vector_is_complete_and_finite_for_reference_geometry() -> None:
    result = normalize_world_landmarks(synthetic_right_hand())
    vector = extract_feature_vector(
        result,
        landmark_source="raw",
        scale_name="wrist_middle_mcp",
    )
    assert vector.values.shape == (FEATURE_COUNT,)
    assert vector.complete
    assert vector.finite_count == FEATURE_COUNT
    assert vector.warnings == []


def test_shape_features_are_similarity_invariant_but_orientation_is_preserved() -> None:
    points = synthetic_right_hand()
    baseline = extract_feature_vector(
        normalize_world_landmarks(points),
        landmark_source="raw",
        scale_name="wrist_middle_mcp",
    )
    transformed_points = apply_similarity_transform(
        points,
        rotation=rotation_z(0.7),
        scale=2.4,
        translation=np.asarray([0.4, -0.2, 0.7]),
    )
    transformed = extract_feature_vector(
        normalize_world_landmarks(transformed_points),
        landmark_source="raw",
        scale_name="wrist_middle_mcp",
    )
    np.testing.assert_allclose(baseline.values[:-6], transformed.values[:-6], atol=1e-10)
    assert not np.allclose(baseline.values[-6:], transformed.values[-6:])


def test_alternative_scale_changes_distances_but_not_angles_or_ratios() -> None:
    result = normalize_world_landmarks(synthetic_right_hand())
    palm = extract_feature_vector(result, landmark_source="raw", scale_name="wrist_middle_mcp")
    middle = extract_feature_vector(result, landmark_source="raw", scale_name="middle_finger")
    assert not np.allclose(palm.values[:63], middle.values[:63])
    np.testing.assert_allclose(palm.values[63:102], middle.values[63:102], atol=1e-12)
    assert palm.values[152] == middle.values[152]
