"""Balanced empirical and regularized covariance helpers."""

from __future__ import annotations

import numpy as np


def empirical_covariance(samples: np.ndarray) -> np.ndarray:
    """Return a symmetric population covariance matrix for row-wise samples."""
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("Covariance samples must be a non-empty 2-D matrix.")
    if matrix.shape[0] == 1:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    centered = matrix - np.mean(matrix, axis=0)
    covariance = centered.T @ centered / float(matrix.shape[0])
    return 0.5 * (covariance + covariance.T)


def weighted_empirical_covariance(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a population covariance where non-negative sample weights sum to one."""
    matrix = np.asarray(samples, dtype=np.float64)
    weight_vector = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("Covariance samples must be a non-empty 2-D matrix.")
    if weight_vector.shape != (matrix.shape[0],):
        raise ValueError("Covariance weights do not match the number of samples.")
    if np.any(weight_vector < 0) or not np.isfinite(weight_vector).all():
        raise ValueError("Covariance weights must be finite and non-negative.")
    total = float(weight_vector.sum())
    if total <= 0:
        raise ValueError("Covariance weights must have a positive sum.")
    normalized = weight_vector / total
    mean = np.sum(matrix * normalized[:, None], axis=0)
    centered = matrix - mean
    covariance = (centered * normalized[:, None]).T @ centered
    return 0.5 * (covariance + covariance.T)


def shrink_covariance(
    covariance: np.ndarray,
    *,
    shrinkage: float = 0.1,
    diagonal_floor: float = 1.0e-8,
) -> np.ndarray:
    """Shrink covariance toward an isotropic target and ensure positive diagonals."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Covariance shrinkage must be in [0, 1].")
    dimension = matrix.shape[0]
    mean_variance = float(np.trace(matrix) / dimension) if dimension else 0.0
    target_scale = max(mean_variance, diagonal_floor)
    regularized = (1.0 - shrinkage) * matrix + shrinkage * target_scale * np.eye(dimension)
    diagonal = np.diag(regularized)
    correction = np.maximum(diagonal_floor - diagonal, 0.0)
    regularized = regularized + np.diag(correction)
    return 0.5 * (regularized + regularized.T)
