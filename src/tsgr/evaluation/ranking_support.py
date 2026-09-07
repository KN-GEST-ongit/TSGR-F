"""Numerical helpers used by the final TSGR-F ranking and routing pipeline."""
from __future__ import annotations

import numpy as np


def normalize_positive_weights(values: np.ndarray, *, epsilon: float = 1.0e-12) -> np.ndarray:
    """Normalize non-negative feature weights to unit mean."""
    weights = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    mean = float(weights.mean()) if weights.size else 0.0
    if mean <= epsilon:
        return np.ones_like(weights, dtype=np.float64)
    return weights / mean


def pairwise_distance_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    metric: str,
    weights: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return pairwise sample distances without materializing a full delta cube."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if metric in {"euclidean", "weighted_euclidean"}:
        if metric == "weighted_euclidean":
            scale = np.sqrt(np.maximum(weights, 0.0))
            a = left * scale
            b = right * scale
        else:
            a = left
            b = right
        a2 = np.sum(a * a, axis=1)[:, None]
        b2 = np.sum(b * b, axis=1)[None, :]
        squared = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
        return np.sqrt(squared)

    result = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    for start in range(0, left.shape[0], chunk_size):
        stop = min(left.shape[0], start + chunk_size)
        delta = np.abs(left[start:stop, None, :] - right[None, :, :])
        if metric == "weighted_manhattan":
            result[start:stop] = np.sum(weights[None, None, :] * delta, axis=2)
        elif metric == "manhattan":
            result[start:stop] = np.sum(delta, axis=2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return result


def global_inverse_within_weights(values: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    """Compute inverse within-class variance weights from TRAIN samples only."""
    values = np.asarray(values, dtype=np.float64)
    variances: list[np.ndarray] = []
    for class_index in range(class_count):
        class_values = values[labels == class_index]
        if class_values.shape[0] < 2:
            continue
        variances.append(np.var(class_values, axis=0, ddof=1))
    within = np.mean(np.stack(variances, axis=0), axis=0)
    floor = max(float(np.median(within[within > 0])) * 1e-3 if np.any(within > 0) else 1e-6, 1e-9)
    return normalize_positive_weights(1.0 / np.maximum(within, floor))


def loo_scores(
    values: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute leave-one-out nearest-exemplar scores for every class."""
    sample_count = values.shape[0]
    scores = np.full((sample_count, class_count), np.inf, dtype=np.float64)
    nearest = np.full((sample_count, class_count), -1, dtype=np.int64)
    for class_index in range(class_count):
        class_indices = np.flatnonzero(labels == class_index)
        distances = pairwise_distance_matrix(
            values,
            values[class_indices],
            metric="weighted_euclidean",
            weights=weights,
        )
        own_positions = {int(global_i): local_i for local_i, global_i in enumerate(class_indices.tolist())}
        for sample_index in range(sample_count):
            row = distances[sample_index].copy()
            if sample_index in own_positions:
                row[own_positions[sample_index]] = np.inf
            nearest_local = int(np.argmin(row))
            scores[sample_index, class_index] = float(row[nearest_local])
            nearest[sample_index, class_index] = int(class_indices[nearest_local])
    return scores, nearest


def class_fused_score(
    values3d: np.ndarray,
    values2d: np.ndarray,
    labels: np.ndarray,
    class_index: int,
    weights3d: np.ndarray,
    weights2d: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the LOO nearest-exemplar score for one class under 3D/2D fusion."""
    class_indices = np.flatnonzero(labels == int(class_index))
    if class_indices.size < 2:
        raise ValueError("Each specialist class requires at least two samples.")
    w3 = np.asarray(weights3d, dtype=np.float64)
    w2 = np.asarray(weights2d, dtype=np.float64)
    w3 = w3 / max(float(np.sum(w3)), 1e-12)
    w2 = w2 / max(float(np.sum(w2)), 1e-12)
    d3 = values3d[:, None, :] - values3d[class_indices][None, :, :]
    d2 = values2d[:, None, :] - values2d[class_indices][None, :, :]
    s3 = np.sum(d3 * d3 * w3[None, None, :], axis=2)
    s2 = np.sum(d2 * d2 * w2[None, None, :], axis=2)
    fused = np.sqrt(np.maximum((1.0 - alpha) * s3 + alpha * s2, 0.0))
    own_positions = {int(global_i): local_i for local_i, global_i in enumerate(class_indices.tolist())}
    result = np.empty(values3d.shape[0], dtype=np.float64)
    for sample_index in range(values3d.shape[0]):
        row = fused[sample_index].copy()
        local = own_positions.get(sample_index)
        if local is not None:
            row[local] = np.inf
        result[sample_index] = float(np.min(row))
    return result


def fisher_scores(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return per-feature Fisher scores for two sample groups."""
    def safe_variance(values: np.ndarray) -> np.ndarray:
        if values.shape[0] < 2:
            return np.zeros(values.shape[1], dtype=np.float64)
        return np.var(values, axis=0, ddof=1)

    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    denominator = safe_variance(a) + safe_variance(b) + epsilon
    score = (mean_a - mean_b) ** 2 / denominator
    return np.where(np.isfinite(score), score, 0.0)
