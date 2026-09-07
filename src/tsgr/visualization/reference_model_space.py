"""Model-space diagnostics for TSGR-F reference gesture prototypes.

The routines in this module are intentionally report-oriented rather than a
classifier. They compare persisted gesture models with exactly the feature mask,
feature schema and uncertainty statistics stored by the reference-model builder.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tsgr.constants import HAND_CONNECTIONS, LANDMARK_NAMES
from tsgr.reference_models.repository import (
    ReferenceGestureModel,
    branch_directory_name,
    load_reference_gesture_model,
)
from tsgr.visualization.matplotlib_backend import configure_headless_backend


MODEL_DISTANCE_METRICS = (
    "euclidean",
    "standardized_euclidean",
    "weighted_standardized_euclidean",
    "manhattan",
    "weighted_manhattan",
    "cosine",
    "diagonal_mahalanobis",
    "full_mahalanobis",
)


@dataclass(slots=True)
class ModelSpace:
    """Loaded branch-level model-space data."""

    model_set_dir: Path
    branch_name: str
    gesture_ids: tuple[str, ...]
    models: tuple[ReferenceGestureModel, ...]
    feature_ids: tuple[str, ...]
    feature_groups: tuple[str, ...]
    feature_group_by_index: tuple[str, ...]
    active_mask: np.ndarray
    compact_mask: np.ndarray
    fisher_weights: np.ndarray
    inverse_within_weights: np.ndarray


@dataclass(slots=True)
class PairDistanceDiagnostics:
    """Distance and diagnostic contributions for one model pair."""

    gesture_a: str
    gesture_b: str
    distance: float
    standardized_components: np.ndarray
    weighted_standardized_components: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_feature_weights(branch_dir: Path, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    fisher = np.ones(feature_count, dtype=np.float64)
    inverse = np.ones(feature_count, dtype=np.float64)
    path = branch_dir / "feature_weight_diagnostics.csv"
    if not path.is_file():
        return fisher, inverse
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["feature_index"])
            fisher[index] = float(row.get("fisher_score_weight") or 0.0)
            inverse[index] = float(row.get("inverse_within_variance_weight") or 0.0)
    return fisher, inverse



def _read_compact_mask(branch_dir: Path, feature_count: int, active_mask: np.ndarray) -> np.ndarray:
    path = branch_dir / "compact_feature_mask.json"
    if not path.is_file():
        return np.asarray(active_mask, dtype=bool).copy()
    payload = _read_json(path)
    features = payload.get("features", [])
    if not isinstance(features, list) or len(features) != feature_count:
        raise ValueError("Compact feature mask does not match model feature count.")
    mask = np.zeros(feature_count, dtype=bool)
    for row in features:
        index = int(row["feature_index"])
        mask[index] = bool(row.get("compact_active", False))
    mask &= np.asarray(active_mask, dtype=bool)
    if not mask.any():
        raise ValueError("Persisted compact feature mask is empty.")
    return mask


def load_model_space(model_set_dir: str | Path, branch_name: str) -> ModelSpace:
    """Load all gesture models belonging to one branch and validate common metadata."""
    root = Path(model_set_dir)
    branch_dir = root / "branches" / branch_directory_name(branch_name)
    branch_payload = _read_json(branch_dir / "branch_model.json")
    schema_payload = _read_json(branch_dir / "feature_schema_snapshot.json")
    gesture_ids = tuple(str(value) for value in branch_payload.get("gesture_ids", []))
    if len(gesture_ids) < 2:
        raise ValueError("Model-space visualization requires at least two gesture models.")
    models = tuple(
        load_reference_gesture_model(root, gesture_id=gesture, branch_name=branch_name)
        for gesture in gesture_ids
    )
    feature_ids = models[0].feature_ids
    active_mask = models[0].active_mask.copy()
    for model in models[1:]:
        if model.feature_ids != feature_ids:
            raise ValueError("Gesture models do not share the same feature identifiers.")
        if not np.array_equal(model.active_mask, active_mask):
            raise ValueError("Gesture models do not share the same active feature mask.")
    definitions = sorted(schema_payload.get("features", []), key=lambda row: int(row["index"]))
    if len(definitions) != len(feature_ids):
        raise ValueError("Feature-schema snapshot does not match gesture-model vector length.")
    groups_by_index = tuple(str(row["group"]) for row in definitions)
    groups: list[str] = []
    for group in groups_by_index:
        if group not in groups:
            groups.append(group)
    fisher, inverse = _read_feature_weights(branch_dir, len(feature_ids))
    compact_mask = _read_compact_mask(branch_dir, len(feature_ids), active_mask)
    return ModelSpace(
        model_set_dir=root,
        branch_name=branch_name,
        gesture_ids=gesture_ids,
        models=models,
        feature_ids=feature_ids,
        feature_groups=tuple(groups),
        feature_group_by_index=groups_by_index,
        active_mask=active_mask,
        compact_mask=compact_mask,
        fisher_weights=fisher,
        inverse_within_weights=inverse,
    )


def selected_feature_mask(
    space: ModelSpace,
    *,
    feature_set: str = "active",
    groups: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a Boolean feature mask for model-space analysis."""
    feature_set = feature_set.strip().lower()
    if feature_set not in {"active", "full", "compact"}:
        raise ValueError("feature_set must be 'active', 'full', or 'compact'.")
    if feature_set == "active":
        mask = space.active_mask.copy()
    elif feature_set == "compact":
        mask = space.compact_mask.copy()
    else:
        mask = np.ones(len(space.feature_ids), dtype=bool)
    if groups:
        requested = {str(group) for group in groups}
        unknown = requested.difference(space.feature_groups)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        group_mask = np.asarray(
            [group in requested for group in space.feature_group_by_index], dtype=bool
        )
        mask &= group_mask
    if not mask.any():
        raise ValueError("Selected model-space feature mask is empty.")
    return mask


def _renormalized_weights(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(weights, dtype=np.float64)
    selected = np.maximum(np.asarray(weights, dtype=np.float64)[mask], 0.0)
    if selected.size == 0:
        return result
    mean = float(selected.mean())
    if mean <= 0.0 or not np.isfinite(mean):
        selected = np.ones_like(selected)
    else:
        selected = selected / mean
    result[mask] = selected
    return result


def _pooled_variance(model_a: ReferenceGestureModel, model_b: ReferenceGestureModel) -> np.ndarray:
    return 0.5 * (
        np.maximum(model_a.standard_deviation, 0.0) ** 2
        + np.maximum(model_b.standard_deviation, 0.0) ** 2
    )


def _active_covariance_subset(
    model_a: ReferenceGestureModel,
    model_b: ReferenceGestureModel,
    full_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_full = np.flatnonzero(full_mask & model_a.active_mask)
    position = {int(full): pos for pos, full in enumerate(model_a.active_indices.tolist())}
    active_positions = np.asarray([position[int(index)] for index in selected_full], dtype=np.int64)
    covariance = 0.5 * (
        model_a.covariance_regularized_active[np.ix_(active_positions, active_positions)]
        + model_b.covariance_regularized_active[np.ix_(active_positions, active_positions)]
    )
    return selected_full, covariance


def model_distance(
    model_a: ReferenceGestureModel,
    model_b: ReferenceGestureModel,
    *,
    metric: str,
    feature_mask: np.ndarray,
    fisher_weights: np.ndarray | None = None,
    epsilon: float = 1.0e-10,
) -> float:
    """Return a symmetric distance between two persisted gesture prototypes.

    This comparison is a model-space diagnostic. For variance-aware metrics the
    within-class uncertainty of both models is pooled symmetrically so that the
    distance from A to B equals the distance from B to A.
    """
    metric = metric.strip().lower()
    if metric not in MODEL_DISTANCE_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    mask = np.asarray(feature_mask, dtype=bool)
    a = np.asarray(model_a.prototype_mean, dtype=np.float64)[mask]
    b = np.asarray(model_b.prototype_mean, dtype=np.float64)[mask]
    delta = a - b
    if metric == "euclidean":
        return float(np.linalg.norm(delta))
    if metric == "manhattan":
        return float(np.sum(np.abs(delta)))
    if metric == "cosine":
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= epsilon:
            return 0.0 if np.allclose(a, b) else 1.0
        cosine = float(np.dot(a, b) / denominator)
        return float(1.0 - np.clip(cosine, -1.0, 1.0))

    variance = np.maximum(_pooled_variance(model_a, model_b)[mask], epsilon)
    if metric in {"standardized_euclidean", "diagonal_mahalanobis"}:
        return float(np.sqrt(np.sum((delta * delta) / variance)))
    if metric == "weighted_standardized_euclidean":
        if fisher_weights is None:
            raise ValueError("weighted_standardized_euclidean requires feature weights.")
        weights = _renormalized_weights(fisher_weights, mask)[mask]
        return float(np.sqrt(np.sum(weights * (delta * delta) / variance)))
    if metric == "weighted_manhattan":
        if fisher_weights is None:
            raise ValueError("weighted_manhattan requires feature weights.")
        weights = _renormalized_weights(fisher_weights, mask)[mask]
        scale = np.sqrt(variance)
        return float(np.sum(weights * np.abs(delta) / scale))
    if metric == "full_mahalanobis":
        # Full covariance exists only for active features. Inactive constants are
        # deliberately excluded even when the report uses feature_set='full'.
        selected_full, covariance = _active_covariance_subset(model_a, model_b, mask)
        delta_active = (
            np.asarray(model_a.prototype_mean, dtype=np.float64)[selected_full]
            - np.asarray(model_b.prototype_mean, dtype=np.float64)[selected_full]
        )
        inverse = np.linalg.pinv(covariance, hermitian=True)
        value = float(delta_active.T @ inverse @ delta_active)
        return float(math.sqrt(max(value, 0.0)))
    raise AssertionError(metric)


def pair_distance_diagnostics(
    space: ModelSpace,
    model_a: ReferenceGestureModel,
    model_b: ReferenceGestureModel,
    *,
    metric: str,
    feature_mask: np.ndarray,
) -> PairDistanceDiagnostics:
    """Return distance plus decomposable standardized feature contributions."""
    mask = np.asarray(feature_mask, dtype=bool)
    variance = np.maximum(_pooled_variance(model_a, model_b), 1.0e-10)
    delta = np.asarray(model_a.prototype_mean) - np.asarray(model_b.prototype_mean)
    standardized = np.zeros_like(delta, dtype=np.float64)
    standardized[mask] = (delta[mask] * delta[mask]) / variance[mask]
    weights = _renormalized_weights(space.fisher_weights, mask)
    weighted = standardized * weights
    return PairDistanceDiagnostics(
        gesture_a=model_a.gesture_id,
        gesture_b=model_b.gesture_id,
        distance=model_distance(
            model_a,
            model_b,
            metric=metric,
            feature_mask=mask,
            fisher_weights=space.fisher_weights,
        ),
        standardized_components=standardized,
        weighted_standardized_components=weighted,
    )


def distance_matrix(
    space: ModelSpace,
    *,
    metric: str,
    feature_mask: np.ndarray,
) -> np.ndarray:
    """Compute the symmetric pairwise gesture-model distance matrix."""
    count = len(space.models)
    result = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = model_distance(
                space.models[i],
                space.models[j],
                metric=metric,
                feature_mask=feature_mask,
                fisher_weights=space.fisher_weights,
            )
            result[i, j] = result[j, i] = value
    return result


def nearest_pairs(labels: Sequence[str], matrix: np.ndarray) -> list[tuple[str, str, float]]:
    """Return each unordered class pair sorted from most to least similar."""
    pairs: list[tuple[str, str, float]] = []
    for i, first in enumerate(labels):
        for j in range(i + 1, len(labels)):
            pairs.append((str(first), str(labels[j]), float(matrix[i, j])))
    return sorted(pairs, key=lambda row: (row[2], row[0], row[1]))


def nearest_neighbor_rows(labels: Sequence[str], matrix: np.ndarray) -> list[dict[str, Any]]:
    """Return a long nearest-neighbour table for every gesture."""
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        order = [index for index in np.argsort(matrix[i]) if index != i]
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "gesture_id": str(label),
                    "rank": rank,
                    "neighbor_gesture_id": str(labels[index]),
                    "distance": float(matrix[i, index]),
                }
            )
    return rows


def classical_mds(matrix: np.ndarray, dimensions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Classical metric multidimensional scaling from a distance matrix."""
    distances = np.asarray(matrix, dtype=np.float64)
    n = distances.shape[0]
    if distances.shape != (n, n):
        raise ValueError("Distance matrix must be square.")
    center = np.eye(n) - np.ones((n, n), dtype=np.float64) / float(n)
    gram = -0.5 * center @ (distances ** 2) @ center
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.maximum(eigenvalues[:dimensions], 0.0)
    coordinates = eigenvectors[:, :dimensions] * np.sqrt(positive)[None, :]
    return coordinates, eigenvalues


def prototype_pca(
    space: ModelSpace,
    feature_mask: np.ndarray,
    dimensions: int = 3,
    *,
    fisher_weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA of class prototypes after across-class feature standardization."""
    mask = np.asarray(feature_mask, dtype=bool)
    matrix = np.vstack([model.prototype_mean[mask] for model in space.models]).astype(np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    standardized = (matrix - mean) / scale
    if fisher_weighted:
        weights = _renormalized_weights(space.fisher_weights, mask)[mask]
        standardized = standardized * np.sqrt(np.maximum(weights, 0.0))[None, :]
    u, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    dimensions = min(dimensions, u.shape[1])
    coordinates = u[:, :dimensions] * singular[:dimensions][None, :]
    variance = singular ** 2
    ratio = variance / variance.sum() if float(variance.sum()) > 0 else variance
    return coordinates, ratio


def average_linkage_merges(matrix: np.ndarray) -> list[tuple[int, int, float, int]]:
    """Small deterministic average-linkage implementation for gesture counts < 100."""
    distances = np.asarray(matrix, dtype=np.float64)
    n = distances.shape[0]
    clusters: dict[int, tuple[int, ...]] = {index: (index,) for index in range(n)}
    active = list(range(n))
    merges: list[tuple[int, int, float, int]] = []
    next_id = n
    while len(active) > 1:
        best: tuple[float, int, int] | None = None
        for offset, first in enumerate(active):
            for second in active[offset + 1 :]:
                members_a = clusters[first]
                members_b = clusters[second]
                value = float(
                    np.mean([distances[i, j] for i in members_a for j in members_b])
                )
                candidate = (value, min(first, second), max(first, second))
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        height, first, second = best
        clusters[next_id] = tuple(sorted(clusters[first] + clusters[second]))
        merges.append((first, second, height, next_id))
        active = [item for item in active if item not in {first, second}]
        active.append(next_id)
        active.sort()
        next_id += 1
    return merges


def _write_matrix_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gesture_id", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in row]])


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(path: Path, labels: Sequence[str], matrix: np.ndarray, title: str) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    image = ax.imshow(matrix, aspect="equal")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Distance")
    if len(labels) <= 20:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    continue
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _dendrogram_layout(
    labels: Sequence[str], merges: Sequence[tuple[int, int, float, int]]
) -> tuple[dict[int, float], dict[int, float], dict[int, tuple[int, int]]]:
    n = len(labels)
    children = {new_id: (first, second) for first, second, _, new_id in merges}
    heights = {index: 0.0 for index in range(n)}
    heights.update({new_id: float(height) for _, _, height, new_id in merges})
    root = merges[-1][3]
    leaf_order: list[int] = []

    def walk(node: int) -> None:
        if node < n:
            leaf_order.append(node)
            return
        first, second = children[node]
        # Stable order based on the smallest source label index in each subtree.
        def minimum_leaf(item: int) -> int:
            if item < n:
                return item
            a, b = children[item]
            return min(minimum_leaf(a), minimum_leaf(b))
        if minimum_leaf(first) > minimum_leaf(second):
            first, second = second, first
        walk(first)
        walk(second)

    walk(root)
    x = {leaf: float(position) for position, leaf in enumerate(leaf_order)}

    def node_x(node: int) -> float:
        if node in x:
            return x[node]
        a, b = children[node]
        x[node] = 0.5 * (node_x(a) + node_x(b))
        return x[node]

    node_x(root)
    return x, heights, children


def _plot_dendrogram(
    path: Path,
    labels: Sequence[str],
    merges: Sequence[tuple[int, int, float, int]],
    title: str,
) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    x, heights, children = _dendrogram_layout(labels, merges)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for first, second, height, new_id in merges:
        x1, x2 = x[first], x[second]
        h1, h2 = heights[first], heights[second]
        ax.plot([x1, x1], [h1, height])
        ax.plot([x2, x2], [h2, height])
        ax.plot([x1, x2], [height, height])
    leaf_positions = sorted((position, leaf) for leaf, position in x.items() if leaf < len(labels))
    ax.set_xticks([position for position, _ in leaf_positions])
    ax.set_xticklabels([labels[leaf] for _, leaf in leaf_positions])
    ax.set_ylabel("Average-linkage distance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_embedding(
    path: Path,
    labels: Sequence[str],
    coordinates: np.ndarray,
    title: str,
    axis_labels: Sequence[str],
) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    dimensions = coordinates.shape[1]
    if dimensions == 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=55)
        for label, point in zip(labels, coordinates):
            ax.annotate(label, point[:2], xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1])
        ax.grid(alpha=0.25)
    elif dimensions == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=45)
        for label, point in zip(labels, coordinates):
            ax.text(point[0], point[1], point[2], label)
        ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1]); ax.set_zlabel(axis_labels[2])
    else:
        raise ValueError("Embedding plot supports exactly 2 or 3 dimensions.")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_similarity_graph(
    path: Path,
    labels: Sequence[str],
    matrix: np.ndarray,
    coordinates: np.ndarray,
    neighbors: int,
    title: str,
) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    count = len(labels)
    neighbors = max(1, min(int(neighbors), count - 1))
    edges: set[tuple[int, int]] = set()
    for i in range(count):
        order = [index for index in np.argsort(matrix[i]) if index != i][:neighbors]
        for j in order:
            edges.add((min(i, int(j)), max(i, int(j))))
    edge_distances = np.asarray([matrix[i, j] for i, j in edges], dtype=np.float64)
    median = float(np.median(edge_distances)) if edge_distances.size else 1.0
    median = max(median, 1.0e-12)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, j in sorted(edges):
        distance = float(matrix[i, j])
        width = 0.6 + 2.4 * math.exp(-distance / median)
        ax.plot(
            [coordinates[i, 0], coordinates[j, 0]],
            [coordinates[i, 1], coordinates[j, 1]],
            linewidth=width,
            alpha=0.42,
        )
        midpoint = 0.5 * (coordinates[i, :2] + coordinates[j, :2])
        ax.text(midpoint[0], midpoint[1], f"{distance:.2f}", fontsize=6, alpha=0.75)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=120, zorder=3)
    for label, point in zip(labels, coordinates):
        ax.text(point[0], point[1], label, ha="center", va="center", fontsize=9, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _landmarks_from_vector(feature_ids: Sequence[str], vector: np.ndarray) -> np.ndarray:
    lookup = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    points = np.zeros((len(LANDMARK_NAMES), 3), dtype=np.float64)
    for landmark_index, name in enumerate(LANDMARK_NAMES):
        for axis_index, axis in enumerate(("x", "y", "z")):
            points[landmark_index, axis_index] = float(vector[lookup[f"landmark.{name}.{axis}"]])
    return points


def _draw_hand_3d(ax: Any, points: np.ndarray, *, label: str | None = None, alpha: float = 1.0, linewidth: float = 1.8) -> None:
    for start, end in HAND_CONNECTIONS:
        ax.plot(
            [points[start, 0], points[end, 0]],
            [points[start, 2], points[end, 2]],
            [points[start, 1], points[end, 1]],
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=9, alpha=alpha, label=label)


def _equal_3d_limits(ax: Any, point_sets: Sequence[np.ndarray]) -> None:
    all_points = np.vstack(point_sets)
    plotted = np.column_stack([all_points[:, 0], all_points[:, 2], all_points[:, 1]])
    center = plotted.mean(axis=0)
    radius = max(float(np.ptp(plotted, axis=0).max()) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_skeleton_grid(
    path: Path,
    space: ModelSpace,
    *,
    elev: float = 18.0,
    azim: float = -80.0,
    view_name: str = "oblique",
) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    count = len(space.models)
    columns = 4
    rows = int(math.ceil(count / columns))
    fig = plt.figure(figsize=(16, 4.1 * rows))
    prototype_points = [_landmarks_from_vector(space.feature_ids, model.prototype_mean) for model in space.models]
    for index, (model, points) in enumerate(zip(space.models, prototype_points), start=1):
        ax = fig.add_subplot(rows, columns, index, projection="3d")
        if model.person_means.size:
            for person_vector in model.person_means:
                _draw_hand_3d(
                    ax,
                    _landmarks_from_vector(space.feature_ids, person_vector),
                    alpha=0.18,
                    linewidth=0.8,
                )
        _draw_hand_3d(ax, points, alpha=1.0, linewidth=2.0)
        _equal_3d_limits(ax, [points])
        ax.set_title(model.gesture_id)
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(
        f"TSGR-F mean canonical hand skeletons | {view_name} view "
        "(faint lines: per-person means)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_skeleton_multiview_grids(output_dir: Path, space: ModelSpace) -> list[dict[str, Any]]:
    """Generate the same gesture grid from canonical orthogonal and oblique views."""
    views = [
        ("front", 0.0, -90.0),
        ("side", 0.0, 0.0),
        ("top", 90.0, -90.0),
        ("oblique", 18.0, -80.0),
    ]
    rows: list[dict[str, Any]] = []
    for name, elev, azim in views:
        path = output_dir / f"mean_skeleton_grid_{name}.png"
        _plot_skeleton_grid(path, space, elev=elev, azim=azim, view_name=name)
        rows.append({"view": name, "elevation_deg": elev, "azimuth_deg": azim, "path": str(path)})
    # Keep the canonical main filename for the oblique view.
    main_output = output_dir / "mean_skeleton_grid.png"
    _plot_skeleton_grid(main_output, space, elev=18.0, azim=-80.0, view_name="oblique")
    return rows


def _plot_pair_skeleton_overlay(
    path: Path,
    space: ModelSpace,
    model_a: ReferenceGestureModel,
    model_b: ReferenceGestureModel,
    distance: float,
) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    a = _landmarks_from_vector(space.feature_ids, model_a.prototype_mean)
    b = _landmarks_from_vector(space.feature_ids, model_b.prototype_mean)
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    _draw_hand_3d(ax, a, label=model_a.gesture_id, alpha=0.9, linewidth=2.2)
    _draw_hand_3d(ax, b, label=model_b.gesture_id, alpha=0.9, linewidth=2.2)
    _equal_3d_limits(ax, [a, b])
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.view_init(elev=18, azim=-80)
    ax.legend(loc="best")
    ax.set_title(f"Prototype overlay: {model_a.gesture_id} vs {model_b.gesture_id} | distance={distance:.3f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _group_contribution_rows(
    space: ModelSpace,
    diagnostics: PairDistanceDiagnostics,
    feature_mask: np.ndarray,
    *,
    weighted: bool,
) -> list[dict[str, Any]]:
    components = (
        diagnostics.weighted_standardized_components
        if weighted
        else diagnostics.standardized_components
    )
    total = float(components[feature_mask].sum())
    rows: list[dict[str, Any]] = []
    for group in space.feature_groups:
        group_mask = np.asarray(
            [name == group for name in space.feature_group_by_index], dtype=bool
        ) & feature_mask
        value = float(components[group_mask].sum())
        rows.append(
            {
                "gesture_a": diagnostics.gesture_a,
                "gesture_b": diagnostics.gesture_b,
                "group": group,
                "standardized_separation_contribution": value,
                "fraction": value / total if total > 0 else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["fraction"]), str(row["group"])))


def _plot_group_contributions(path: Path, rows: Sequence[dict[str, Any]], title: str) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    filtered = [row for row in rows if float(row["fraction"]) > 0]
    labels = [str(row["group"]) for row in filtered][::-1]
    values = [100.0 * float(row["fraction"]) for row in filtered][::-1]
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.38 * len(labels) + 2)))
    ax.barh(np.arange(len(labels)), values)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Share of standardized separation [%]")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _top_feature_rows(
    space: ModelSpace,
    diagnostics: PairDistanceDiagnostics,
    feature_mask: np.ndarray,
    *,
    weighted: bool,
    top_n: int,
) -> list[dict[str, Any]]:
    components = diagnostics.weighted_standardized_components if weighted else diagnostics.standardized_components
    indices = np.flatnonzero(feature_mask)
    ordered = sorted(indices.tolist(), key=lambda index: (-float(components[index]), index))[:top_n]
    total = float(components[feature_mask].sum())
    model_lookup = {model.gesture_id: model for model in space.models}
    a = model_lookup[diagnostics.gesture_a]
    b = model_lookup[diagnostics.gesture_b]
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(ordered, start=1):
        pooled_std = math.sqrt(max(float(_pooled_variance(a, b)[index]), 1.0e-10))
        midpoint = 0.5 * (float(a.prototype_mean[index]) + float(b.prototype_mean[index]))
        rows.append(
            {
                "gesture_a": diagnostics.gesture_a,
                "gesture_b": diagnostics.gesture_b,
                "rank": rank,
                "feature_index": index,
                "feature_id": space.feature_ids[index],
                "group": space.feature_group_by_index[index],
                "value_a": float(a.prototype_mean[index]),
                "value_b": float(b.prototype_mean[index]),
                "pooled_std": pooled_std,
                "z_a_about_midpoint": (float(a.prototype_mean[index]) - midpoint) / pooled_std,
                "z_b_about_midpoint": (float(b.prototype_mean[index]) - midpoint) / pooled_std,
                "separation_component": float(components[index]),
                "fraction": float(components[index]) / total if total > 0 else 0.0,
            }
        )
    return rows


def _plot_feature_profile(path: Path, rows: Sequence[dict[str, Any]], title: str) -> None:
    configure_headless_backend()
    import matplotlib.pyplot as plt

    labels = [str(row["feature_id"]) for row in rows][::-1]
    a_values = [float(row["z_a_about_midpoint"]) for row in rows][::-1]
    b_values = [float(row["z_b_about_midpoint"]) for row in rows][::-1]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(labels) + 2)))
    ax.barh(y - 0.18, a_values, height=0.34, label=str(rows[0]["gesture_a"]) if rows else "A")
    ax.barh(y + 0.18, b_values, height=0.34, label=str(rows[0]["gesture_b"]) if rows else "B")
    ax.set_yticks(y, labels=labels)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("Prototype value relative to pair midpoint [pooled within-class SD]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_model_space_report(
    model_set_dir: str | Path,
    *,
    branch_name: str,
    output_dir: str | Path,
    metric: str = "weighted_standardized_euclidean",
    feature_set: str = "active",
    feature_groups: Sequence[str] | None = None,
    top_pairs: int = 10,
    graph_neighbors: int = 2,
    top_features_per_pair: int = 20,
    per_group_heatmaps: bool = True,
    separation_concentration_warning: float = 0.50,
) -> Path:
    """Generate a complete static report for similarity between reference models."""
    metric = metric.strip().lower()
    if metric not in MODEL_DISTANCE_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}. Choices: {MODEL_DISTANCE_METRICS}")
    space = load_model_space(model_set_dir, branch_name)
    mask = selected_feature_mask(space, feature_set=feature_set, groups=feature_groups)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    matrix = distance_matrix(space, metric=metric, feature_mask=mask)
    pairs = nearest_pairs(space.gesture_ids, matrix)
    _write_matrix_csv(output / "distance_matrix.csv", space.gesture_ids, matrix)
    _write_rows(
        output / "nearest_neighbors.csv",
        nearest_neighbor_rows(space.gesture_ids, matrix),
        ["gesture_id", "rank", "neighbor_gesture_id", "distance"],
    )
    _write_rows(
        output / "nearest_pairs.csv",
        [
            {"rank": rank, "gesture_a": first, "gesture_b": second, "distance": distance}
            for rank, (first, second, distance) in enumerate(pairs, start=1)
        ],
        ["rank", "gesture_a", "gesture_b", "distance"],
    )

    _plot_heatmap(
        output / "plots" / "distance_heatmap.png",
        space.gesture_ids,
        matrix,
        f"TSGR-F model distances | {branch_name} | {metric}",
    )
    merges = average_linkage_merges(matrix)
    _plot_dendrogram(
        output / "plots" / "dendrogram.png",
        space.gesture_ids,
        merges,
        f"Average-linkage clustering | {metric}",
    )
    mds2, eigenvalues = classical_mds(matrix, 2)
    _plot_embedding(
        output / "plots" / "mds_2d.png",
        space.gesture_ids,
        mds2,
        f"Classical MDS of gesture models | {metric}",
        ("MDS 1", "MDS 2"),
    )
    mds3, _ = classical_mds(matrix, 3)
    _plot_embedding(
        output / "plots" / "mds_3d.png",
        space.gesture_ids,
        mds3,
        f"Classical MDS 3D | {metric}",
        ("MDS 1", "MDS 2", "MDS 3"),
    )
    pca3, explained = prototype_pca(
        space,
        mask,
        3,
        fisher_weighted=metric.startswith("weighted_"),
    )
    _plot_embedding(
        output / "plots" / "pca_2d.png",
        space.gesture_ids,
        pca3[:, :2],
        "PCA of standardized gesture prototypes",
        (
            f"PC1 ({100.0*explained[0]:.1f}%)",
            f"PC2 ({100.0*explained[1]:.1f}%)",
        ),
    )
    _plot_embedding(
        output / "plots" / "pca_3d.png",
        space.gesture_ids,
        pca3[:, :3],
        "PCA 3D of standardized gesture prototypes",
        (
            f"PC1 ({100.0*explained[0]:.1f}%)",
            f"PC2 ({100.0*explained[1]:.1f}%)",
            f"PC3 ({100.0*explained[2]:.1f}%)",
        ),
    )
    _plot_similarity_graph(
        output / "plots" / "similarity_graph.png",
        space.gesture_ids,
        matrix,
        mds2,
        graph_neighbors,
        f"Nearest-neighbour gesture graph | {metric}",
    )
    skeleton_views = _plot_skeleton_multiview_grids(output / "plots", space)

    model_lookup = {model.gesture_id: model for model in space.models}
    all_group_rows: list[dict[str, Any]] = []
    all_feature_rows: list[dict[str, Any]] = []
    separation_warning_rows: list[dict[str, Any]] = []
    pair_limit = min(max(int(top_pairs), 0), len(pairs))
    weighted_diagnostics = metric in {"weighted_standardized_euclidean", "weighted_manhattan"}
    for rank, (first, second, distance) in enumerate(pairs[:pair_limit], start=1):
        diagnostics = pair_distance_diagnostics(
            space,
            model_lookup[first],
            model_lookup[second],
            metric=metric,
            feature_mask=mask,
        )
        group_rows = _group_contribution_rows(
            space, diagnostics, mask, weighted=weighted_diagnostics
        )
        for row in group_rows:
            row["pair_rank"] = rank
            row["distance"] = distance
        all_group_rows.extend(group_rows)
        feature_rows = _top_feature_rows(
            space,
            diagnostics,
            mask,
            weighted=weighted_diagnostics,
            top_n=top_features_per_pair,
        )
        for row in feature_rows:
            row["pair_rank"] = rank
            row["distance"] = distance
        all_feature_rows.extend(feature_rows)
        if feature_rows and float(feature_rows[0]["fraction"]) >= separation_concentration_warning:
            separation_warning_rows.append(
                {
                    "pair_rank": rank,
                    "gesture_a": first,
                    "gesture_b": second,
                    "distance": distance,
                    "warning_type": "single_feature_dependency",
                    "component": feature_rows[0]["feature_id"],
                    "group": feature_rows[0]["group"],
                    "fraction": float(feature_rows[0]["fraction"]),
                    "threshold": separation_concentration_warning,
                    "interpretation": "diagnostic_only_expected_or_spurious_dependency_requires_domain_review",
                }
            )
        if group_rows and float(group_rows[0]["fraction"]) >= separation_concentration_warning:
            separation_warning_rows.append(
                {
                    "pair_rank": rank,
                    "gesture_a": first,
                    "gesture_b": second,
                    "distance": distance,
                    "warning_type": "single_group_dependency",
                    "component": group_rows[0]["group"],
                    "group": group_rows[0]["group"],
                    "fraction": float(group_rows[0]["fraction"]),
                    "threshold": separation_concentration_warning,
                    "interpretation": "diagnostic_only_expected_or_spurious_dependency_requires_domain_review",
                }
            )
        stem = f"{rank:02d}_{first}_{second}"
        _plot_pair_skeleton_overlay(
            output / "plots" / "nearest_pair_skeletons" / f"{stem}.png",
            space,
            model_lookup[first],
            model_lookup[second],
            distance,
        )
        _plot_group_contributions(
            output / "plots" / "group_contributions" / f"{stem}.png",
            group_rows,
            f"Feature-group separation: {first} vs {second}",
        )
        _plot_feature_profile(
            output / "plots" / "feature_profiles" / f"{stem}.png",
            feature_rows,
            f"Top separating features: {first} vs {second}",
        )

    _write_rows(
        output / "pair_group_contributions.csv",
        all_group_rows,
        [
            "pair_rank", "gesture_a", "gesture_b", "distance", "group",
            "standardized_separation_contribution", "fraction",
        ],
    )
    _write_rows(
        output / "pair_feature_contributions.csv",
        all_feature_rows,
        [
            "pair_rank", "gesture_a", "gesture_b", "distance", "rank",
            "feature_index", "feature_id", "group", "value_a", "value_b",
            "pooled_std", "z_a_about_midpoint", "z_b_about_midpoint",
            "separation_component", "fraction",
        ],
    )

    _write_rows(
        output / "pair_separation_warnings.csv",
        separation_warning_rows,
        [
            "pair_rank", "gesture_a", "gesture_b", "distance", "warning_type",
            "component", "group", "fraction", "threshold", "interpretation",
        ],
    )

    if per_group_heatmaps:
        for group in space.feature_groups:
            group_mask = selected_feature_mask(space, feature_set=feature_set, groups=[group])
            if not group_mask.any():
                continue
            # Full Mahalanobis is valid only when the group contains active features;
            # the distance implementation already projects onto those active entries.
            group_matrix = distance_matrix(space, metric=metric, feature_mask=group_mask)
            safe = group.replace("/", "_").replace(" ", "_")
            _write_matrix_csv(
                output / "group_distance_matrices" / f"{safe}.csv",
                space.gesture_ids,
                group_matrix,
            )
            _plot_heatmap(
                output / "plots" / "group_distance_heatmaps" / f"{safe}.png",
                space.gesture_ids,
                group_matrix,
                f"Model distance using only group: {group}",
            )

    summary = {
        "model_set": str(Path(model_set_dir).resolve()),
        "branch_name": branch_name,
        "metric": metric,
        "feature_set": feature_set,
        "feature_groups": list(feature_groups or []),
        "gesture_count": len(space.gesture_ids),
        "gesture_ids": list(space.gesture_ids),
        "selected_feature_count": int(mask.sum()),
        "active_feature_count": int(space.active_mask.sum()),
        "compact_feature_count": int(space.compact_mask.sum()),
        "skeleton_views": skeleton_views,
        "separation_concentration_warning_threshold": float(separation_concentration_warning),
        "separation_warning_count": len(separation_warning_rows),
        "top_pairs": [
            {"rank": rank, "gesture_a": first, "gesture_b": second, "distance": distance}
            for rank, (first, second, distance) in enumerate(pairs[:pair_limit], start=1)
        ],
        "mds_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "pca_explained_variance_ratio": [float(value) for value in explained.tolist()],
        "notes": [
            "Distances are diagnostic symmetric model-to-model distances, not calibrated probabilities.",
            "Group/feature contribution plots decompose pooled within-class standardized prototype separation; for full Mahalanobis they are explanatory diagnostics rather than an additive decomposition of the covariance metric.",
            "The report uses the persisted branch feature mask and feature schema from the supplied model set.",
            "feature_set=compact uses the fold-local nonredundant mask generated only from reference-training data; old model sets without this artifact fall back to the active mask.",
            "Pair-separation concentration warnings are diagnostic only: a high fraction may be fully expected when two gestures differ mainly in one anatomical relation.",
        ],
    }
    (output / "model_space_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
