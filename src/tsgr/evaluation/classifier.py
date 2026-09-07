"""Vectorized TSGRF prototype classifier with explicit acceptance geometry.

Class ranking and acceptance are intentionally separated. Ranking always uses
prototype distance. Acceptance can use a linear standard-deviation
radius, a gesture-specific linear radius covering all valid training samples,
or a gesture-specific anisotropic standardized boundary covering all valid
training samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from tsgr.visualization.reference_model_space import ModelSpace, load_model_space, selected_feature_mask

DistanceMetric = Literal["euclidean", "weighted_euclidean", "manhattan", "weighted_manhattan"]
ACCEPTANCE_BOUNDARY_MODES = (
    "linear_std",
    "per_class_max_linear",
    "per_class_anisotropic_max",
)
ORIENTATION_MODES = ("camera_aware", "rotation_invariant")


@dataclass(frozen=True, slots=True)
class AdaptiveThresholdConfig:
    mode: str = "model_deviation"
    deviation_measure: str = "std"
    deviation_multiplier: float = 1.0
    global_fixed_threshold: float | None = None
    calibration_source: str = "prototype_person_means"
    boundary_mode: str = "linear_std"
    training_acceptance_margin: float = 1.0e-9
    require_training_self_classification: bool = False

    def validate(self) -> None:
        if self.mode not in {"model_deviation", "global_fixed"}:
            raise ValueError("acceptance threshold mode must be 'model_deviation' or 'global_fixed'.")
        if self.deviation_measure != "std":
            raise ValueError("TSGR-F implements the explicit deviation measure 'std'.")
        if self.calibration_source not in {"prototype_person_means", "all_training_complete"}:
            raise ValueError(
                "acceptance calibration source must be 'prototype_person_means' or 'all_training_complete'."
            )
        if self.boundary_mode not in ACCEPTANCE_BOUNDARY_MODES:
            raise ValueError(f"Unsupported acceptance boundary mode: {self.boundary_mode}")
        if self.deviation_multiplier <= 0:
            raise ValueError("deviation_multiplier must be positive.")
        if self.training_acceptance_margin < 0:
            raise ValueError("training_acceptance_margin must be non-negative.")
        if self.boundary_mode != "linear_std" and self.calibration_source != "all_training_complete":
            raise ValueError(
                "Per-class training-complete acceptance boundaries require calibration_source='all_training_complete'."
            )
        if self.mode == "global_fixed":
            if self.global_fixed_threshold is None or float(self.global_fixed_threshold) <= 0:
                raise ValueError("global_fixed mode requires a positive global_fixed_threshold.")
            if self.boundary_mode != "linear_std":
                raise ValueError("global_fixed threshold is compatible only with boundary_mode='linear_std'.")


@dataclass(slots=True)
class TSGRFClassificationBatch:
    top1_index: np.ndarray
    top2_index: np.ndarray
    top3_index: np.ndarray
    top1_distance: np.ndarray
    top2_distance: np.ndarray
    top3_distance: np.ndarray
    accepted: np.ndarray
    normalized_distance: np.ndarray
    quality_score: np.ndarray
    gap_1_2: np.ndarray
    gap_1_3: np.ndarray
    relative_gap_1_2: np.ndarray
    relative_gap_1_3: np.ndarray
    ranking_confidence: np.ndarray
    distances: np.ndarray
    acceptance_score_top1: np.ndarray
    acceptance_threshold_top1: np.ndarray
    acceptance_scores: np.ndarray


class TSGRFClassifier:
    """Classify complete TSGR-F feature vectors against one fold-local model set."""

    def __init__(
        self,
        model_set_dir: str | Path,
        *,
        branch_name: str = "raw.wrist_middle_mcp",
        feature_set: str = "compact",
        distance_metric: DistanceMetric = "weighted_euclidean",
        threshold_config: AdaptiveThresholdConfig | None = None,
        threshold_standard_deviations: np.ndarray | None = None,
        threshold_calibration_values: dict[str, np.ndarray] | None = None,
        orientation_mode: str = "camera_aware",
        epsilon: float = 1.0e-12,
    ) -> None:
        self.model_set_dir = Path(model_set_dir)
        self.space: ModelSpace = load_model_space(self.model_set_dir, branch_name)
        self.branch_name = branch_name
        self.orientation_mode = str(orientation_mode).strip().lower()
        if self.orientation_mode not in ORIENTATION_MODES:
            raise ValueError(f"Unsupported orientation mode: {orientation_mode}")
        mask = selected_feature_mask(self.space, feature_set=feature_set)
        if self.orientation_mode == "rotation_invariant":
            orientation_mask = np.asarray(
                [not str(feature_id).startswith("orientation.") for feature_id in self.space.feature_ids],
                dtype=bool,
            )
            mask &= orientation_mask
            if not mask.any():
                raise ValueError("rotation_invariant orientation mode removed every selected feature.")
        self.feature_mask = mask
        self.feature_indices = np.flatnonzero(self.feature_mask)
        self.distance_metric: DistanceMetric = distance_metric
        if distance_metric not in {"euclidean", "weighted_euclidean", "manhattan", "weighted_manhattan"}:
            raise ValueError(f"Unsupported TSGRF classifier distance metric: {distance_metric}")
        self.threshold_config = threshold_config or AdaptiveThresholdConfig()
        self.threshold_config.validate()
        self.epsilon = float(epsilon)
        self.gesture_ids = tuple(self.space.gesture_ids)
        self.prototypes = np.vstack(
            [np.asarray(model.prototype_mean, dtype=np.float64)[self.feature_mask] for model in self.space.models]
        )
        model_standard_deviations = np.vstack(
            [np.maximum(np.asarray(model.standard_deviation, dtype=np.float64), 0.0) for model in self.space.models]
        )
        if threshold_standard_deviations is None:
            full_threshold_deviations = model_standard_deviations
        else:
            full_threshold_deviations = np.asarray(threshold_standard_deviations, dtype=np.float64)
            expected = (len(self.space.models), len(self.space.feature_ids))
            if full_threshold_deviations.shape != expected:
                raise ValueError(
                    f"threshold_standard_deviations must have shape {expected}, got {full_threshold_deviations.shape}."
                )
            if not np.isfinite(full_threshold_deviations).all():
                raise ValueError("threshold_standard_deviations must be finite.")
            full_threshold_deviations = np.maximum(full_threshold_deviations, 0.0)
        self.standard_deviations = full_threshold_deviations[:, self.feature_mask]
        raw_weights = np.maximum(np.asarray(self.space.fisher_weights, dtype=np.float64)[self.feature_mask], 0.0)
        if raw_weights.size == 0 or not np.isfinite(raw_weights).all() or float(raw_weights.mean()) <= 0:
            raw_weights = np.ones(self.feature_indices.size, dtype=np.float64)
        else:
            raw_weights = raw_weights / float(raw_weights.mean())
        self.weights = raw_weights
        self._calibration_values = threshold_calibration_values
        self.base_thresholds = self._build_linear_std_thresholds(multiplier=1.0)
        self.class_multipliers = np.ones(len(self.gesture_ids), dtype=np.float64)
        self.thresholds = self._build_thresholds()

    @property
    def selected_feature_count(self) -> int:
        return int(self.feature_indices.size)

    @property
    def removed_orientation_feature_count(self) -> int:
        return int(
            sum(
                1
                for index, feature_id in enumerate(self.space.feature_ids)
                if str(feature_id).startswith("orientation.") and not self.feature_mask[index]
            )
        )

    def _distance_vectors(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum(delta * delta, axis=-1))
        if self.distance_metric == "weighted_euclidean":
            return np.sqrt(np.sum(self.weights * delta * delta, axis=-1))
        if self.distance_metric == "manhattan":
            return np.sum(np.abs(delta), axis=-1)
        if self.distance_metric == "weighted_manhattan":
            return np.sum(self.weights * np.abs(delta), axis=-1)
        raise AssertionError(self.distance_metric)

    def _build_linear_std_thresholds(self, *, multiplier: float) -> np.ndarray:
        deviations = float(multiplier) * self.standard_deviations
        zeros = np.zeros_like(deviations)
        return np.maximum(self._distance_vectors(deviations, zeros), self.epsilon)

    def _selected_calibration_matrix(self, gesture_id: str) -> np.ndarray:
        if self._calibration_values is None or gesture_id not in self._calibration_values:
            raise ValueError(
                f"Acceptance boundary {self.threshold_config.boundary_mode!r} requires full training values for {gesture_id}."
            )
        values = np.asarray(self._calibration_values[gesture_id], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.space.feature_ids) or not np.isfinite(values).all():
            raise ValueError(f"Invalid calibration values for {gesture_id}.")
        return values[:, self.feature_mask]

    def _anisotropic_scores_selected(self, selected: np.ndarray) -> np.ndarray:
        # output: samples x gestures. The per-feature std creates an ellipsoidal
        # boundary; the class-specific maximum score then covers every valid
        # training sample without isotropically inflating every direction.
        scale = np.maximum(self.standard_deviations, self.epsilon)
        delta = (selected[:, None, :] - self.prototypes[None, :, :]) / scale[None, :, :]
        if self.distance_metric in {"euclidean", "weighted_euclidean"}:
            if self.distance_metric == "weighted_euclidean":
                return np.sqrt(np.sum(self.weights[None, None, :] * delta * delta, axis=2))
            return np.sqrt(np.sum(delta * delta, axis=2))
        absolute = np.abs(delta)
        if self.distance_metric == "weighted_manhattan":
            return np.sum(self.weights[None, None, :] * absolute, axis=2)
        return np.sum(absolute, axis=2)

    def _build_thresholds(self) -> np.ndarray:
        config = self.threshold_config
        if config.mode == "global_fixed":
            return np.full(len(self.gesture_ids), float(config.global_fixed_threshold), dtype=np.float64)
        if config.boundary_mode == "linear_std":
            return self._build_linear_std_thresholds(multiplier=config.deviation_multiplier)

        if self._calibration_values is None:
            raise ValueError(f"{config.boundary_mode} requires threshold_calibration_values.")

        thresholds: list[float] = []
        if config.boundary_mode == "per_class_max_linear":
            for index, gesture_id in enumerate(self.gesture_ids):
                selected = self._selected_calibration_matrix(gesture_id)
                distances = self._distance_vectors(selected, self.prototypes[index])
                threshold = float(np.max(distances)) * float(config.deviation_multiplier)
                threshold += max(config.training_acceptance_margin, self.epsilon)
                thresholds.append(max(threshold, self.epsilon))
            result = np.asarray(thresholds, dtype=np.float64)
            self.class_multipliers = result / np.maximum(self.base_thresholds, self.epsilon)
            return result

        if config.boundary_mode == "per_class_anisotropic_max":
            for index, gesture_id in enumerate(self.gesture_ids):
                selected = self._selected_calibration_matrix(gesture_id)
                scores = self._anisotropic_scores_selected(selected)[:, index]
                threshold = float(np.max(scores)) * float(config.deviation_multiplier)
                threshold += max(config.training_acceptance_margin, self.epsilon)
                thresholds.append(max(threshold, self.epsilon))
            result = np.asarray(thresholds, dtype=np.float64)
            # For anisotropic geometry this is a dimensionless reporting factor
            # relative to the mean score of a one-sigma displacement.
            one_sigma = self._anisotropic_scores_selected(self.prototypes + self.standard_deviations)
            diagonal = np.asarray([one_sigma[i, i] for i in range(len(self.gesture_ids))], dtype=np.float64)
            self.class_multipliers = result / np.maximum(diagonal, self.epsilon)
            return result
        raise AssertionError(config.boundary_mode)

    def distance_to_gesture(self, values: np.ndarray, gesture_id: str) -> np.ndarray:
        index = self.gesture_ids.index(str(gesture_id).upper())
        selected = np.asarray(values, dtype=np.float64)[..., self.feature_mask]
        return self._distance_vectors(selected, self.prototypes[index])

    def acceptance_score_to_gesture(self, values: np.ndarray, gesture_id: str) -> np.ndarray:
        index = self.gesture_ids.index(str(gesture_id).upper())
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        selected = matrix[:, self.feature_mask]
        if self.threshold_config.boundary_mode == "per_class_anisotropic_max":
            return self._anisotropic_scores_selected(selected)[:, index]
        return self._distance_vectors(selected, self.prototypes[index])

    def classify(self, values: np.ndarray) -> TSGRFClassificationBatch:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(self.space.feature_ids):
            raise ValueError(
                f"Expected feature matrix with {len(self.space.feature_ids)} columns, got {matrix.shape}."
            )
        selected = matrix[:, self.feature_mask]
        if not np.isfinite(selected).all():
            raise ValueError("TSGRFClassifier.classify expects complete finite feature vectors.")
        delta = selected[:, None, :] - self.prototypes[None, :, :]
        if self.distance_metric == "euclidean":
            distances = np.sqrt(np.sum(delta * delta, axis=2))
        elif self.distance_metric == "weighted_euclidean":
            distances = np.sqrt(np.sum(self.weights[None, None, :] * delta * delta, axis=2))
        elif self.distance_metric == "manhattan":
            distances = np.sum(np.abs(delta), axis=2)
        else:
            distances = np.sum(self.weights[None, None, :] * np.abs(delta), axis=2)

        order = np.argsort(distances, axis=1, kind="stable")[:, :3]
        rows = np.arange(matrix.shape[0])
        top1 = order[:, 0]
        top2 = order[:, 1]
        top3 = order[:, 2]
        d1 = distances[rows, top1]
        d2 = distances[rows, top2]
        d3 = distances[rows, top3]

        if self.threshold_config.boundary_mode == "per_class_anisotropic_max":
            acceptance_scores = self._anisotropic_scores_selected(selected)
        else:
            acceptance_scores = distances.copy()
        acceptance_score = acceptance_scores[rows, top1]
        threshold = self.thresholds[top1]
        normalized = acceptance_score / np.maximum(threshold, self.epsilon)
        quality = np.exp2(-normalized)
        gap12 = d2 - d1
        gap13 = d3 - d1
        relative12 = gap12 / np.maximum(d2, self.epsilon)
        relative13 = gap13 / np.maximum(d3, self.epsilon)
        confidence = np.clip((2.0 / 3.0) * relative12 + (1.0 / 3.0) * relative13, 0.0, 1.0)
        return TSGRFClassificationBatch(
            top1_index=top1,
            top2_index=top2,
            top3_index=top3,
            top1_distance=d1,
            top2_distance=d2,
            top3_distance=d3,
            accepted=acceptance_score <= threshold,
            normalized_distance=normalized,
            quality_score=quality,
            gap_1_2=gap12,
            gap_1_3=gap13,
            relative_gap_1_2=relative12,
            relative_gap_1_3=relative13,
            ranking_confidence=confidence,
            distances=distances,
            acceptance_score_top1=acceptance_score,
            acceptance_threshold_top1=threshold,
            acceptance_scores=acceptance_scores,
        )
