"""Plan-level TSGRF evaluation using cached test features and fold-local reference models."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tsgr.dataset.contract import resolve_dataset_root_from_plan
from tsgr.evaluation.acceptance_calibration import load_fold_acceptance_calibration
from tsgr.evaluation.classifier import AdaptiveThresholdConfig, TSGRFClassifier
from tsgr.evaluation.ground_truth import NO_GESTURE, NO_HAND, UNANNOTATED, state_gesture_id
from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows, write_workbook
from tsgr.evaluation.temporal_sequence import sequence_metrics
from tsgr.reference_models.repository import branch_directory_name
from tsgr.utils.serialization import write_json
from tsgr.visualization.matplotlib_backend import configure_headless_backend
from tsgr.visualization.reference_model_space import load_model_space

configure_headless_backend()
import matplotlib.pyplot as plt

METHOD_ID = "TSGRF"
METHOD_SORT_ORDER = 9999
MISSING_HAND = "MISSING_HAND"


@dataclass(slots=True)
class FoldSpec:
    scenario: str
    fold_id: str
    fold_dir: Path
    model_set_dir: Path
    test_take_ids: set[str]
    classifier: TSGRFClassifier
    acceptance_calibration_rows: list[dict[str, Any]] = field(default_factory=list)
    training_self_classification_ok: bool = True


@dataclass(slots=True)
class FoldAccumulator:
    true_pred: Counter[tuple[str, str]] = field(default_factory=Counter)
    gesture_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    evaluated_frames: int = 0
    gesture_frames: int = 0
    no_gesture_frames: int = 0
    no_hand_frames: int = 0
    missing_predictions: int = 0
    accepted_gesture_predictions: int = 0
    gesture_frames_with_prediction_input: int = 0
    raw_top1_correct: int = 0
    raw_top2_correct: int = 0
    raw_top3_correct: int = 0
    accepted_on_gt_gesture_frames: int = 0
    accepted_correct_on_gt_gesture_frames: int = 0
    correct_no_hand_missing: int = 0
    unexpected_missing_hand: int = 0
    classification_batches: list[float] = field(default_factory=list)
    ranking_confidences: list[float] = field(default_factory=list)
    top1_quality_scores: list[float] = field(default_factory=list)
    gt_quality_scores: list[float] = field(default_factory=list)
    top1_normalized_distances: list[float] = field(default_factory=list)
    gt_normalized_distances: list[float] = field(default_factory=list)


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_q1": math.nan,
            f"{prefix}_q3": math.nan,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_q1": float(np.quantile(array, 0.25)),
        f"{prefix}_q3": float(np.quantile(array, 0.75)),
    }


def _safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _optional_divide(a: float, b: float) -> float:
    return float(a / b) if b else math.nan


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def _binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "mcc": _mcc(tp, fp, tn, fn),
    }


def _find_model_set(fold_dir: Path, model_variant: str | None) -> Path:
    root = fold_dir / "reference_models" if not model_variant else fold_dir / "reference_model_variants" / model_variant
    candidates = sorted(path.parent for path in root.glob("*/model_set.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one model set for {fold_dir} variant={model_variant!r}; found {len(candidates)}."
        )
    return candidates[0]


def _calibration_rows_for_classifier(
    *,
    scenario: str,
    fold_id: str,
    classifier: TSGRFClassifier,
    calibration_source: str,
    calibration_values: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gesture_index, gesture_id in enumerate(classifier.gesture_ids):
        if calibration_values is None:
            model = classifier.space.models[gesture_index]
            values = np.asarray(model.person_means, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] == 0:
                values = np.asarray(model.prototype_mean, dtype=np.float64)[None, :]
        else:
            values = np.asarray(calibration_values[gesture_id], dtype=np.float64)
        distances = np.asarray(classifier.distance_to_gesture(values, gesture_id), dtype=np.float64)
        acceptance_scores = np.asarray(classifier.acceptance_score_to_gesture(values, gesture_id), dtype=np.float64)
        current_threshold = float(classifier.thresholds[gesture_index])
        base_threshold = float(classifier.base_thresholds[gesture_index])
        ratios_k1 = distances / max(base_threshold, 1.0e-12)
        batch = classifier.classify(values)
        own_index = classifier.gesture_ids.index(gesture_id)
        raw_top1_correct = np.asarray(batch.top1_index == own_index, dtype=bool)
        accepted_correct = raw_top1_correct & np.asarray(batch.accepted, dtype=bool)
        rows.append(
            {
                "method_id": METHOD_ID,
                "scenario": scenario,
                "fold_id": fold_id,
                "gesture_id": gesture_id,
                "calibration_source": calibration_source,
                "acceptance_boundary_mode": classifier.threshold_config.boundary_mode,
                "orientation_mode": classifier.orientation_mode,
                "calibration_sample_count": int(values.shape[0]),
                "base_linear_std_threshold_k1": base_threshold,
                "acceptance_threshold_current": current_threshold,
                "class_acceptance_multiplier": float(classifier.class_multipliers[gesture_index]),
                "deviation_multiplier": float(classifier.threshold_config.deviation_multiplier),
                "training_raw_top1_accuracy": float(np.mean(raw_top1_correct)),
                "training_acceptance_rate_given_own_class": float(np.mean(acceptance_scores <= current_threshold)),
                "training_accepted_correct_rate": float(np.mean(accepted_correct)),
                "training_distance_ratio_k1_mean": float(np.mean(ratios_k1)),
                "training_distance_ratio_k1_median": float(np.median(ratios_k1)),
                "training_distance_ratio_k1_q1": float(np.quantile(ratios_k1, 0.25)),
                "training_distance_ratio_k1_q3": float(np.quantile(ratios_k1, 0.75)),
                "training_distance_ratio_k1_q90": float(np.quantile(ratios_k1, 0.90)),
                "training_distance_ratio_k1_q95": float(np.quantile(ratios_k1, 0.95)),
                "training_distance_ratio_k1_q99": float(np.quantile(ratios_k1, 0.99)),
                "training_distance_ratio_k1_max": float(np.max(ratios_k1)),
                "training_acceptance_score_mean": float(np.mean(acceptance_scores)),
                "training_acceptance_score_max": float(np.max(acceptance_scores)),
            }
        )
    return rows

def _discover_folds(
    experiment_plan_dir: Path,
    *,
    dataset_root: Path,
    scenarios: set[str] | None,
    folds: set[str] | None,
    model_variant: str | None,
    branch_name: str,
    feature_set: str,
    distance_metric: str,
    threshold_config: AdaptiveThresholdConfig,
    orientation_mode: str,
) -> list[FoldSpec]:
    output: list[FoldSpec] = []
    for fold_json in sorted(experiment_plan_dir.glob("*/*/fold.json")):
        payload = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(payload["scenario"])
        fold_id = str(payload["fold_id"])
        if scenarios and scenario not in scenarios:
            continue
        if folds and fold_id not in folds:
            continue
        if str(payload.get("fold_status", "active")) != "active":
            continue
        test_rows = read_csv_rows(fold_json.parent / "test_takes.csv")
        if not test_rows:
            continue
        model_set = _find_model_set(fold_json.parent, model_variant)

        threshold_deviations: np.ndarray | None = None
        calibration_values: dict[str, np.ndarray] | None = None
        calibration = load_fold_acceptance_calibration(
            fold_json.parent,
            branch_name=branch_name,
            source=threshold_config.calibration_source,
            dataset_root=dataset_root,
        )
        if calibration is not None:
            space = load_model_space(model_set, branch_name)
            if calibration.feature_ids != space.feature_ids:
                raise ValueError(
                    f"Acceptance calibration feature identifiers do not match model space for {scenario}/{fold_id}."
                )
            missing = [gesture for gesture in space.gesture_ids if gesture not in calibration.by_gesture]
            if missing:
                raise ValueError(
                    f"Acceptance calibration is missing gestures for {scenario}/{fold_id}: {missing}"
                )
            threshold_deviations = np.vstack(
                [calibration.by_gesture[gesture].standard_deviation for gesture in space.gesture_ids]
            )
            calibration_values = {
                gesture: calibration.by_gesture[gesture].values for gesture in space.gesture_ids
            }

        classifier = TSGRFClassifier(
            model_set,
            branch_name=branch_name,
            feature_set=feature_set,
            distance_metric=distance_metric,
            threshold_config=threshold_config,
            threshold_standard_deviations=threshold_deviations,
            threshold_calibration_values=calibration_values,
            orientation_mode=orientation_mode,
        )
        calibration_rows = _calibration_rows_for_classifier(
            scenario=scenario,
            fold_id=fold_id,
            classifier=classifier,
            calibration_source=threshold_config.calibration_source,
            calibration_values=calibration_values,
        )
        self_ok = all(
            float(row.get("training_raw_top1_accuracy", 0.0)) >= 1.0 - 1.0e-12
            and float(row.get("training_accepted_correct_rate", 0.0)) >= 1.0 - 1.0e-12
            for row in calibration_rows
        )
        if threshold_config.require_training_self_classification and not self_ok:
            failing = [
                str(row["gesture_id"])
                for row in calibration_rows
                if float(row.get("training_raw_top1_accuracy", 0.0)) < 1.0 - 1.0e-12
                or float(row.get("training_accepted_correct_rate", 0.0)) < 1.0 - 1.0e-12
            ]
            raise ValueError(
                f"Training-complete self-classification requirement failed for {scenario}/{fold_id}: {failing}"
            )
        output.append(
            FoldSpec(
                scenario=scenario,
                fold_id=fold_id,
                fold_dir=fold_json.parent,
                model_set_dir=model_set,
                test_take_ids={str(row["take_id"]) for row in test_rows},
                classifier=classifier,
                acceptance_calibration_rows=calibration_rows,
                training_self_classification_ok=self_ok,
            )
        )
    if not output:
        raise ValueError("No active evaluation folds with models and test takes were selected.")
    return output


def _processing_manifest(report_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(report_dir / "test_take_mediapipe_status.csv")
    if not rows:
        raise ValueError(f"Missing test_take_mediapipe_status.csv in {report_dir}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        take_id = str(row["take_id"])
        if take_id in result:
            raise ValueError(f"Duplicate take_id in processing report: {take_id}")
        result[take_id] = row
    return result


def _ground_truth_index(report_dir: Path) -> dict[str, dict[int, dict[str, str]]]:
    rows = read_csv_rows(report_dir / "frame_ground_truth.csv")
    if not rows:
        raise ValueError(f"Missing frame_ground_truth.csv in {report_dir}")
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["take_id"])][int(row["frame_index"])] = row
    return grouped


def _load_run_features(run_dir: Path, branch_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "features" / "feature_arrays.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached feature arrays: {path}")
    with np.load(path, allow_pickle=False) as archive:
        branches = [str(value) for value in archive["branch_names"].tolist()]
        if branch_name not in branches:
            raise ValueError(f"Branch {branch_name!r} is absent from {path}; available={branches}")
        index = branches.index(branch_name)
        frame_indices = np.asarray(archive["frame_index"], dtype=np.int64)
        statuses = np.asarray(archive["status"]).astype(str)
        values = np.asarray(archive["values"][:, index, :], dtype=np.float64)
    return frame_indices, statuses, values


def _prediction_from_missing(status: str) -> tuple[str, str]:
    normalized = str(status).strip().lower()
    if normalized in {"no_hand", "no_right_hand"}:
        return MISSING_HAND, "mediapipe_missing_hand"
    return NO_GESTURE, f"feature_unavailable:{normalized or 'unknown'}"


def _operational_state_match(gt_state: str, pred_state: str) -> bool:
    """Return whether the prediction is operationally correct.

    ``NO_HAND`` is the reference state and ``MISSING_HAND`` is the TSGRF output
    action for that state, so the pair is a correct operational decision even
    though the literal strings intentionally differ.
    """
    return bool(gt_state == pred_state or (gt_state == NO_HAND and pred_state == MISSING_HAND))


def _error_type(gt_state: str, pred_state: str) -> str:
    gt_gesture = state_gesture_id(gt_state)
    pred_gesture = state_gesture_id(pred_state)
    if gt_gesture:
        if pred_gesture == gt_gesture:
            return ""
        if pred_gesture:
            return "wrong_gesture_inside_gt"
        if pred_state == MISSING_HAND:
            return "missing_hand_inside_gesture"
        return "rejected_gt_gesture"
    if gt_state == NO_GESTURE:
        if pred_gesture:
            return "false_gesture_outside_gt"
        if pred_state == MISSING_HAND:
            return "missing_hand_while_hand_present"
        return ""
    if gt_state == NO_HAND:
        if pred_state == MISSING_HAND:
            return ""
        if pred_gesture:
            return "false_gesture_when_no_hand"
        return "no_hand_not_explicitly_detected"
    return ""


def _update_counts(accumulator: FoldAccumulator, gestures: tuple[str, ...], gt_state: str, pred_state: str) -> None:
    accumulator.evaluated_frames += 1
    accumulator.true_pred[(gt_state, pred_state)] += 1
    if gt_state.startswith("GESTURE_"):
        accumulator.gesture_frames += 1
    elif gt_state == NO_GESTURE:
        accumulator.no_gesture_frames += 1
    elif gt_state == NO_HAND:
        accumulator.no_hand_frames += 1
    accumulator.missing_predictions += int(pred_state == MISSING_HAND)
    accumulator.accepted_gesture_predictions += int(pred_state.startswith("GESTURE_"))
    accumulator.correct_no_hand_missing += int(gt_state == NO_HAND and pred_state == MISSING_HAND)
    accumulator.unexpected_missing_hand += int(gt_state != NO_HAND and pred_state == MISSING_HAND)
    for gesture in gestures:
        positive_true = gt_state == f"GESTURE_{gesture}"
        positive_pred = pred_state == f"GESTURE_{gesture}"
        counts = accumulator.gesture_counts[gesture]
        if positive_true and positive_pred:
            counts["tp"] += 1
        elif not positive_true and positive_pred:
            counts["fp"] += 1
        elif positive_true and not positive_pred:
            counts["fn"] += 1
        else:
            counts["tn"] += 1


def _video_summary(rows: list[dict[str, Any]], target_gesture: str, *, max_lag_frames: int = 10) -> dict[str, Any]:
    gt_positive = f"GESTURE_{target_gesture}"
    tp = sum(row["gt_state"] == gt_positive and row["predicted_state"] == gt_positive for row in rows)
    fn = sum(row["gt_state"] == gt_positive and row["predicted_state"] != gt_positive for row in rows)
    fp = sum(row["gt_state"] != gt_positive and row["predicted_state"] == gt_positive for row in rows)
    tn = sum(row["gt_state"] != gt_positive and row["predicted_state"] != gt_positive for row in rows)
    evaluated = len(rows)
    exact = sum(_operational_state_match(str(row["gt_state"]), str(row["predicted_state"])) for row in rows)
    missing = sum(row["predicted_state"] == MISSING_HAND for row in rows)
    correct_no_hand_missing = sum(row["gt_state"] == NO_HAND and row["predicted_state"] == MISSING_HAND for row in rows)
    no_hand_frames = sum(row["gt_state"] == NO_HAND for row in rows)
    unexpected_missing = sum(row["gt_state"] != NO_HAND and row["predicted_state"] == MISSING_HAND for row in rows)
    hand_present_reference_frames = evaluated - no_hand_frames
    wrong_gesture = sum(
        row["gt_state"] == gt_positive
        and row["predicted_state"].startswith("GESTURE_")
        and row["predicted_state"] != gt_positive
        for row in rows
    )
    gt_gesture_rows = [row for row in rows if row["gt_state"] == gt_positive]
    gt_gesture_with_input = [row for row in gt_gesture_rows if int(row.get("prediction_input_available", 0)) == 1]
    accepted_gt_rows = [row for row in gt_gesture_rows if int(row.get("top1_accepted", 0)) == 1]
    accepted_correct = sum(int(row.get("accepted_class_correct", 0)) for row in accepted_gt_rows)
    raw_top1 = sum(int(row.get("raw_top1_correct", row.get("top1_class_correct", 0))) for row in gt_gesture_with_input)
    raw_top2 = sum(int(row.get("raw_top2_correct", 0)) for row in gt_gesture_with_input)
    raw_top3 = sum(int(row.get("raw_top3_correct", 0)) for row in gt_gesture_with_input)
    confidence = [float(row["ranking_confidence"]) for row in rows if row["ranking_confidence"] != ""]
    quality_gt = [float(row["quality_score_gt"]) for row in rows if row["quality_score_gt"] != ""]
    normalized_gt = [float(row["normalized_distance_gt"]) for row in rows if row["normalized_distance_gt"] != ""]

    target_ref = np.asarray([str(row["gt_state"]) == gt_positive for row in rows], dtype=bool)
    target_pred = np.asarray([str(row["predicted_state"]) == gt_positive for row in rows], dtype=bool)
    any_ref = np.asarray([str(row["gt_state"]).startswith("GESTURE_") for row in rows], dtype=bool)
    any_pred = np.asarray([str(row["predicted_state"]).startswith("GESTURE_") for row in rows], dtype=bool)
    target_temporal = sequence_metrics(target_ref, target_pred, max_abs_lag_frames=max_lag_frames)
    any_temporal = sequence_metrics(any_ref, any_pred, max_abs_lag_frames=max_lag_frames)

    result = {
        "frame_count": evaluated,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "exact_state_accuracy": _safe_divide(exact, evaluated),
        "target_recall": _safe_divide(tp, tp + fn),
        "target_precision": _safe_divide(tp, tp + fp),
        "missing_prediction_rate": _safe_divide(missing, evaluated),
        "wrong_class_contamination_rate": _safe_divide(wrong_gesture, tp + fn),
        "raw_gesture_frames_with_input": len(gt_gesture_with_input),
        "raw_top1_accuracy_given_input": _safe_divide(raw_top1, len(gt_gesture_with_input)),
        "raw_top2_accuracy_given_input": _safe_divide(raw_top2, len(gt_gesture_with_input)),
        "raw_top3_accuracy_given_input": _safe_divide(raw_top3, len(gt_gesture_with_input)),
        "gesture_coverage": _safe_divide(len(accepted_gt_rows), len(gt_gesture_rows)),
        "gesture_coverage_given_input": _safe_divide(len(accepted_gt_rows), len(gt_gesture_with_input)),
        "conditional_accuracy_when_accepted": _optional_divide(accepted_correct, len(accepted_gt_rows)),
        "no_hand_frame_count": no_hand_frames,
        "correct_no_hand_missing_count": correct_no_hand_missing,
        "no_hand_recall": _optional_divide(correct_no_hand_missing, no_hand_frames),
        "missing_hand_precision": _optional_divide(correct_no_hand_missing, missing),
        "unexpected_missing_hand_count": unexpected_missing,
        "unexpected_missing_hand_rate": _safe_divide(unexpected_missing, hand_present_reference_frames),
        "target_gesture_sequence_mcc": target_temporal.mcc,
        "target_gesture_temporal_iou": target_temporal.temporal_iou,
        "target_best_lag_mcc": target_temporal.best_lag_mcc,
        "target_best_lag_frames": target_temporal.best_lag_frames,
        "target_onset_error_frames": target_temporal.onset_error_frames,
        "target_offset_error_frames": target_temporal.offset_error_frames,
        "target_predicted_segment_count": target_temporal.predicted_segment_count,
        "target_longest_predicted_segment_frames": target_temporal.longest_predicted_segment_frames,
        "any_gesture_sequence_mcc": any_temporal.mcc,
        "any_gesture_temporal_iou": any_temporal.temporal_iou,
        "any_best_lag_mcc": any_temporal.best_lag_mcc,
        "any_best_lag_frames": any_temporal.best_lag_frames,
        "any_onset_error_frames": any_temporal.onset_error_frames,
        "any_offset_error_frames": any_temporal.offset_error_frames,
        "any_predicted_segment_count": any_temporal.predicted_segment_count,
        "any_longest_predicted_segment_frames": any_temporal.longest_predicted_segment_frames,
    }
    result.update(_stats(confidence, "ranking_confidence"))
    result.update(_stats(quality_gt, "quality_score_gt"))
    result.update(_stats(normalized_gt, "normalized_distance_gt"))
    return result

def _fold_metrics(spec: FoldSpec, accumulator: FoldAccumulator) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gesture_rows: list[dict[str, Any]] = []
    for gesture in spec.classifier.gesture_ids:
        counts = accumulator.gesture_counts[gesture]
        metrics = _binary_metrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"])
        gesture_rows.append(
            {
                "method_id": METHOD_ID,
                "method_sort_order": METHOD_SORT_ORDER,
                "scenario": spec.scenario,
                "fold_id": spec.fold_id,
                "gesture_id": gesture,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "tn": counts["tn"],
                "fn": counts["fn"],
                "raw_frames_with_prediction_input": counts["raw_input"],
                "raw_top1_accuracy_given_input": _safe_divide(counts["raw_top1"], counts["raw_input"]),
                "raw_top2_accuracy_given_input": _safe_divide(counts["raw_top2"], counts["raw_input"]),
                "raw_top3_accuracy_given_input": _safe_divide(counts["raw_top3"], counts["raw_input"]),
                "gesture_coverage": _safe_divide(counts["accepted_any_on_gt"], counts["tp"] + counts["fn"]),
                "gesture_coverage_given_input": _safe_divide(counts["accepted_any_on_gt"], counts["raw_input"]),
                "conditional_accuracy_when_accepted": _optional_divide(
                    counts["accepted_correct_on_gt"], counts["accepted_any_on_gt"]
                ),
                **metrics,
            }
        )
    macro_fields = ("precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc")
    exact_correct = sum(
        count
        for (true, pred), count in accumulator.true_pred.items()
        if _operational_state_match(true, pred)
    )
    gesture_correct = sum(
        count for (true, pred), count in accumulator.true_pred.items() if true.startswith("GESTURE_") and true == pred
    )
    hand_present_reference_frames = accumulator.evaluated_frames - accumulator.no_hand_frames
    summary = {
        "method_id": METHOD_ID,
        "method_sort_order": METHOD_SORT_ORDER,
        "scenario": spec.scenario,
        "fold_id": spec.fold_id,
        "model_set_dir": spec.model_set_dir.relative_to(spec.fold_dir).as_posix(),
        "selected_feature_count": spec.classifier.selected_feature_count,
        "orientation_mode": spec.classifier.orientation_mode,
        "acceptance_boundary_mode": spec.classifier.threshold_config.boundary_mode,
        "training_self_classification_ok": int(spec.training_self_classification_ok),
        "distance_metric": spec.classifier.distance_metric,
        "acceptance_threshold_mode": spec.classifier.threshold_config.mode,
        "acceptance_deviation_measure": spec.classifier.threshold_config.deviation_measure,
        "acceptance_deviation_multiplier": spec.classifier.threshold_config.deviation_multiplier,
        "acceptance_calibration_source": spec.classifier.threshold_config.calibration_source,
        "evaluated_frames": accumulator.evaluated_frames,
        "gesture_frames": accumulator.gesture_frames,
        "no_gesture_frames": accumulator.no_gesture_frames,
        "no_hand_frames": accumulator.no_hand_frames,
        "exact_state_accuracy": _safe_divide(exact_correct, accumulator.evaluated_frames),
        "gesture_top1_accepted_accuracy": _safe_divide(gesture_correct, accumulator.gesture_frames),
        "gesture_frames_with_prediction_input": accumulator.gesture_frames_with_prediction_input,
        "raw_top1_accuracy_given_input": _safe_divide(
            accumulator.raw_top1_correct, accumulator.gesture_frames_with_prediction_input
        ),
        "raw_top2_accuracy_given_input": _safe_divide(
            accumulator.raw_top2_correct, accumulator.gesture_frames_with_prediction_input
        ),
        "raw_top3_accuracy_given_input": _safe_divide(
            accumulator.raw_top3_correct, accumulator.gesture_frames_with_prediction_input
        ),
        "gesture_coverage": _safe_divide(accumulator.accepted_on_gt_gesture_frames, accumulator.gesture_frames),
        "gesture_coverage_given_input": _safe_divide(
            accumulator.accepted_on_gt_gesture_frames, accumulator.gesture_frames_with_prediction_input
        ),
        "conditional_accuracy_when_accepted": _optional_divide(
            accumulator.accepted_correct_on_gt_gesture_frames, accumulator.accepted_on_gt_gesture_frames
        ),
        "missing_prediction_rate": _safe_divide(accumulator.missing_predictions, accumulator.evaluated_frames),
        "correct_no_hand_missing_count": accumulator.correct_no_hand_missing,
        "no_hand_recall": _optional_divide(accumulator.correct_no_hand_missing, accumulator.no_hand_frames),
        "missing_hand_precision": _optional_divide(accumulator.correct_no_hand_missing, accumulator.missing_predictions),
        "unexpected_missing_hand_count": accumulator.unexpected_missing_hand,
        "unexpected_missing_hand_rate": _safe_divide(
            accumulator.unexpected_missing_hand, hand_present_reference_frames
        ),
        "accepted_gesture_prediction_rate": _safe_divide(accumulator.accepted_gesture_predictions, accumulator.evaluated_frames),
        **{f"macro_{field}": float(np.mean([row[field] for row in gesture_rows])) for field in macro_fields},
    }
    summary.update(_stats(accumulator.ranking_confidences, "ranking_confidence"))
    summary.update(_stats(accumulator.top1_quality_scores, "quality_score_top1"))
    summary.update(_stats(accumulator.gt_quality_scores, "quality_score_gt"))
    summary.update(_stats(accumulator.top1_normalized_distances, "normalized_distance_top1"))
    summary.update(_stats(accumulator.gt_normalized_distances, "normalized_distance_gt"))
    summary.update(_stats(accumulator.classification_batches, "classification_ms_per_frame"))
    confusion_rows = [
        {
            "method_id": METHOD_ID,
            "scenario": spec.scenario,
            "fold_id": spec.fold_id,
            "true_state": true,
            "predicted_state": pred,
            "operationally_correct": int(_operational_state_match(true, pred)),
            "count": count,
        }
        for (true, pred), count in sorted(accumulator.true_pred.items())
    ]
    return summary, gesture_rows, confusion_rows


def _scenario_summary(video_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in fold_rows})
    for scenario in scenarios:
        videos = [row for row in video_rows if row["scenario"] == scenario]
        folds = [row for row in fold_rows if row["scenario"] == scenario]
        result: dict[str, Any] = {
            "method_id": METHOD_ID,
            "method_sort_order": METHOD_SORT_ORDER,
            "scenario": scenario,
            "fold_count": len(folds),
            "video_count": len(videos),
            "evaluated_frames": sum(int(row["frame_count"]) for row in videos),
        }
        for metric in (
            "exact_state_accuracy",
            "target_recall",
            "target_precision",
            "missing_prediction_rate",
            "wrong_class_contamination_rate",
            "raw_top1_accuracy_given_input",
            "raw_top2_accuracy_given_input",
            "raw_top3_accuracy_given_input",
            "gesture_coverage",
            "gesture_coverage_given_input",
            "conditional_accuracy_when_accepted",
            "no_hand_recall",
            "missing_hand_precision",
            "unexpected_missing_hand_rate",
            "ranking_confidence_mean",
            "quality_score_gt_mean",
            "normalized_distance_gt_mean",
        ):
            values = [float(row[metric]) for row in videos if row.get(metric) not in (None, "") and np.isfinite(float(row[metric]))]
            result.update(_stats(values, metric))
        for metric in ("macro_f1", "macro_mcc", "macro_balanced_accuracy"):
            values = [float(row[metric]) for row in folds]
            result.update(_stats(values, metric))
        output.append(result)
    return output


def _group_summary(video_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (("scenario", "background"), "scenario_background"),
        (("scenario", "public_subject_id"), "scenario_subject"),
        (("scenario", "gesture_id"), "scenario_gesture"),
        (("scenario", "public_subject_id", "background"), "scenario_subject_background"),
    ]
    output: list[dict[str, Any]] = []
    for fields, level in specs:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in video_rows:
            grouped[tuple(str(row.get(field, "")) for field in fields)].append(row)
        for key, rows in sorted(grouped.items()):
            result: dict[str, Any] = {
                "method_id": METHOD_ID,
                "method_sort_order": METHOD_SORT_ORDER,
                "aggregation_level": level,
                **{field: value for field, value in zip(fields, key)},
                "video_count": len(rows),
                "frame_count": sum(int(row["frame_count"]) for row in rows),
            }
            for metric in (
                "exact_state_accuracy",
                "target_recall",
                "target_precision",
                "missing_prediction_rate",
                "wrong_class_contamination_rate",
                "raw_top1_accuracy_given_input",
                "raw_top2_accuracy_given_input",
                "raw_top3_accuracy_given_input",
                "gesture_coverage",
                "gesture_coverage_given_input",
                "conditional_accuracy_when_accepted",
                "no_hand_recall",
                "missing_hand_precision",
                "unexpected_missing_hand_rate",
                "target_gesture_sequence_mcc",
                "target_gesture_temporal_iou",
                "target_best_lag_mcc",
                "target_best_lag_frames",
                "target_onset_error_frames",
                "target_offset_error_frames",
                "target_predicted_segment_count",
                "target_longest_predicted_segment_frames",
                "any_gesture_sequence_mcc",
                "any_gesture_temporal_iou",
                "any_best_lag_mcc",
                "any_best_lag_frames",
                "any_onset_error_frames",
                "any_offset_error_frames",
                "any_predicted_segment_count",
                "any_longest_predicted_segment_frames",
                "ranking_confidence_mean",
                "quality_score_gt_mean",
                "normalized_distance_gt_mean",
            ):
                values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "") and np.isfinite(float(row[metric]))]
                result.update(_stats(values, metric))
            output.append(result)
    return output


def _plot_scenario_confusions(confusion_rows: list[dict[str, Any]], output_dir: Path) -> None:
    states = sorted({str(row["true_state"]) for row in confusion_rows} | {str(row["predicted_state"]) for row in confusion_rows})
    for scenario in sorted({str(row["scenario"]) for row in confusion_rows}):
        matrix = np.zeros((len(states), len(states)), dtype=np.int64)
        index = {state: i for i, state in enumerate(states)}
        for row in confusion_rows:
            if row["scenario"] != scenario:
                continue
            matrix[index[str(row["true_state"])], index[str(row["predicted_state"])]] += int(row["count"])
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0)
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(normalized, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(states)), states, rotation=90)
        ax.set_yticks(range(len(states)), states)
        ax.set_xlabel("Predicted state")
        ax.set_ylabel("Ground truth")
        ax.set_title(f"{scenario}: normalized operational confusion matrix")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix__{scenario}.png", dpi=300)
        plt.close(fig)


def _plot_gesture_f1(gesture_rows: list[dict[str, Any]], output: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in gesture_rows})
    gestures = sorted({str(row["gesture_id"]) for row in gesture_rows})
    x = np.arange(len(gestures), dtype=float)
    width = 0.8 / max(1, len(scenarios))
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, scenario in enumerate(scenarios):
        values = []
        for gesture in gestures:
            rows = [row for row in gesture_rows if row["scenario"] == scenario and row["gesture_id"] == gesture]
            values.append(float(np.mean([float(row["f1"]) for row in rows])) if rows else 0.0)
        ax.bar(x + index * width, values, width=width, label=scenario)
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2.0, gestures)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Gesture-level F1 across experiment scenarios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_s4_heatmap(fold_rows: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in fold_rows if row["scenario"] == "S4_LOSO_BACKGROUND"]
    if not rows:
        return
    subjects = sorted({str(row["fold_id"]).split("__")[0].removeprefix("subject_") for row in rows})
    backgrounds = sorted({str(row["fold_id"]).split("__")[-1].removeprefix("background_") for row in rows})
    matrix = np.full((len(subjects), len(backgrounds)), np.nan, dtype=float)
    for row in rows:
        parts = str(row["fold_id"]).split("__")
        subject = parts[0].removeprefix("subject_")
        background = parts[-1].removeprefix("background_")
        matrix[subjects.index(subject), backgrounds.index(background)] = float(row["macro_f1"])
    fig, ax = plt.subplots(figsize=(7, max(4, 0.7 * len(subjects) + 2)))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(backgrounds)), backgrounds)
    ax.set_yticks(range(len(subjects)), subjects)
    ax.set_title("S4 LOSO+background: Macro F1")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def evaluate_tsgrf_experiment(
    experiment_plan_dir: str | Path,
    *,
    test_processing_report: str | Path,
    ground_truth_report: str | Path,
    output_dir: str | Path,
    branch_name: str,
    feature_set: str = "compact",
    distance_metric: str = "weighted_euclidean",
    threshold_config: AdaptiveThresholdConfig | None = None,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    model_variant: str | None = None,
    frame_log_mode: str = "full",
    orientation_mode: str = "camera_aware",
    sequence_max_lag_frames: int = 10,
    progress: bool = True,
) -> Path:
    plan = Path(experiment_plan_dir)
    dataset_root = resolve_dataset_root_from_plan(plan)
    processing_report = Path(test_processing_report)
    gt_report = Path(ground_truth_report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    threshold_config = threshold_config or AdaptiveThresholdConfig()
    threshold_config.validate()
    frame_log_mode = str(frame_log_mode).strip().lower()
    if frame_log_mode not in {"full", "errors", "none"}:
        raise ValueError("frame_log_mode must be 'full', 'errors', or 'none'.")
    specs = _discover_folds(
        plan,
        dataset_root=dataset_root,
        scenarios=scenarios,
        folds=folds,
        model_variant=model_variant,
        branch_name=branch_name,
        feature_set=feature_set,
        distance_metric=distance_metric,
        threshold_config=threshold_config,
        orientation_mode=orientation_mode,
    )
    manifest = _processing_manifest(processing_report)
    ground_truth = _ground_truth_index(gt_report)
    take_to_specs: dict[str, list[FoldSpec]] = defaultdict(list)
    for spec in specs:
        for take_id in spec.test_take_ids:
            take_to_specs[take_id].append(spec)
    accumulators = {(spec.scenario, spec.fold_id): FoldAccumulator() for spec in specs}
    video_rows: list[dict[str, Any]] = []
    frame_fields = [
        "method_id", "method_sort_order", "scenario", "fold_id", "model_variant", "take_id", "public_subject_id",
        "background", "gesture_id", "frame_index", "frame_relative_path",
        "gt_state", "gt_gesture", "cached_status", "prediction_input_available", "predicted_state",
        "prediction_reason", "top1_gesture", "top2_gesture", "top3_gesture", "top1_accepted",
        "distance_gt", "distance_top1", "distance_top2", "distance_top3",
        "acceptance_score_gt", "acceptance_score_top1", "acceptance_threshold_gt", "acceptance_threshold_top1",
        "threshold_gt", "threshold_top1", "normalized_distance_gt", "normalized_distance_top1", "quality_score_gt", "quality_score_top1",
        "gap_1_2", "gap_1_3", "relative_gap_1_2", "relative_gap_1_3", "ranking_confidence",
        "raw_top1_correct", "raw_top2_correct", "raw_top3_correct",
        "top1_class_correct", "accepted_class_correct", "operational_state_correct", "error_type",
    ]
    error_fields = frame_fields
    frame_file = None
    error_file = None
    frame_writer = None
    error_writer = None
    if frame_log_mode == "full":
        frame_file = (out / "frame_predictions.csv").open("w", encoding="utf-8", newline="")
        frame_writer = csv.DictWriter(frame_file, fieldnames=frame_fields, extrasaction="ignore")
        frame_writer.writeheader()
    if frame_log_mode in {"full", "errors"}:
        error_file = (out / "error_frames.csv").open("w", encoding="utf-8", newline="")
        error_writer = csv.DictWriter(error_file, fieldnames=error_fields, extrasaction="ignore")
        error_writer.writeheader()
    total_started = time.perf_counter()
    try:
        take_ids = sorted(take_to_specs)
        for take_number, take_id in enumerate(take_ids, start=1):
            report_row = manifest.get(take_id)
            if report_row is None:
                raise ValueError(f"Test processing report does not contain required take {take_id}.")
            run_path = Path(report_row.get("run_path", ""))
            if not run_path.is_absolute():
                run_path = dataset_root / run_path
            if not run_path.is_dir():
                raise FileNotFoundError(f"Cached run for take {take_id} does not exist: {run_path}")
            frame_indices, statuses, values = _load_run_features(run_path, branch_name)
            gt_by_frame = ground_truth.get(take_id, {})
            frame_position = {int(index): pos for pos, index in enumerate(frame_indices.tolist())}
            for spec in take_to_specs[take_id]:
                accumulator = accumulators[(spec.scenario, spec.fold_id)]
                gt_rows = [gt_by_frame[index] for index in sorted(gt_by_frame) if str(gt_by_frame[index].get("evaluation_state")) != UNANNOTATED]
                if not gt_rows:
                    continue
                positions = np.asarray([frame_position[int(row["frame_index"])] for row in gt_rows], dtype=np.int64)
                local_values = values[positions]
                local_statuses = statuses[positions]
                complete = np.isfinite(local_values).all(axis=1)
                classified: dict[int, tuple[Any, int]] = {}
                if complete.any():
                    started = time.perf_counter()
                    batch = spec.classifier.classify(local_values[complete])
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    complete_positions = np.flatnonzero(complete)
                    accumulator.classification_batches.append(elapsed_ms / max(1, complete_positions.size))
                    for batch_pos, local_pos in enumerate(complete_positions.tolist()):
                        classified[local_pos] = (batch, batch_pos)
                local_frame_rows: list[dict[str, Any]] = []
                for local_pos, gt_row in enumerate(gt_rows):
                    gt_state = str(gt_row["evaluation_state"])
                    gt_gesture = state_gesture_id(gt_state)
                    base = {
                        "method_id": METHOD_ID,
                        "method_sort_order": METHOD_SORT_ORDER,
                        "scenario": spec.scenario,
                        "fold_id": spec.fold_id,
                        "model_variant": model_variant or "reference",
                        "take_id": take_id,
                        "public_subject_id": gt_row.get("public_subject_id", ""),
                        "background": gt_row.get("background", ""),
                        "gesture_id": gt_row.get("gesture_id", ""),
                        "frame_index": int(gt_row["frame_index"]),
                        "frame_relative_path": gt_row.get("frame_relative_path", ""),
                        "gt_state": gt_state,
                        "gt_gesture": gt_gesture,
                        "cached_status": str(local_statuses[local_pos]),
                    }
                    if local_pos not in classified:
                        predicted_state, reason = _prediction_from_missing(str(local_statuses[local_pos]))
                        row = {
                            **base,
                            "prediction_input_available": 0,
                            "predicted_state": predicted_state,
                            "prediction_reason": reason,
                            "top1_gesture": "", "top2_gesture": "", "top3_gesture": "", "top1_accepted": 0,
                            "distance_gt": "", "distance_top1": "", "distance_top2": "", "distance_top3": "",
                            "acceptance_score_gt": "", "acceptance_score_top1": "",
                            "acceptance_threshold_gt": "", "acceptance_threshold_top1": "",
                            "threshold_gt": "", "threshold_top1": "", "normalized_distance_gt": "",
                            "normalized_distance_top1": "", "quality_score_gt": "", "quality_score_top1": "",
                            "gap_1_2": "", "gap_1_3": "", "relative_gap_1_2": "", "relative_gap_1_3": "",
                            "ranking_confidence": "", "raw_top1_correct": 0, "raw_top2_correct": 0, "raw_top3_correct": 0,
                            "top1_class_correct": 0, "accepted_class_correct": 0, "operational_state_correct": int(_operational_state_match(gt_state, predicted_state)),
                        }
                    else:
                        batch, batch_pos = classified[local_pos]
                        i1 = int(batch.top1_index[batch_pos])
                        i2 = int(batch.top2_index[batch_pos])
                        i3 = int(batch.top3_index[batch_pos])
                        top1 = spec.classifier.gesture_ids[i1]
                        top2 = spec.classifier.gesture_ids[i2]
                        top3 = spec.classifier.gesture_ids[i3]
                        accepted = bool(batch.accepted[batch_pos])
                        predicted_state = f"GESTURE_{top1}" if accepted else NO_GESTURE
                        raw_top1_correct = int(bool(gt_gesture and top1 == gt_gesture))
                        raw_top2_correct = int(bool(gt_gesture and gt_gesture in {top1, top2}))
                        raw_top3_correct = int(bool(gt_gesture and gt_gesture in {top1, top2, top3}))
                        if gt_gesture:
                            accumulator.gesture_frames_with_prediction_input += 1
                            accumulator.raw_top1_correct += raw_top1_correct
                            accumulator.raw_top2_correct += raw_top2_correct
                            accumulator.raw_top3_correct += raw_top3_correct
                            accumulator.accepted_on_gt_gesture_frames += int(accepted)
                            accumulator.accepted_correct_on_gt_gesture_frames += int(accepted and top1 == gt_gesture)
                            gesture_counts = accumulator.gesture_counts[gt_gesture]
                            gesture_counts["raw_input"] += 1
                            gesture_counts["raw_top1"] += raw_top1_correct
                            gesture_counts["raw_top2"] += raw_top2_correct
                            gesture_counts["raw_top3"] += raw_top3_correct
                            gesture_counts["accepted_any_on_gt"] += int(accepted)
                            gesture_counts["accepted_correct_on_gt"] += int(accepted and top1 == gt_gesture)
                        threshold_top1 = float(spec.classifier.thresholds[i1])
                        acceptance_score_top1 = float(batch.acceptance_score_top1[batch_pos])
                        distance_gt: float | str = ""
                        acceptance_score_gt: float | str = ""
                        threshold_gt: float | str = ""
                        normalized_gt: float | str = ""
                        quality_gt: float | str = ""
                        if gt_gesture and gt_gesture in spec.classifier.gesture_ids:
                            gt_index = spec.classifier.gesture_ids.index(gt_gesture)
                            distance_gt = float(batch.distances[batch_pos, gt_index])
                            acceptance_score_gt = float(batch.acceptance_scores[batch_pos, gt_index])
                            threshold_gt = float(spec.classifier.thresholds[gt_index])
                            normalized_gt = acceptance_score_gt / max(threshold_gt, 1.0e-12)
                            quality_gt = float(np.exp2(-normalized_gt))
                            accumulator.gt_quality_scores.append(float(quality_gt))
                            accumulator.gt_normalized_distances.append(float(normalized_gt))
                        accumulator.ranking_confidences.append(float(batch.ranking_confidence[batch_pos]))
                        accumulator.top1_quality_scores.append(float(batch.quality_score[batch_pos]))
                        accumulator.top1_normalized_distances.append(float(batch.normalized_distance[batch_pos]))
                        row = {
                            **base,
                            "prediction_input_available": 1,
                            "predicted_state": predicted_state,
                            "prediction_reason": "accepted_top1" if accepted else "top1_above_adaptive_threshold",
                            "top1_gesture": top1,
                            "top2_gesture": top2,
                            "top3_gesture": top3,
                            "top1_accepted": int(accepted),
                            "distance_gt": distance_gt,
                            "distance_top1": float(batch.top1_distance[batch_pos]),
                            "distance_top2": float(batch.top2_distance[batch_pos]),
                            "distance_top3": float(batch.top3_distance[batch_pos]),
                            "acceptance_score_gt": acceptance_score_gt,
                            "acceptance_score_top1": acceptance_score_top1,
                            "acceptance_threshold_gt": threshold_gt,
                            "acceptance_threshold_top1": threshold_top1,
                            "threshold_gt": threshold_gt,
                            "threshold_top1": threshold_top1,
                            "normalized_distance_gt": normalized_gt,
                            "normalized_distance_top1": float(batch.normalized_distance[batch_pos]),
                            "quality_score_gt": quality_gt,
                            "quality_score_top1": float(batch.quality_score[batch_pos]),
                            "gap_1_2": float(batch.gap_1_2[batch_pos]),
                            "gap_1_3": float(batch.gap_1_3[batch_pos]),
                            "relative_gap_1_2": float(batch.relative_gap_1_2[batch_pos]),
                            "relative_gap_1_3": float(batch.relative_gap_1_3[batch_pos]),
                            "ranking_confidence": float(batch.ranking_confidence[batch_pos]),
                            "raw_top1_correct": raw_top1_correct,
                            "raw_top2_correct": raw_top2_correct,
                            "raw_top3_correct": raw_top3_correct,
                            "top1_class_correct": raw_top1_correct,
                            "accepted_class_correct": int(bool(gt_gesture and predicted_state == gt_state)),
                            "operational_state_correct": int(_operational_state_match(gt_state, predicted_state)),
                        }
                    row["error_type"] = _error_type(gt_state, str(row["predicted_state"]))
                    _update_counts(accumulator, spec.classifier.gesture_ids, gt_state, str(row["predicted_state"]))
                    if frame_writer is not None:
                        frame_writer.writerow(row)
                    if error_writer is not None and row["error_type"]:
                        error_writer.writerow(row)
                    local_frame_rows.append(row)
                target_gesture = str(gt_rows[0].get("gesture_id", "")).upper()
                summary = _video_summary(local_frame_rows, target_gesture, max_lag_frames=sequence_max_lag_frames)
                video_rows.append(
                    {
                        "method_id": METHOD_ID,
                        "method_sort_order": METHOD_SORT_ORDER,
                        "scenario": spec.scenario,
                        "fold_id": spec.fold_id,
                        "model_variant": model_variant or "reference",
                        "take_id": take_id,
                        "public_subject_id": gt_rows[0].get("public_subject_id", ""),
                        "background": gt_rows[0].get("background", ""),
                        "gesture_id": target_gesture,
                        **summary,
                    }
                )
            if progress and (take_number % 50 == 0 or take_number == len(take_ids)):
                print(f"[evaluate {take_number}/{len(take_ids)}] cached takes", flush=True)
    finally:
        if frame_file is not None:
            frame_file.close()
        if error_file is not None:
            error_file.close()

    fold_rows: list[dict[str, Any]] = []
    gesture_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for spec in specs:
        summary, gestures, confusion = _fold_metrics(spec, accumulators[(spec.scenario, spec.fold_id)])
        fold_rows.append(summary)
        gesture_rows.extend(gestures)
        confusion_rows.extend(confusion)
    scenario_rows = _scenario_summary(video_rows, fold_rows)
    group_rows = _group_summary(video_rows)
    acceptance_calibration_rows = [
        row for spec in specs for row in spec.acceptance_calibration_rows
    ]

    for filename, rows in (
        ("video_summary.csv", video_rows),
        ("gesture_metrics.csv", gesture_rows),
        ("fold_summary.csv", fold_rows),
        ("scenario_summary.csv", scenario_rows),
        ("group_summary.csv", group_rows),
        ("confusion_matrix_long.csv", confusion_rows),
        ("acceptance_calibration_summary.csv", acceptance_calibration_rows),
    ):
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        write_csv_rows(out / filename, rows, fields)

    workbook_sheets: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    for name, rows in (
        ("scenario_summary", scenario_rows),
        ("fold_summary", fold_rows),
        ("gesture_metrics", gesture_rows),
        ("group_summary", group_rows),
        ("video_summary", video_rows),
        ("confusion_matrix", confusion_rows),
        ("acceptance_calibration", acceptance_calibration_rows),
    ):
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        workbook_sheets[name] = (fields, rows)
    write_workbook(out / "tsgrf_experiment_results.xlsx", workbook_sheets)

    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_scenario_confusions(confusion_rows, plots_dir)
    _plot_gesture_f1(gesture_rows, plots_dir / "gesture_f1_by_scenario.png")
    _plot_s4_heatmap(fold_rows, plots_dir / "s4_macro_f1_heatmap.png")

    write_json(
        out / "evaluation_report.json",
        {
            "schema_version": "tsgrf_experiment_evaluation_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method_id": METHOD_ID,
            "experiment_plan": plan.name,
            "test_processing_report": processing_report.name,
            "ground_truth_report": gt_report.name,
            "dataset_root": ".",
            "dataset_contract": "tsgr_public_dataset_v1",
            "model_variant": model_variant or "reference",
            "branch_name": branch_name,
            "feature_set": feature_set,
            "distance_metric": distance_metric,
            "acceptance_threshold": {
                "mode": threshold_config.mode,
                "deviation_measure": threshold_config.deviation_measure,
                "deviation_multiplier": threshold_config.deviation_multiplier,
                "global_fixed_threshold": threshold_config.global_fixed_threshold,
                "calibration_source": threshold_config.calibration_source,
                "boundary_mode": threshold_config.boundary_mode,
                "require_training_self_classification": threshold_config.require_training_self_classification,
            },
            "orientation_mode": orientation_mode,
            "sequence_max_lag_frames": int(sequence_max_lag_frames),
            "fold_count": len(specs),
            "scenario_count": len({spec.scenario for spec in specs}),
            "video_result_count": len(video_rows),
            "elapsed_wall_time_s": time.perf_counter() - total_started,
            "frame_level_xlsx_exported": False,
            "frame_log_mode": frame_log_mode,
            "frame_level_csv": "frame_predictions.csv" if frame_log_mode == "full" else None,
            "error_frame_csv": "error_frames.csv" if frame_log_mode in {"full", "errors"} else None,
            "external_method_input_contract": "canonical image; external baselines do not receive TSGRF landmarks/cache",
        },
    )
    return out
