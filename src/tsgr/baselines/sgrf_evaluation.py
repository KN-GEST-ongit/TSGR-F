"""Evaluation orchestration for the thirteen comparable upstream SGRF methods.

The external algorithms are executed in an isolated Python 3.11 environment.
Each adapter receives the canonical raw image and is responsible for its own
preprocessing.  TSGR-F landmarks, MediaPipe results and hand ROIs are never
injected into an external method.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tsgr.dataset.contract import resolve_dataset_root_from_plan

from tsgr.baselines.registry import SGRFMethodSpec, normalize_method_selection, registry_rows
from tsgr.baselines.sgrf_bridge import audit_sgrf_environment, payload_sha256, run_worker, write_job
from tsgr.baselines.sgrf_models import resolve_external_worker_count
from tsgr.evaluation.ground_truth import NO_GESTURE, NO_HAND, UNANNOTATED, state_gesture_id
from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows, write_workbook
from tsgr.utils.serialization import write_json
from tsgr.visualization.matplotlib_backend import configure_headless_backend

configure_headless_backend()
import matplotlib.pyplot as plt

PREDICTION_ERROR = "PREDICTION_ERROR"
GESTURE_ORDER = ("A", "B", "C", "E", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")


# Execution order is deliberately separate from publication/table sort_order.
# Fast methods run first so useful complete scenario results become available
# early, while the two known CPU-heavy methods are deferred to the end.
SGRF_FAST_FIRST_METHOD_IDS = (
    "MOHMMAD_DADI",
    "MAUNG",
    "CHANG_CHEN",
    "NGUYEN_HUYNH",
    "GUPTA_JAAFAR",
    "JOSHI_KUMAR",
    "EID_SCHWENKER",
    "PINTO_BORGES",
    "ADITHYA_RAJESH",
    "ZHUANG_YANG",
    "NAIDOO_OMLIN",
    "MOHANTY_RAMBHATLA",
    "OYEDOTUN_KHASHMAN",
)
SGRF_SCENARIO_EXECUTION_PRIORITY = {
    "S1_ALL_IN_DOMAIN": 10,
    "S2_LOBO": 20,
    "S3_LOSO": 30,
    "S4_LOSO_BACKGROUND": 40,
}


def _execution_method_order(methods: Iterable[SGRFMethodSpec], policy: str) -> tuple[SGRFMethodSpec, ...]:
    values = tuple(methods)
    if policy == "registry":
        return tuple(sorted(values, key=lambda item: item.sort_order))
    if policy != "fast_first":
        raise ValueError("execution_order must be fast_first or registry.")
    priority = {method_id: index for index, method_id in enumerate(SGRF_FAST_FIRST_METHOD_IDS)}
    return tuple(sorted(values, key=lambda item: (priority.get(item.method_id, 10_000), item.sort_order)))


def _execution_fold_order(folds: Iterable[EvaluationFold]) -> tuple[EvaluationFold, ...]:
    return tuple(
        sorted(
            folds,
            key=lambda fold: (
                SGRF_SCENARIO_EXECUTION_PRIORITY.get(fold.scenario, 10_000),
                fold.fold_id,
            ),
        )
    )

FRAME_RESULT_FIELDS = [
    "method_id", "method_display_name", "method_sort_order", "scenario", "fold_id", "take_id", "frame_index",
    "frame_relative_path", "public_subject_id", "background", "gesture_id",
    "operational_gt_state", "benchmark_gt_state", "raw_predicted_label", "raw_predicted_state", "predicted_label",
    "predicted_state", "rejected_by_policy", "raw_certainty", "normalized_certainty", "correct_exact", "error_type",
    "image_decode_ms", "classification_ms", "worker_error", "no_hand_collapsed_for_benchmark",
]


@dataclass(frozen=True, slots=True)
class EvaluationFold:
    scenario: str
    fold_id: str
    fold_dir: Path
    dataset_root: Path
    test_take_ids: frozenset[str]


@dataclass(slots=True)
class FoldAccumulator:
    true_pred: Counter[tuple[str, str]] = field(default_factory=Counter)
    gesture_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    evaluated_frames: int = 0
    prediction_failures: int = 0
    rejected_predictions: int = 0
    normalized_certainties: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)
    classification_ms: list[float] = field(default_factory=list)


def _safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


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


def _stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_q1": math.nan,
            f"{prefix}_q3": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_q1": float(np.quantile(array, 0.25)),
        f"{prefix}_q3": float(np.quantile(array, 0.75)),
    }


def benchmark_ground_truth_state(operational_state: str) -> str:
    """Map operational GT to the common static-classifier benchmark state.

    External SGRF methods do not expose a missing-hand state.  For the common
    benchmark, detector-derived NO_HAND is therefore mapped to NO_GESTURE while
    the original operational state remains in frame-level audit output.
    """
    state = str(operational_state).strip().upper()
    if state == NO_HAND:
        return NO_GESTURE
    return state


def _discover_folds(
    plan_dir: Path,
    *,
    scenarios: set[str] | None,
    folds: set[str] | None,
    dataset_root_override: str | Path | None = None,
) -> list[EvaluationFold]:
    meta_path = plan_dir / "experiment_plan.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing experiment_plan.json: {meta_path}")
    plan_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_root = resolve_dataset_root_from_plan(plan_dir, override=dataset_root_override)
    result: list[EvaluationFold] = []
    for fold_json in sorted(plan_dir.glob("*/*/fold.json")):
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
        result.append(
            EvaluationFold(
                scenario=scenario,
                fold_id=fold_id,
                fold_dir=fold_json.parent,
                dataset_root=dataset_root,
                test_take_ids=frozenset(str(row["take_id"]) for row in test_rows),
            )
        )
    if not result:
        raise ValueError("No active experiment folds with test takes matched the requested scope.")
    return result


def _ground_truth_rows(report_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(report_dir / "frame_ground_truth.csv")
    if not rows:
        raise ValueError(f"Missing frame_ground_truth.csv in {report_dir}")
    return [row for row in rows if str(row.get("include_in_metrics", "1")).strip() not in {"0", "false", "False"}]


def _model_manifest(models_root: Path, scenario: str, fold_id: str, method_id: str) -> tuple[Path, dict[str, Any]]:
    path = models_root / "models" / scenario / fold_id / method_id / "baseline_model.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained baseline model manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("method_id", "")) != method_id or str(payload.get("scenario", "")) != scenario or str(payload.get("fold_id", "")) != fold_id:
        raise ValueError(f"Baseline model manifest identity mismatch: {path}")
    return path, payload


def _prediction_fingerprint(
    *,
    method: SGRFMethodSpec,
    model_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    rejection_policy: str,
    certainty_threshold_normalized: float | None,
    model_load_policy: str,
    seed: int,
) -> str:
    return payload_sha256(
        {
            "adapter_schema": "tsgr_sgrf_prediction_v1",
            "method": method.to_dict(),
            "model_training_fingerprint": model_manifest.get("training_fingerprint", ""),
            "model_files": model_manifest.get("model_files", []),
            "frames": [
                {
                    "take_id": row["take_id"],
                    "frame_index": row["frame_index"],
                    "frame_relative_path": row["frame_relative_path"],
                    "operational_gt_state": row["operational_gt_state"],
                }
                for row in rows
            ],
            "rejection_policy": rejection_policy,
            "certainty_threshold_normalized": certainty_threshold_normalized,
            "model_load_policy": model_load_policy,
            "seed": seed,
        }
    )


def _prepare_prediction_job(
    output_root: Path,
    fold: EvaluationFold,
    method: SGRFMethodSpec,
    model_manifest: dict[str, Any],
    model_manifest_path: Path,
    gt_rows: list[dict[str, str]],
    *,
    rejection_policy: str,
    certainty_threshold_normalized: float | None,
    model_load_policy: str,
    seed: int,
    fail_fast: bool,
    progress_every: int,
) -> tuple[dict[str, Any], Path, str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for gt in gt_rows:
        take_id = str(gt["take_id"])
        if take_id not in fold.test_take_ids:
            continue
        relative = str(gt.get("frame_relative_path", "")).replace("\\", "/")
        frame_path = fold.dataset_root / Path(relative)
        if not frame_path.is_file():
            raise FileNotFoundError(f"Canonical test frame is missing: {frame_path}")
        rows.append(
            {
                "row_id": f"{take_id}::{int(gt['frame_index']):09d}",
                "take_id": take_id,
                "frame_index": int(gt["frame_index"]),
                "frame_relative_path": relative,
                "frame_path": str(frame_path.resolve()),
                "public_subject_id": gt.get("public_subject_id", ""),
                "background": gt.get("background", ""),
                "gesture_id": str(gt.get("gesture_id", "")).upper(),
                "operational_gt_state": str(gt.get("evaluation_state", "")).upper(),
                "benchmark_gt_state": benchmark_ground_truth_state(str(gt.get("evaluation_state", ""))),
            }
        )
    if not rows:
        raise ValueError(f"No ground-truth frames matched {fold.scenario}/{fold.fold_id}.")
    rows.sort(key=lambda row: (row["take_id"], int(row["frame_index"])))
    job_dir = output_root / "jobs" / fold.scenario / fold.fold_id / method.method_id
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "prediction_manifest.csv"
    write_csv_rows(
        manifest_path,
        rows,
        [
            "row_id", "take_id", "frame_index", "frame_relative_path", "frame_path", "public_subject_id",
            "background", "gesture_id", "operational_gt_state", "benchmark_gt_state",
        ],
    )
    fingerprint = _prediction_fingerprint(
        method=method,
        model_manifest=model_manifest,
        rows=rows,
        rejection_policy=rejection_policy,
        certainty_threshold_normalized=certainty_threshold_normalized,
        model_load_policy=model_load_policy,
        seed=seed,
    )
    portable_model_dir = model_manifest_path.parent / str(model_manifest.get("model_dir_relative", "model"))
    if not portable_model_dir.is_dir():
        stored_model_dir = Path(str(model_manifest.get("model_dir", "")))
        if not stored_model_dir.is_dir():
            raise FileNotFoundError(f"Cannot resolve baseline model directory from {model_manifest_path}.")
        portable_model_dir = stored_model_dir
    job = {
        "schema_version": "tsgr_sgrf_prediction_job_v1",
        "method_id": method.method_id,
        "method_display_name": method.display_name,
        "scenario": fold.scenario,
        "fold_id": fold.fold_id,
        "prediction_manifest": str(manifest_path.resolve()),
        "output_csv": str((job_dir / "worker_predictions.csv").resolve()),
        "model_dir": str(portable_model_dir.resolve()),
        "gestures": list(model_manifest.get("gestures") or GESTURE_ORDER),
        "coordinate_policy": method.coordinate_policy,
        "certainty_scale": method.certainty_scale,
        "custom_options": dict(model_manifest.get("custom_options") or {}),
        "rejection_policy": rejection_policy,
        "certainty_threshold_normalized": certainty_threshold_normalized,
        "model_load_policy": model_load_policy,
        "seed": seed,
        "fail_fast": fail_fast,
        "progress_every": progress_every,
        "prediction_fingerprint": fingerprint,
        "input_contract": "canonical raw image; no TSGRF landmarks, MediaPipe cache, or detected hand ROI",
    }
    write_job(job_dir / "predict_job.json", job)
    return job, job_dir, fingerprint, rows


def _run_prediction_job(
    sgrf_python: str | Path,
    job: dict[str, Any],
    job_dir: Path,
    *,
    stream_output: bool = False,
) -> dict[str, Any]:
    result = run_worker(
        sgrf_python,
        "predict",
        job_json=job_dir / "predict_job.json",
        result_json=job_dir / "predict_result.json",
        log_path=job_dir / "predict.log",
        stream_output=stream_output,
    )
    result["prediction_fingerprint"] = job["prediction_fingerprint"]
    write_json(job_dir / "prediction_run.json", result)
    return result


def _error_type(gt_state: str, pred_state: str) -> str:
    gt_gesture = state_gesture_id(gt_state)
    pred_gesture = state_gesture_id(pred_state)
    if pred_state == PREDICTION_ERROR:
        return "prediction_error"
    if gt_gesture:
        if pred_gesture == gt_gesture:
            return ""
        if pred_gesture:
            return "wrong_gesture_inside_gt"
        return "missed_gesture"
    if gt_state == NO_GESTURE:
        if pred_gesture:
            return "false_gesture_outside_gt"
        return ""
    return ""


def _update_counts(acc: FoldAccumulator, gestures: Iterable[str], gt_state: str, pred_state: str) -> None:
    acc.evaluated_frames += 1
    acc.true_pred[(gt_state, pred_state)] += 1
    acc.prediction_failures += int(pred_state == PREDICTION_ERROR)
    for gesture in gestures:
        positive_true = gt_state == f"GESTURE_{gesture}"
        positive_pred = pred_state == f"GESTURE_{gesture}"
        counts = acc.gesture_counts[gesture]
        if positive_true and positive_pred:
            counts["tp"] += 1
        elif not positive_true and positive_pred:
            counts["fp"] += 1
        elif positive_true and not positive_pred:
            counts["fn"] += 1
        else:
            counts["tn"] += 1


def _video_summary(rows: list[dict[str, Any]], target_gesture: str) -> dict[str, Any]:
    gt_positive = f"GESTURE_{target_gesture}"
    tp = sum(row["benchmark_gt_state"] == gt_positive and row["predicted_state"] == gt_positive for row in rows)
    fn = sum(row["benchmark_gt_state"] == gt_positive and row["predicted_state"] != gt_positive for row in rows)
    fp = sum(row["benchmark_gt_state"] != gt_positive and row["predicted_state"] == gt_positive for row in rows)
    tn = sum(row["benchmark_gt_state"] != gt_positive and row["predicted_state"] != gt_positive for row in rows)
    exact = sum(row["benchmark_gt_state"] == row["predicted_state"] for row in rows)
    failures = sum(row["predicted_state"] == PREDICTION_ERROR for row in rows)
    rejected = sum(int(row.get("rejected_by_policy", 0)) for row in rows)
    wrong = sum(
        row["benchmark_gt_state"] == gt_positive
        and str(row["predicted_state"]).startswith("GESTURE_")
        and row["predicted_state"] != gt_positive
        for row in rows
    )
    certainty = [float(row["normalized_certainty"]) for row in rows if row.get("normalized_certainty") not in (None, "")]
    classification = [float(row["classification_ms"]) for row in rows if row.get("classification_ms") not in (None, "")]
    decode = [float(row["image_decode_ms"]) for row in rows if row.get("image_decode_ms") not in (None, "")]
    result: dict[str, Any] = {
        "frame_count": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "exact_state_accuracy": _safe_divide(exact, len(rows)),
        "target_recall": _safe_divide(tp, tp + fn),
        "target_precision": _safe_divide(tp, tp + fp),
        "prediction_failure_rate": _safe_divide(failures, len(rows)),
        "rejection_rate": _safe_divide(rejected, len(rows)),
        "wrong_class_contamination_rate": _safe_divide(wrong, tp + fn),
    }
    result.update(_stats(certainty, "normalized_certainty"))
    result.update(_stats(classification, "classification_ms"))
    result.update(_stats(decode, "image_decode_ms"))
    return result


def _fold_metrics(
    method: SGRFMethodSpec,
    fold: EvaluationFold,
    acc: FoldAccumulator,
    *,
    model_manifest_path: Path,
    rejection_policy: str,
    certainty_threshold_normalized: float | None,
    model_load_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gesture_rows: list[dict[str, Any]] = []
    for gesture in GESTURE_ORDER:
        counts = acc.gesture_counts[gesture]
        metrics = _binary_metrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"])
        gesture_rows.append(
            {
                "method_id": method.method_id,
                "method_display_name": method.display_name,
                "method_sort_order": method.sort_order,
                "scenario": fold.scenario,
                "fold_id": fold.fold_id,
                "gesture_id": gesture,
                "tp": counts["tp"], "fp": counts["fp"], "tn": counts["tn"], "fn": counts["fn"],
                **metrics,
            }
        )
    macro_fields = ("precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc")
    exact_correct = sum(count for (true, pred), count in acc.true_pred.items() if true == pred)
    summary: dict[str, Any] = {
        "method_id": method.method_id,
        "method_display_name": method.display_name,
        "method_sort_order": method.sort_order,
        "scenario": fold.scenario,
        "fold_id": fold.fold_id,
        "model_manifest": str(model_manifest_path.resolve()),
        "rejection_policy": rejection_policy,
        "certainty_threshold_normalized": "" if certainty_threshold_normalized is None else certainty_threshold_normalized,
        "model_load_policy": model_load_policy,
        "evaluated_frames": acc.evaluated_frames,
        "prediction_failure_count": acc.prediction_failures,
        "prediction_failure_rate": _safe_divide(acc.prediction_failures, acc.evaluated_frames),
        "rejected_prediction_count": acc.rejected_predictions,
        "rejection_rate": _safe_divide(acc.rejected_predictions, acc.evaluated_frames),
        "exact_state_accuracy": _safe_divide(exact_correct, acc.evaluated_frames),
        **{f"macro_{field}": float(np.mean([row[field] for row in gesture_rows])) for field in macro_fields},
    }
    summary.update(_stats(acc.normalized_certainties, "normalized_certainty"))
    summary.update(_stats(acc.classification_ms, "classification_ms_per_frame"))
    summary.update(_stats(acc.decode_ms, "image_decode_ms_per_frame"))
    confusion = [
        {
            "method_id": method.method_id,
            "method_display_name": method.display_name,
            "method_sort_order": method.sort_order,
            "scenario": fold.scenario,
            "fold_id": fold.fold_id,
            "true_state": true,
            "predicted_state": pred,
            "count": count,
        }
        for (true, pred), count in sorted(acc.true_pred.items())
    ]
    return summary, gesture_rows, confusion


def _scenario_summary(video_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["method_id"]), str(row["scenario"])) for row in fold_rows})
    for method_id, scenario in keys:
        folds = [row for row in fold_rows if row["method_id"] == method_id and row["scenario"] == scenario]
        videos = [row for row in video_rows if row["method_id"] == method_id and row["scenario"] == scenario]
        first = folds[0]
        result: dict[str, Any] = {
            "method_id": method_id,
            "method_display_name": first["method_display_name"],
            "method_sort_order": first["method_sort_order"],
            "scenario": scenario,
            "fold_count": len(folds),
            "video_count": len(videos),
            "evaluated_frames": sum(int(row["frame_count"]) for row in videos),
        }
        for metric in (
            "exact_state_accuracy", "target_recall", "target_precision", "prediction_failure_rate",
            "rejection_rate", "wrong_class_contamination_rate", "normalized_certainty_mean", "classification_ms_mean",
        ):
            values = [float(row[metric]) for row in videos if row.get(metric) not in (None, "") and np.isfinite(float(row[metric]))]
            result.update(_stats(values, metric))
        for metric in ("macro_f1", "macro_mcc", "macro_balanced_accuracy"):
            values = [float(row[metric]) for row in folds if row.get(metric) not in (None, "")]
            result.update(_stats(values, metric))
        timing_values = [float(row["classification_ms_per_frame_mean"]) for row in folds if row.get("classification_ms_per_frame_mean") not in (None, "")]
        result.update(_stats(timing_values, "classification_ms_per_frame"))
        output.append(result)
    return sorted(output, key=lambda row: (int(row["method_sort_order"]), str(row["scenario"])))


def _group_summary(video_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (("method_id", "scenario", "background"), "scenario_background"),
        (("method_id", "scenario", "public_subject_id"), "scenario_subject"),
        (("method_id", "scenario", "gesture_id"), "scenario_gesture"),
        (("method_id", "scenario", "public_subject_id", "background"), "scenario_subject_background"),
    ]
    output: list[dict[str, Any]] = []
    for fields, level in specs:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in video_rows:
            grouped[tuple(str(row.get(field, "")) for field in fields)].append(row)
        for key, rows in sorted(grouped.items()):
            first = rows[0]
            result: dict[str, Any] = {
                "method_id": first["method_id"],
                "method_display_name": first["method_display_name"],
                "method_sort_order": first["method_sort_order"],
                "aggregation_level": level,
                **{field: value for field, value in zip(fields, key)},
                "video_count": len(rows),
                "frame_count": sum(int(row["frame_count"]) for row in rows),
            }
            for metric in (
                "exact_state_accuracy", "target_recall", "target_precision", "prediction_failure_rate",
                "rejection_rate", "wrong_class_contamination_rate", "normalized_certainty_mean", "classification_ms_mean",
            ):
                values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "") and np.isfinite(float(row[metric]))]
                result.update(_stats(values, metric))
            output.append(result)
    return sorted(output, key=lambda row: (int(row["method_sort_order"]), str(row.get("aggregation_level", "")), str(row.get("scenario", ""))))


def _collect_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _plot_method_scenario_f1(scenario_rows: list[dict[str, Any]], output: Path) -> None:
    if not scenario_rows:
        return
    methods = sorted({(int(row["method_sort_order"]), str(row["method_id"])) for row in scenario_rows})
    scenarios = sorted({str(row["scenario"]) for row in scenario_rows})
    matrix = np.full((len(methods), len(scenarios)), np.nan, dtype=float)
    index_m = {method_id: idx for idx, (_, method_id) in enumerate(methods)}
    index_s = {scenario: idx for idx, scenario in enumerate(scenarios)}
    for row in scenario_rows:
        matrix[index_m[str(row["method_id"])], index_s[str(row["scenario"])]] = float(row.get("macro_f1_mean", math.nan))
    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 2), max(6, len(methods) * 0.45 + 2)))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=25, ha="right")
    ax.set_yticks(range(len(methods)), [method_id for _, method_id in methods])
    ax.set_title("SGRF baseline macro F1 by scenario")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Macro F1")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_speed_quality(scenario_rows: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in scenario_rows if str(row.get("scenario", "")) == "S1_ALL_IN_DOMAIN"]
    if not rows:
        rows = scenario_rows
    x: list[float] = []
    y: list[float] = []
    labels: list[str] = []
    for row in rows:
        try:
            timing = float(row["classification_ms_mean_mean"])
            quality = float(row["macro_f1_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(timing) and np.isfinite(quality):
            x.append(timing)
            y.append(quality)
            labels.append(str(row["method_id"]))
    if not x:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y)
    for xx, yy, label in zip(x, y, labels):
        ax.annotate(label, (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean classification time per frame [ms]")
    ax.set_ylabel("Macro F1")
    ax.set_title("SGRF baselines: speed-quality relationship")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def evaluate_sgrf_baselines(
    experiment_plan_dir: str | Path,
    *,
    sgrf_python: str | Path,
    baseline_models_dir: str | Path,
    ground_truth_report: str | Path,
    output_dir: str | Path,
    methods: Iterable[str] | None = None,
    all_methods: bool = False,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    dataset_root_override: str | Path | None = None,
    rejection_policy: str = "closed_set",
    certainty_threshold_normalized: float | None = None,
    model_load_policy: str = "process_cache",
    seed: int = 2026,
    workers: int = 1,
    force: bool = False,
    fail_fast: bool = False,
    progress: bool = True,
    progress_every: int = 1000,
    allow_nonreference_python: bool = False,
    allow_nonreference_sgrf: bool = False,
    execution_order: str = "fast_first",
    aggregate_only: bool = False,
) -> Path:
    if rejection_policy not in {"closed_set", "certainty_reject"}:
        raise ValueError("rejection_policy must be closed_set or certainty_reject.")
    if rejection_policy == "certainty_reject" and certainty_threshold_normalized is None:
        raise ValueError("certainty_threshold_normalized is required for certainty_reject.")
    if certainty_threshold_normalized is not None and not 0.0 <= float(certainty_threshold_normalized) <= 1.0:
        raise ValueError("certainty_threshold_normalized must be in [0, 1].")
    if model_load_policy not in {"process_cache", "upstream_each_call"}:
        raise ValueError("model_load_policy must be process_cache or upstream_each_call.")
    plan_dir = Path(experiment_plan_dir)
    models_root = Path(baseline_models_dir)
    gt_dir = Path(ground_truth_report)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_methods = normalize_method_selection(methods, all_methods=all_methods)
    if rejection_policy == "certainty_reject":
        without_certainty = [method.method_id for method in selected_methods if method.certainty_scale == "none"]
        if without_certainty:
            raise ValueError(
                "certainty_reject cannot be used for methods without upstream certainty: " + ", ".join(without_certainty)
            )
    selected_folds = _execution_fold_order(
        _discover_folds(plan_dir, scenarios=scenarios, folds=folds, dataset_root_override=dataset_root_override)
    )
    execution_methods = _execution_method_order(selected_methods, execution_order)
    gt_rows = _ground_truth_rows(gt_dir)
    audit = audit_sgrf_environment(
        sgrf_python,
        output_dir=output_root / "environment",
        allow_nonreference_python=allow_nonreference_python,
        allow_nonreference_sgrf=allow_nonreference_sgrf,
    )
    write_csv_rows(
        output_root / "method_registry.csv",
        registry_rows(),
        ["method_id", "display_name", "sort_order", "payload_kind", "learning_data_kind", "coordinate_policy", "certainty_scale", "included", "exclusion_reason"],
    )

    prepared: list[tuple[EvaluationFold, SGRFMethodSpec, Path, dict[str, Any], dict[str, Any], Path, str, list[dict[str, Any]]]] = []
    scenario_order = sorted(
        {fold.scenario for fold in selected_folds},
        key=lambda scenario: (SGRF_SCENARIO_EXECUTION_PRIORITY.get(scenario, 10_000), scenario),
    )
    for scenario in scenario_order:
        scenario_folds = [fold for fold in selected_folds if fold.scenario == scenario]
        for method in execution_methods:
            for fold in scenario_folds:
                manifest_path, model_manifest = _model_manifest(models_root, fold.scenario, fold.fold_id, method.method_id)
                job, job_dir, fingerprint, rows = _prepare_prediction_job(
                    output_root,
                    fold,
                    method,
                    model_manifest,
                    manifest_path,
                    gt_rows,
                    rejection_policy=rejection_policy,
                    certainty_threshold_normalized=certainty_threshold_normalized,
                    model_load_policy=model_load_policy,
                    seed=seed,
                    fail_fast=fail_fast,
                    progress_every=progress_every if progress else 0,
                )
                prepared.append((fold, method, manifest_path, model_manifest, job, job_dir, fingerprint, rows))

    run_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    to_run: list[tuple[EvaluationFold, SGRFMethodSpec, Path, dict[str, Any], dict[str, Any], Path, str, list[dict[str, Any]]]] = []
    for item in prepared:
        fold, method, _, _, job, job_dir, fingerprint, _ = item
        run_manifest = job_dir / "prediction_run.json"
        worker_csv = job_dir / "worker_predictions.csv"
        if run_manifest.is_file() and worker_csv.is_file() and not force:
            existing = json.loads(run_manifest.read_text(encoding="utf-8"))
            if str(existing.get("prediction_fingerprint", "")) != fingerprint:
                raise RuntimeError(
                    f"Existing SGRF prediction output does not match the requested model/data/options: {job_dir}. "
                    "Use --force only when replacement is intentional."
                )
            run_results[(fold.scenario, fold.fold_id, method.method_id)] = existing
        else:
            to_run.append(item)

    if aggregate_only and to_run:
        missing = [f"{item[0].scenario}/{item[0].fold_id}/{item[1].method_id}" for item in to_run]
        preview = "\n".join(f"  - {value}" for value in missing[:50])
        suffix = "" if len(missing) <= 50 else f"\n  ... and {len(missing) - 50} more"
        raise RuntimeError(
            f"aggregate-only requires all selected SGRF jobs to be complete; missing {len(missing)}/{len(prepared)}:\n"
            f"{preview}{suffix}"
        )

    resolved_workers = resolve_external_worker_count(workers, len(to_run))
    started = time.perf_counter()
    if progress:
        print("=== SGRF evaluation schedule ===", flush=True)
        print(f"Scenarios: {', '.join(scenario_order)}", flush=True)
        print(f"Methods: {len(execution_methods)} | execution_order={execution_order}", flush=True)
        print("Method execution order: " + " -> ".join(method.method_id for method in execution_methods), flush=True)
        print(
            f"Jobs: complete={len(run_results)}/{len(prepared)} | pending={len(to_run)} | workers={resolved_workers}",
            flush=True,
        )
        for scenario in scenario_order:
            scenario_items = [item for item in prepared if item[0].scenario == scenario]
            scenario_complete = sum(
                (item[0].scenario, item[0].fold_id, item[1].method_id) in run_results
                for item in scenario_items
            )
            print(f"  {scenario}: {scenario_complete}/{len(scenario_items)} complete", flush=True)
    failures: list[dict[str, str]] = []

    def execute(item):
        fold, method, _, _, job, job_dir, _, _ = item
        try:
            result = _run_prediction_job(
                sgrf_python, job, job_dir, stream_output=(resolved_workers <= 1 and progress)
            )
            return (fold.scenario, fold.fold_id, method.method_id), result, ""
        except BaseException as exc:
            return (fold.scenario, fold.fold_id, method.method_id), {}, f"{type(exc).__name__}: {exc}"

    if resolved_workers <= 1:
        total_complete = len(run_results)
        method_position = {method.method_id: index for index, method in enumerate(execution_methods, start=1)}
        scenario_fold_counts = {
            scenario: len([fold for fold in selected_folds if fold.scenario == scenario])
            for scenario in scenario_order
        }
        for index, item in enumerate(to_run, start=1):
            fold, method = item[0], item[1]
            fold_position = 1 + [f.fold_id for f in selected_folds if f.scenario == fold.scenario].index(fold.fold_id)
            if progress:
                method_done = sum(
                    (candidate.scenario, candidate.fold_id, method.method_id) in run_results
                    for candidate in selected_folds if candidate.scenario == fold.scenario
                )
                print("", flush=True)
                print(
                    f"[START] {fold.scenario} | method {method_position[method.method_id]}/{len(execution_methods)} "
                    f"{method.method_id} | fold {fold_position}/{scenario_fold_counts[fold.scenario]} {fold.fold_id}",
                    flush=True,
                )
                print(
                    f"        method folds complete={method_done}/{scenario_fold_counts[fold.scenario]} | "
                    f"overall={total_complete}/{len(prepared)} | pending={len(prepared) - total_complete}",
                    flush=True,
                )
            job_started = time.perf_counter()
            key, result, error = execute(item)
            job_elapsed = time.perf_counter() - job_started
            if error:
                failures.append({"scenario": key[0], "fold_id": key[1], "method_id": key[2], "error": error})
                if fail_fast:
                    break
            else:
                run_results[key] = result
                total_complete += 1
            if progress:
                method_done = sum(
                    (candidate.scenario, candidate.fold_id, method.method_id) in run_results
                    for candidate in selected_folds if candidate.scenario == fold.scenario
                )
                print(
                    f"[DONE]  {key[0]} | {key[2]} | {key[1]} | "
                    f"{'ERROR' if error else result.get('worker_status', result.get('status', 'ok'))} | "
                    f"wall={job_elapsed / 3600.0:.2f} h",
                    flush=True,
                )
                print(
                    f"        method folds={method_done}/{scenario_fold_counts[fold.scenario]} | "
                    f"overall={total_complete}/{len(prepared)} | pending={len(prepared) - total_complete}",
                    flush=True,
                )
                if method_done == scenario_fold_counts[fold.scenario]:
                    scenario_done_methods = sum(
                        all(
                            (candidate.scenario, candidate.fold_id, candidate_method.method_id) in run_results
                            for candidate in selected_folds if candidate.scenario == fold.scenario
                        )
                        for candidate_method in execution_methods
                    )
                    print(
                        f"[METHOD COMPLETE] {fold.scenario} | {method.method_id} | "
                        f"methods={scenario_done_methods}/{len(execution_methods)}",
                        flush=True,
                    )
    else:
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            futures = {executor.submit(execute, item): item for item in to_run}
            completed = 0
            for future in as_completed(futures):
                key, result, error = future.result()
                completed += 1
                if error:
                    failures.append({"scenario": key[0], "fold_id": key[1], "method_id": key[2], "error": error})
                else:
                    run_results[key] = result
                if progress:
                    print(f"[SGRF predict {completed}/{len(to_run)}] {key[0]}/{key[1]} {key[2]}: {'error' if error else result.get('status', 'ok')}", flush=True)
                if fail_fast and failures:
                    for pending in futures:
                        pending.cancel()
                    break
    if failures and fail_fast:
        write_csv_rows(output_root / "prediction_job_failures.csv", failures, ["scenario", "fold_id", "method_id", "error"])
        raise RuntimeError(f"SGRF baseline inference stopped after {len(failures)} external job failure(s).")

    frame_result_count = 0
    error_frame_count = 0
    video_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    gesture_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    frame_output = output_root / "frame_predictions.csv"
    error_output = output_root / "error_frames.csv"
    frame_handle = frame_output.open("w", encoding="utf-8", newline="")
    error_handle = error_output.open("w", encoding="utf-8", newline="")
    frame_writer = csv.DictWriter(frame_handle, fieldnames=FRAME_RESULT_FIELDS, extrasaction="ignore")
    error_writer = csv.DictWriter(error_handle, fieldnames=FRAME_RESULT_FIELDS, extrasaction="ignore")
    frame_writer.writeheader()
    error_writer.writeheader()
    try:
        for item in prepared:
            fold, method, model_manifest_path, _, _, job_dir, _, manifest_rows = item
            key = (fold.scenario, fold.fold_id, method.method_id)
            if key not in run_results:
                continue
            worker_rows = read_csv_rows(job_dir / "worker_predictions.csv")
            worker_by_id = {str(row["row_id"]): row for row in worker_rows}
            if len(worker_by_id) != len(manifest_rows):
                raise ValueError(f"Worker prediction row count mismatch in {job_dir}.")
            acc = FoldAccumulator()
            by_take: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source in manifest_rows:
                worker = worker_by_id.get(str(source["row_id"]))
                if worker is None:
                    raise ValueError(f"Missing worker prediction {source['row_id']} in {job_dir}.")
                operational_gt = str(source["operational_gt_state"])
                benchmark_gt = str(source["benchmark_gt_state"])
                predicted_state = str(worker.get("predicted_state", PREDICTION_ERROR) or PREDICTION_ERROR)
                raw_certainty_value: float | str = ""
                if worker.get("raw_certainty") not in (None, ""):
                    raw_certainty_value = float(worker["raw_certainty"])
                normalized_certainty_value: float | str = ""
                if worker.get("normalized_certainty") not in (None, ""):
                    normalized_certainty_value = float(worker["normalized_certainty"])
                    acc.normalized_certainties.append(float(normalized_certainty_value))
                decode_value: float | str = ""
                if worker.get("image_decode_ms") not in (None, ""):
                    decode_value = float(worker["image_decode_ms"])
                    acc.decode_ms.append(float(decode_value))
                classify_value: float | str = ""
                if worker.get("classification_ms") not in (None, ""):
                    classify_value = float(worker["classification_ms"])
                    acc.classification_ms.append(float(classify_value))
                rejected = int(str(worker.get("rejected_by_policy", "0") or "0"))
                acc.rejected_predictions += rejected
                _update_counts(acc, GESTURE_ORDER, benchmark_gt, predicted_state)
                row = {
                    "method_id": method.method_id,
                    "method_display_name": method.display_name,
                    "method_sort_order": method.sort_order,
                    "scenario": fold.scenario,
                    "fold_id": fold.fold_id,
                    "take_id": source["take_id"],
                    "frame_index": source["frame_index"],
                    "frame_relative_path": source["frame_relative_path"],
                    "public_subject_id": source["public_subject_id"],
                    "background": source["background"],
                    "gesture_id": source["gesture_id"],
                    "operational_gt_state": operational_gt,
                    "benchmark_gt_state": benchmark_gt,
                    "raw_predicted_label": worker.get("raw_predicted_label", ""),
                    "raw_predicted_state": worker.get("raw_predicted_state", ""),
                    "predicted_label": worker.get("predicted_label", ""),
                    "predicted_state": predicted_state,
                    "rejected_by_policy": rejected,
                    "raw_certainty": raw_certainty_value,
                    "normalized_certainty": normalized_certainty_value,
                    "correct_exact": int(predicted_state == benchmark_gt),
                    "error_type": _error_type(benchmark_gt, predicted_state),
                    "image_decode_ms": decode_value,
                    "classification_ms": classify_value,
                    "worker_error": worker.get("error", ""),
                    "no_hand_collapsed_for_benchmark": int(operational_gt == NO_HAND),
                }
                frame_writer.writerow(row)
                frame_result_count += 1
                by_take[str(source["take_id"])].append(row)
                if row["error_type"]:
                    error_writer.writerow(row)
                    error_frame_count += 1
            for take_id, rows in sorted(by_take.items()):
                target = str(rows[0]["gesture_id"]).upper()
                video_rows.append(
                    {
                        "method_id": method.method_id,
                        "method_display_name": method.display_name,
                        "method_sort_order": method.sort_order,
                        "scenario": fold.scenario,
                        "fold_id": fold.fold_id,
                        "take_id": take_id,
                        "public_subject_id": rows[0]["public_subject_id"],
                        "background": rows[0]["background"],
                        "gesture_id": target,
                        **_video_summary(rows, target),
                    }
                )
            fold_summary, gesture_summary, confusion = _fold_metrics(
                method,
                fold,
                acc,
                model_manifest_path=model_manifest_path,
                rejection_policy=rejection_policy,
                certainty_threshold_normalized=certainty_threshold_normalized,
                model_load_policy=model_load_policy,
            )
            fold_rows.append(fold_summary)
            gesture_rows.extend(gesture_summary)
            confusion_rows.extend(confusion)

    finally:
        frame_handle.close()
        error_handle.close()

    scenario_rows = _scenario_summary(video_rows, fold_rows)
    group_rows = _group_summary(video_rows)
    video_rows.sort(key=lambda row: (int(row["method_sort_order"]), str(row["scenario"]), str(row["fold_id"]), str(row["take_id"])))
    fold_rows.sort(key=lambda row: (int(row["method_sort_order"]), str(row["scenario"]), str(row["fold_id"])))
    gesture_rows.sort(key=lambda row: (int(row["method_sort_order"]), str(row["scenario"]), str(row["fold_id"]), str(row["gesture_id"])))
    confusion_rows.sort(key=lambda row: (int(row["method_sort_order"]), str(row["scenario"]), str(row["fold_id"]), str(row["true_state"]), str(row["predicted_state"])))

    for filename, rows in (
        ("video_summary.csv", video_rows),
        ("gesture_metrics.csv", gesture_rows),
        ("fold_summary.csv", fold_rows),
        ("scenario_summary.csv", scenario_rows),
        ("group_summary.csv", group_rows),
        ("confusion_matrix_long.csv", confusion_rows),
    ):
        write_csv_rows(output_root / filename, rows, _collect_fields(rows))

    sheets = {}
    for name, rows in (
        ("scenario_summary", scenario_rows),
        ("fold_summary", fold_rows),
        ("gesture_metrics", gesture_rows),
        ("group_summary", group_rows),
        ("video_summary", video_rows),
        ("confusion_matrix", confusion_rows),
    ):
        sheets[name] = (_collect_fields(rows), rows)
    write_workbook(output_root / "sgrf_baseline_results.xlsx", sheets)

    plots = output_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _plot_method_scenario_f1(scenario_rows, plots / "baseline_macro_f1_by_scenario.png")
    _plot_speed_quality(scenario_rows, plots / "baseline_speed_quality.png")

    if failures:
        write_csv_rows(output_root / "prediction_job_failures.csv", failures, ["scenario", "fold_id", "method_id", "error"])
    write_json(
        output_root / "evaluation_report.json",
        {
            "schema_version": "tsgr_sgrf_baseline_evaluation_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": plan_dir.name,
            "dataset_root": ".",
            "dataset_contract": "tsgr_public_dataset_v1",
            "baseline_models_dir": models_root.name,
            "ground_truth_report": gt_dir.name,
            "methods": [method.method_id for method in selected_methods],
            "execution_order": execution_order,
            "execution_method_order": [method.method_id for method in execution_methods],
            "scenario_execution_order": scenario_order,
            "aggregate_only": aggregate_only,
            "scenarios": sorted(scenarios or set()),
            "fold_filter": sorted(folds or set()),
            "rejection_policy": rejection_policy,
            "certainty_threshold_normalized": certainty_threshold_normalized,
            "model_load_policy": model_load_policy,
            "seed": seed,
            "workers_requested": workers,
            "workers_resolved": resolved_workers,
            "external_job_count": len(to_run),
            "external_job_failure_count": len(failures),
            "frame_result_count": frame_result_count,
            "error_frame_count": error_frame_count,
            "frame_output_streamed": True,
            "video_result_count": len(video_rows),
            "elapsed_wall_time_s": time.perf_counter() - started,
            "external_environment": audit.payload,
            "benchmark_ground_truth_policy": "NO_HAND is preserved in operational_gt_state and collapsed to NO_GESTURE only for common static-classifier metrics.",
            "input_contract": "Canonical raw image only; no TSGRF landmarks, MediaPipe cache, or detected hand ROI are supplied to SGRF methods.",
            "frame_level_xlsx_exported": False,
        },
    )
    return output_root
