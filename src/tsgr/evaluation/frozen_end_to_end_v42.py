"""Leakage-safe acceptance calibration and end-to-end evaluation for fixed TSGR-F routing.

Only the explicit TRAIN-only acceptance layer is calibrated. Held-out evaluation covers
GESTURE_*, NO_GESTURE and NO_HAND states using cached IMAGE/VIDEO processing
reports; MediaPipe is never re-run by this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from tsgr.dataset.contract import resolve_dataset_root_from_plan
from tsgr.evaluation.experiment_evaluator import (
    MISSING_HAND,
    _binary_metrics,
    _error_type,
    _load_run_features,
    _operational_state_match,
    _prediction_from_missing,
)
from tsgr.evaluation.frozen_routing_v40 import (
    _load_model,
    _query_class_scores,
    _query_fused_score,
    _query_pair_scores,
    _run_image_features,
    _safe_scale,
    _top2_gate,
)
from tsgr.evaluation.ground_truth import NO_GESTURE, NO_HAND, UNANNOTATED, state_gesture_id
from tsgr.evaluation.io_utils import read_csv_rows, write_workbook
from tsgr.evaluation.ranking_support import class_fused_score, loo_scores
from tsgr.evaluation.temporal_sequence import sequence_metrics
from tsgr.utils.serialization import write_json
from tsgr.visualization.landmark_overlay import draw_landmarks_overlay

CALIBRATION_SCHEMA = "tsgrf_frozen_acceptance_calibration_v42"
EVALUATION_SCHEMA = "tsgrf_frozen_end_to_end_evaluation_v42"
DEFAULT_OBJECTIVE = "mcc"
DEFAULT_MIN_POSITIVE_RECALL = 0.99
ACCEPTANCE_OBJECTIVES = ("mcc", "balanced_accuracy", "f1")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{q}": float("nan") for q in ("min", "q01", "q05", "median", "q95", "q99", "max")}
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_q01": float(np.quantile(arr, 0.01)),
        f"{prefix}_q05": float(np.quantile(arr, 0.05)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q95": float(np.quantile(arr, 0.95)),
        f"{prefix}_q99": float(np.quantile(arr, 0.99)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _binary_stats(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    metrics = _binary_metrics(tp, fp, tn, fn)
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "mcc": float(metrics["mcc"]),
    }


def select_acceptance_threshold(
    scores: np.ndarray,
    positive: np.ndarray,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    min_positive_recall: float = DEFAULT_MIN_POSITIVE_RECALL,
) -> dict[str, Any]:
    """Select a deterministic lower-is-better threshold using TRAIN labels only.

    Candidate boundaries are the distinct observed score values. The primary objective
    is optimized subject to a minimum positive recall constraint. Ties prefer higher
    balanced accuracy, then specificity, then recall, then the lower threshold.
    """
    objective = str(objective).strip().lower()
    if objective not in ACCEPTANCE_OBJECTIVES:
        raise ValueError(f"Unsupported acceptance objective: {objective}")
    if not 0.0 <= float(min_positive_recall) <= 1.0:
        raise ValueError("min_positive_recall must be in [0,1]")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)
    finite = np.isfinite(s)
    s = s[finite]
    y = y[finite]
    if s.size == 0 or not np.any(y) or not np.any(~y):
        raise ValueError("Acceptance calibration requires finite positive and negative TRAIN scores.")

    order = np.argsort(s, kind="stable")
    s = s[order]
    y = y[order]
    total_pos = int(np.sum(y))
    total_neg = int(np.sum(~y))
    cum_pos = np.cumsum(y.astype(np.int64))
    cum_neg = np.cumsum((~y).astype(np.int64))
    last_of_value = np.flatnonzero(np.r_[s[1:] != s[:-1], True])

    candidates: list[dict[str, Any]] = []
    for idx in last_of_value.tolist():
        tp = int(cum_pos[idx])
        fp = int(cum_neg[idx])
        fn = total_pos - tp
        tn = total_neg - fp
        stats = _binary_stats(tp, fp, tn, fn)
        candidates.append({"threshold": float(s[idx]), "tp": tp, "fp": fp, "tn": tn, "fn": fn, **stats})

    constrained = [row for row in candidates if float(row["recall"]) + 1e-15 >= float(min_positive_recall)]
    pool = constrained if constrained else candidates
    chosen = max(
        pool,
        key=lambda row: (
            float(row[objective]),
            float(row["balanced_accuracy"]),
            float(row["specificity"]),
            float(row["recall"]),
            -float(row["threshold"]),
        ),
    )
    return {
        **chosen,
        "objective": objective,
        "objective_value": float(chosen[objective]),
        "min_positive_recall": float(min_positive_recall),
        "constraint_satisfied": int(bool(constrained)),
        "candidate_count": len(candidates),
        "positive_count": total_pos,
        "negative_count": total_neg,
    }


def _loo_pair_scores_all(values: np.ndarray, labels: np.ndarray, class_a: int, class_b: int, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.full((len(x), 2), np.inf, dtype=np.float64)
    for col, cls in enumerate((int(class_a), int(class_b))):
        idx = np.flatnonzero(y == cls)
        delta = x[:, None, :] - x[idx][None, :, :]
        dist = np.sqrt(np.sum(delta * delta * w[None, None, :], axis=2))
        own = {int(global_idx): local_idx for local_idx, global_idx in enumerate(idx.tolist())}
        for sample_idx in range(len(x)):
            row = dist[sample_idx].copy()
            local = own.get(sample_idx)
            if local is not None:
                row[local] = np.inf
            out[sample_idx, col] = float(np.min(row))
    return out


def _robust_z_scores_to_class_loo(values: np.ndarray, labels: np.ndarray, class_index: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    cls_idx = np.flatnonzero(y == int(class_index))
    if cls_idx.size < 3:
        raise ValueError("At least three class samples are required for robust LOO conformity calibration.")
    full_center, full_scale = _safe_scale(x[cls_idx])
    result = np.empty(len(x), dtype=np.float64)
    cls_set = set(int(v) for v in cls_idx.tolist())
    for i in range(len(x)):
        if i in cls_set:
            ref = x[cls_idx[cls_idx != i]]
            center, scale = _safe_scale(ref)
        else:
            center, scale = full_center, full_scale
        z = (x[i] - center) / scale
        result[i] = float(np.sqrt(np.mean(z * z)))
    return result


def _training_candidate_scores(model: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    gestures = tuple(str(g) for g in model["gesture_ids"])
    gi = {g: i for i, g in enumerate(gestures)}
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    train3 = np.asarray(arrays["train3"], dtype=np.float64)
    train2 = np.asarray(arrays["train2"], dtype=np.float64)
    w3 = np.asarray(arrays["w3"], dtype=np.float64)
    w2 = np.asarray(arrays["w2"], dtype=np.float64)
    global_scores, _ = loo_scores(train3, labels, len(gestures), w3)
    result: dict[tuple[str, str], np.ndarray] = {}
    for idx, gesture in enumerate(gestures):
        result[("GLOBAL", gesture)] = np.asarray(global_scores[:, idx], dtype=np.float64)

    alpha = float(model["os_specialist"]["alpha"])
    result[("OS", "O")] = class_fused_score(train3, train2, labels, gi["O"], w3, w2, alpha)
    result[("OS", "S")] = class_fused_score(train3, train2, labels, gi["S"], w3, w2, alpha)

    combined = np.column_stack((train3, train2))
    iy_idx = np.asarray(arrays["iy_indices"], dtype=np.int64)
    iy_scores = _loo_pair_scores_all(
        combined[:, iy_idx], labels, gi["I"], gi["Y"], np.asarray(arrays["iy_weights"], dtype=np.float64)
    )
    result[("IY", "I")] = iy_scores[:, 0]
    result[("IY", "Y")] = iy_scores[:, 1]

    c_idx = np.asarray(arrays["c_indices"], dtype=np.int64)
    result[("C_RESCUE", "C")] = _robust_z_scores_to_class_loo(combined[:, c_idx], labels, gi["C"])
    return result


def build_frozen_acceptance_calibration(
    experiment_plan_dir: str | Path,
    *,
    frozen_model_dir: str | Path,
    output_dir: str | Path,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    objective: str = DEFAULT_OBJECTIVE,
    min_positive_recall: float = DEFAULT_MIN_POSITIVE_RECALL,
    progress: bool = True,
) -> Path:
    """Build route/class-specific acceptance thresholds from frozen TRAIN arrays only."""
    plan = Path(experiment_plan_dir)
    models = Path(frozen_model_dir)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    guards: list[dict[str, Any]] = []
    fold_count = 0
    for fold_json in sorted(plan.glob("*/*/fold.json")):
        meta = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(meta["scenario"])
        fold_id = str(meta["fold_id"])
        if scenarios and scenario not in scenarios:
            continue
        if folds and fold_id not in folds:
            continue
        if str(meta.get("fold_status", "active")) != "active":
            continue
        try:
            model, arrays = _load_model(models, scenario, fold_id)
        except FileNotFoundError:
            continue
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        gestures = tuple(str(g) for g in model["gesture_ids"])
        gi = {g: i for i, g in enumerate(gestures)}
        candidate_scores = _training_candidate_scores(model, arrays)
        thresholds: dict[str, dict[str, Any]] = {}
        fold_rows: list[dict[str, Any]] = []
        for (route, gesture), scores in candidate_scores.items():
            positive = labels == gi[gesture]
            selected = select_acceptance_threshold(
                scores,
                positive,
                objective=objective,
                min_positive_recall=min_positive_recall,
            )
            key = f"{route}:{gesture}"
            thresholds[key] = {
                "route": route,
                "gesture_id": gesture,
                "threshold": float(selected["threshold"]),
                "objective": objective,
                "objective_value": float(selected["objective_value"]),
                "min_positive_recall": float(min_positive_recall),
                "constraint_satisfied": bool(selected["constraint_satisfied"]),
                "train_metrics": {k: selected[k] for k in ("tp", "fp", "tn", "fn", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc")},
            }
            pos_scores = np.asarray(scores[positive], dtype=np.float64)
            neg_scores = np.asarray(scores[~positive], dtype=np.float64)
            row = {
                "scenario": scenario,
                "fold_id": fold_id,
                "route": route,
                "gesture_id": gesture,
                "threshold": float(selected["threshold"]),
                "objective": objective,
                "objective_value": float(selected["objective_value"]),
                "min_positive_recall": float(min_positive_recall),
                "constraint_satisfied": int(selected["constraint_satisfied"]),
                "positive_count": int(selected["positive_count"]),
                "negative_count": int(selected["negative_count"]),
                "train_tp": int(selected["tp"]),
                "train_fp": int(selected["fp"]),
                "train_tn": int(selected["tn"]),
                "train_fn": int(selected["fn"]),
                "train_precision": float(selected["precision"]),
                "train_recall": float(selected["recall"]),
                "train_specificity": float(selected["specificity"]),
                "train_f1": float(selected["f1"]),
                "train_balanced_accuracy": float(selected["balanced_accuracy"]),
                "train_mcc": float(selected["mcc"]),
                **_quantiles(pos_scores, "positive_score"),
                **_quantiles(neg_scores, "negative_score"),
            }
            rows.append(row)
            fold_rows.append(row)

        fold_out = out / scenario / fold_id
        fold_out.mkdir(parents=True, exist_ok=True)
        source_model_path = models / scenario / fold_id / "model.json"
        payload = {
            "schema_version": CALIBRATION_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "fold_id": fold_id,
            "gesture_ids": list(gestures),
            "source_frozen_model": str(source_model_path.relative_to(models)).replace("\\", "/"),
            "source_frozen_model_sha256": _sha256(source_model_path),
            "selection_policy": {
                "score_direction": "lower_is_more_conforming",
                "objective": objective,
                "min_positive_recall": float(min_positive_recall),
                "negative_proxy": "all other TRAIN gesture classes; no held-out NO_GESTURE/NO_HAND frames are used",
                "tie_break": "balanced_accuracy_then_specificity_then_recall_then_lower_threshold",
            },
            "thresholds": thresholds,
            "leakage_contract": {
                "test_processing_read_during_build": False,
                "test_ground_truth_read_during_build": False,
                "heldout_no_gesture_used_for_calibration": False,
                "all_thresholds_source": "fold_TRAIN_only",
            },
        }
        write_json(fold_out / "acceptance.json", payload)
        guards.append({
            "scenario": scenario,
            "fold_id": fold_id,
            "phase": "ACCEPTANCE_BUILD_TRAIN_ONLY",
            "test_processing_report_read": 0,
            "ground_truth_report_read": 0,
            "heldout_no_gesture_read": 0,
            "fit_on_test": 0,
            "acceptance_json": str((fold_out / "acceptance.json").relative_to(out)).replace("\\", "/"),
        })
        fold_count += 1
        if progress:
            global_mean_recall = float(np.mean([float(r["train_recall"]) for r in fold_rows if r["route"] == "GLOBAL"]))
            print(f"[{scenario}/{fold_id}] acceptance thresholds={len(fold_rows)} | mean GLOBAL train recall={global_mean_recall:.4f}", flush=True)

    if fold_count == 0:
        raise ValueError("No active folds with frozen routing models were selected for acceptance calibration.")
    _write_csv(out / "acceptance_thresholds.csv", rows)
    _write_csv(out / "leakage_guard.csv", guards)
    write_workbook(
        out / "frozen_acceptance_calibration.xlsx",
        {
            "thresholds": (list(rows[0]) if rows else [], rows),
            "leakage_guard": (list(guards[0]) if guards else [], guards),
        },
    )
    report = {
        "schema_version": CALIBRATION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_plan": str(plan),
        "frozen_model_dir": str(models),
        "fold_count": fold_count,
        "objective": objective,
        "min_positive_recall": float(min_positive_recall),
        "notes": [
            "TRAIN-only acceptance calibration; API accepts no processing-report or ground-truth path.",
            "The ranking/routing architecture is not modified.",
            "Other TRAIN gesture classes serve only as an impostor proxy; held-out NO_GESTURE/NO_HAND frames remain untouched until evaluation.",
        ],
    }
    write_json(out / "frozen_acceptance_build_report.json", report)
    return out / "frozen_acceptance_build_report.json"


def _load_acceptance(root: Path, scenario: str, fold_id: str, frozen_model_path: Path) -> dict[str, Any]:
    path = root / scenario / fold_id / "acceptance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen acceptance calibration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(f"Unsupported acceptance schema in {path}: {payload.get('schema_version')}")
    expected = str(payload.get("source_frozen_model_sha256", ""))
    actual = _sha256(frozen_model_path)
    if expected and expected != actual:
        raise ValueError(f"Acceptance calibration was built for a different frozen model: {scenario}/{fold_id}")
    return payload


def predict_frozen_detailed(
    model: dict[str, Any],
    arrays: dict[str, np.ndarray],
    qfull: np.ndarray,
    q2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return frozen ranking stages plus the route-specific acceptance score."""
    gestures = tuple(str(g) for g in model["gesture_ids"])
    gi = {g: i for i, g in enumerate(gestures)}
    fidx = np.asarray(arrays["feature_indices"], dtype=np.int64)
    q3 = np.asarray(qfull[:, fidx], dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    train3 = np.asarray(arrays["train3"], dtype=np.float64)
    train2 = np.asarray(arrays["train2"], dtype=np.float64)
    w3 = np.asarray(arrays["w3"], dtype=np.float64)
    w2 = np.asarray(arrays["w2"], dtype=np.float64)

    global_scores = _query_class_scores(q3, train3, labels, w3, len(gestures))
    order = np.argsort(global_scores, axis=1)
    base = order[:, 0].copy()
    pred_os = base.copy()
    alpha = float(model["os_specialist"]["alpha"])
    score_o = _query_fused_score(q3, q2, train3, train2, labels, gi["O"], w3, w2, alpha)
    score_s = _query_fused_score(q3, q2, train3, train2, labels, gi["S"], w3, w2, alpha)
    gate_os = _top2_gate(global_scores, gi["O"], gi["S"])
    pred_os[gate_os] = np.where(
        (score_o - score_s)[gate_os] <= float(model["os_specialist"]["threshold"]),
        gi["O"],
        gi["S"],
    )

    combined = np.column_stack((q3, q2))
    train_combined = np.column_stack((train3, train2))
    iy_idx = np.asarray(arrays["iy_indices"], dtype=np.int64)
    iy_scores = _query_pair_scores(
        combined[:, iy_idx], train_combined[:, iy_idx], labels, gi["I"], gi["Y"], np.asarray(arrays["iy_weights"], dtype=np.float64)
    )
    gate_iy = _top2_gate(global_scores, gi["I"], gi["Y"])
    pred_iy = pred_os.copy()
    pred_iy[gate_iy] = np.where(iy_scores[gate_iy, 0] <= iy_scores[gate_iy, 1], gi["I"], gi["Y"])

    cidx = np.asarray(arrays["c_indices"], dtype=np.int64)
    cx = combined[:, cidx]
    za = np.sqrt(np.mean(((cx - arrays["c_a_center"]) / arrays["c_a_scale"]) ** 2, axis=1))
    zc = np.sqrt(np.mean(((cx - arrays["c_c_center"]) / arrays["c_c_scale"]) ** 2, axis=1))
    rescue = (base == gi["A"]) & (zc < za)
    final = pred_iy.copy()
    final[rescue] = gi["C"]

    n = len(final)
    route = np.full(n, "GLOBAL", dtype="U16")
    route[gate_os] = "OS"
    route[gate_iy] = "IY"
    route[rescue] = "C_RESCUE"
    acceptance_score = global_scores[np.arange(n), final].copy()
    os_mask_o = gate_os & (final == gi["O"])
    os_mask_s = gate_os & (final == gi["S"])
    acceptance_score[os_mask_o] = score_o[os_mask_o]
    acceptance_score[os_mask_s] = score_s[os_mask_s]
    iy_mask_i = gate_iy & (final == gi["I"])
    iy_mask_y = gate_iy & (final == gi["Y"])
    acceptance_score[iy_mask_i] = iy_scores[iy_mask_i, 0]
    acceptance_score[iy_mask_y] = iy_scores[iy_mask_y, 1]
    acceptance_score[rescue] = zc[rescue]

    return {
        "q3": q3,
        "global_scores": global_scores,
        "global_order": order,
        "baseline": base,
        "after_os": pred_os,
        "after_iy": pred_iy,
        "final": final,
        "route": route,
        "acceptance_score": acceptance_score,
        "os_score_o": score_o,
        "os_score_s": score_s,
        "iy_scores": iy_scores,
        "c_score_a": za,
        "c_score_c": zc,
    }


def _acceptance_thresholds_for_predictions(
    acceptance: dict[str, Any], gestures: tuple[str, ...], route: np.ndarray, final: np.ndarray
) -> np.ndarray:
    thresholds = acceptance.get("thresholds", {})
    out = np.empty(len(final), dtype=np.float64)
    for i in range(len(final)):
        gesture = gestures[int(final[i])]
        key = f"{str(route[i])}:{gesture}"
        entry = thresholds.get(key)
        if entry is None:
            raise ValueError(f"Missing acceptance threshold {key} for {acceptance.get('scenario')}/{acceptance.get('fold_id')}")
        out[i] = float(entry["threshold"])
    return out


@dataclass(slots=True)
class E2EAccumulator:
    true_pred: Counter[tuple[str, str]] = field(default_factory=Counter)
    per_gesture: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    route_usage: Counter[tuple[str, str]] = field(default_factory=Counter)
    error_types: Counter[str] = field(default_factory=Counter)
    evaluated: int = 0
    input_available: int = 0
    gesture_frames: int = 0
    no_gesture_frames: int = 0
    no_hand_frames: int = 0
    exact_correct: int = 0
    raw_gesture_input: int = 0
    raw_ranking_correct: int = 0
    accepted_gesture_frames: int = 0
    accepted_correct_gesture_frames: int = 0

    def merge(self, other: "E2EAccumulator") -> None:
        self.true_pred.update(other.true_pred)
        for key, value in other.per_gesture.items():
            self.per_gesture[key].update(value)
        self.route_usage.update(other.route_usage)
        self.error_types.update(other.error_types)
        for attr in (
            "evaluated", "input_available", "gesture_frames", "no_gesture_frames", "no_hand_frames",
            "exact_correct", "raw_gesture_input", "raw_ranking_correct", "accepted_gesture_frames",
            "accepted_correct_gesture_frames",
        ):
            setattr(self, attr, int(getattr(self, attr)) + int(getattr(other, attr)))


def _update_acc(acc: E2EAccumulator, gestures: tuple[str, ...], row: dict[str, Any]) -> None:
    gt = str(row["gt_state"])
    pred = str(row["predicted_state"])
    acc.evaluated += 1
    acc.input_available += int(row.get("prediction_input_available", 0))
    acc.exact_correct += int(_operational_state_match(gt, pred))
    acc.true_pred[(gt, pred)] += 1
    if gt.startswith("GESTURE_"):
        acc.gesture_frames += 1
        if int(row.get("prediction_input_available", 0)):
            acc.raw_gesture_input += 1
            acc.raw_ranking_correct += int(str(row.get("final_gesture", "")) == str(row.get("gt_gesture", "")))
        if pred.startswith("GESTURE_"):
            acc.accepted_gesture_frames += 1
            acc.accepted_correct_gesture_frames += int(pred == gt)
    elif gt == NO_GESTURE:
        acc.no_gesture_frames += 1
    elif gt == NO_HAND:
        acc.no_hand_frames += 1
    route = str(row.get("acceptance_route", ""))
    final_gesture = str(row.get("final_gesture", ""))
    if route and final_gesture:
        acc.route_usage[(route, final_gesture)] += 1
    error = str(row.get("error_type", ""))
    if error:
        acc.error_types[error] += 1
    for gesture in gestures:
        counts = acc.per_gesture[gesture]
        positive_true = gt == f"GESTURE_{gesture}"
        positive_pred = pred == f"GESTURE_{gesture}"
        if positive_true and positive_pred:
            counts["tp"] += 1
        elif (not positive_true) and positive_pred:
            counts["fp"] += 1
        elif positive_true and (not positive_pred):
            counts["fn"] += 1
        else:
            counts["tn"] += 1


def _state_counts(acc: E2EAccumulator) -> dict[str, int]:
    no_gesture_correct = int(acc.true_pred[(NO_GESTURE, NO_GESTURE)])
    no_hand_missing = int(acc.true_pred[(NO_HAND, MISSING_HAND)])
    false_gesture_no_gesture = sum(v for (gt, pred), v in acc.true_pred.items() if gt == NO_GESTURE and pred.startswith("GESTURE_"))
    false_gesture_no_hand = sum(v for (gt, pred), v in acc.true_pred.items() if gt == NO_HAND and pred.startswith("GESTURE_"))
    unexpected_missing = sum(v for (gt, pred), v in acc.true_pred.items() if gt != NO_HAND and pred == MISSING_HAND)
    return {
        "no_gesture_correct_rejections": no_gesture_correct,
        "no_hand_correct_missing": no_hand_missing,
        "false_gesture_on_no_gesture": int(false_gesture_no_gesture),
        "false_gesture_on_no_hand": int(false_gesture_no_hand),
        "unexpected_missing_hand": int(unexpected_missing),
    }


def _summary_row(scope: dict[str, Any], acc: E2EAccumulator, gestures: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = _state_counts(acc)
    gesture_rows: list[dict[str, Any]] = []
    for gesture in gestures:
        c = acc.per_gesture[gesture]
        metrics = _binary_metrics(c["tp"], c["fp"], c["tn"], c["fn"])
        gesture_rows.append({**scope, "gesture_id": gesture, "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"], **metrics})
    macro_f1 = float(np.mean([float(r["f1"]) for r in gesture_rows])) if gesture_rows else float("nan")
    macro_mcc = float(np.mean([float(r["mcc"]) for r in gesture_rows])) if gesture_rows else float("nan")
    any_tp = sum(v for (gt, pred), v in acc.true_pred.items() if gt.startswith("GESTURE_") and pred.startswith("GESTURE_"))
    any_fn = sum(v for (gt, pred), v in acc.true_pred.items() if gt.startswith("GESTURE_") and not pred.startswith("GESTURE_"))
    any_fp = sum(v for (gt, pred), v in acc.true_pred.items() if not gt.startswith("GESTURE_") and pred.startswith("GESTURE_"))
    any_tn = acc.evaluated - any_tp - any_fn - any_fp
    any_metrics = _binary_metrics(any_tp, any_fp, any_tn, any_fn)
    summary = {
        **scope,
        "evaluated_frames": acc.evaluated,
        "prediction_input_available_frames": acc.input_available,
        "input_coverage": _safe_div(acc.input_available, acc.evaluated),
        "gesture_frames": acc.gesture_frames,
        "no_gesture_frames": acc.no_gesture_frames,
        "no_hand_frames": acc.no_hand_frames,
        "exact_state_accuracy": _safe_div(acc.exact_correct, acc.evaluated),
        "raw_ranking_accuracy_on_gesture_given_input": _safe_div(acc.raw_ranking_correct, acc.raw_gesture_input),
        "gesture_coverage": _safe_div(acc.accepted_gesture_frames, acc.gesture_frames),
        "conditional_gesture_accuracy_when_accepted": _safe_div(acc.accepted_correct_gesture_frames, acc.accepted_gesture_frames),
        "gesture_frame_end_to_end_accuracy": _safe_div(acc.accepted_correct_gesture_frames, acc.gesture_frames),
        "no_gesture_correct_rejection_rate": _safe_div(state["no_gesture_correct_rejections"], acc.no_gesture_frames),
        "false_gesture_rate_on_no_gesture": _safe_div(state["false_gesture_on_no_gesture"], acc.no_gesture_frames),
        "no_hand_recall": _safe_div(state["no_hand_correct_missing"], acc.no_hand_frames),
        "false_gesture_rate_on_no_hand": _safe_div(state["false_gesture_on_no_hand"], acc.no_hand_frames),
        "unexpected_missing_hand_rate": _safe_div(state["unexpected_missing_hand"], acc.evaluated - acc.no_hand_frames),
        "any_gesture_precision": float(any_metrics["precision"]),
        "any_gesture_recall": float(any_metrics["recall"]),
        "any_gesture_f1": float(any_metrics["f1"]),
        "any_gesture_mcc": float(any_metrics["mcc"]),
        "macro_gesture_f1": macro_f1,
        "macro_gesture_mcc": macro_mcc,
        **state,
    }
    return summary, gesture_rows


def _video_metrics(rows: list[dict[str, Any]], target_gesture: str, *, max_lag_frames: int) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: int(r["frame_index"]))
    ref_target = np.asarray([str(r["gt_state"]) == f"GESTURE_{target_gesture}" for r in rows], dtype=bool)
    pred_target = np.asarray([str(r["predicted_state"]) == f"GESTURE_{target_gesture}" for r in rows], dtype=bool)
    ref_any = np.asarray([str(r["gt_state"]).startswith("GESTURE_") for r in rows], dtype=bool)
    pred_any = np.asarray([str(r["predicted_state"]).startswith("GESTURE_") for r in rows], dtype=bool)
    target = sequence_metrics(ref_target, pred_target, max_abs_lag_frames=max_lag_frames)
    anym = sequence_metrics(ref_any, pred_any, max_abs_lag_frames=max_lag_frames)
    exact = sum(_operational_state_match(str(r["gt_state"]), str(r["predicted_state"])) for r in rows)
    return {
        "frame_count": len(rows),
        "exact_state_accuracy": _safe_div(exact, len(rows)),
        "target_gesture_sequence_mcc": target.mcc,
        "target_gesture_temporal_iou": target.temporal_iou,
        "target_best_lag_mcc": target.best_lag_mcc,
        "target_best_lag_frames": target.best_lag_frames,
        "target_onset_error_frames": target.onset_error_frames,
        "target_offset_error_frames": target.offset_error_frames,
        "target_predicted_segment_count": target.predicted_segment_count,
        "target_longest_predicted_segment_frames": target.longest_predicted_segment_frames,
        "any_gesture_sequence_mcc": anym.mcc,
        "any_gesture_temporal_iou": anym.temporal_iou,
        "any_best_lag_mcc": anym.best_lag_mcc,
        "any_best_lag_frames": anym.best_lag_frames,
        "any_onset_error_frames": anym.onset_error_frames,
        "any_offset_error_frames": anym.offset_error_frames,
        "any_predicted_segment_count": anym.predicted_segment_count,
        "any_longest_predicted_segment_frames": anym.longest_predicted_segment_frames,
    }


def _processing_manifest(report_dir: Path) -> dict[str, dict[str, str]]:
    path = report_dir / "test_take_mediapipe_status.csv"
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Missing or empty processing manifest: {path}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        take = str(row["take_id"])
        if take in result:
            raise ValueError(f"Duplicate take_id in processing report: {take}")
        result[take] = row
    return result


def _ground_truth(report_dir: Path) -> dict[str, dict[int, dict[str, str]]]:
    path = report_dir / "frame_ground_truth.csv"
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Missing or empty ground truth: {path}")
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["take_id"])][int(row["frame_index"])] = row
    return grouped


def evaluate_frozen_end_to_end(
    experiment_plan_dir: str | Path,
    *,
    frozen_model_dir: str | Path,
    acceptance_dir: str | Path,
    test_processing_report: str | Path,
    ground_truth_report: str | Path,
    output_dir: str | Path,
    branch_name: str,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    sequence_max_lag_frames: int = 10,
    prediction_batch_size: int = 64,
    copy_problem_cases: bool = True,
    max_problem_cases_per_error_type_fold: int = 20,
    dataset_root_override: str | Path | None = None,
    progress: bool = True,
) -> Path:
    """Evaluate the immutable fixed routing + TRAIN-only acceptance layer on all GT states."""
    plan = Path(experiment_plan_dir)
    dataset = resolve_dataset_root_from_plan(plan, override=dataset_root_override)
    models = Path(frozen_model_dir)
    acceptance_root = Path(acceptance_dir)
    proc = Path(test_processing_report)
    gtroot = Path(ground_truth_report)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    if int(prediction_batch_size) < 1:
        raise ValueError("prediction_batch_size must be >= 1")
    manifest = _processing_manifest(proc)
    gt_by_take = _ground_truth(gtroot)

    selected: list[tuple[Path, str, str, dict[str, Any], dict[str, np.ndarray], dict[str, Any], set[str]]] = []
    required_takes: set[str] = set()
    for fold_json in sorted(plan.glob("*/*/fold.json")):
        meta = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(meta["scenario"])
        fold_id = str(meta["fold_id"])
        if scenarios and scenario not in scenarios:
            continue
        if folds and fold_id not in folds:
            continue
        if str(meta.get("fold_status", "active")) != "active":
            continue
        try:
            model, arrays = _load_model(models, scenario, fold_id)
        except FileNotFoundError:
            continue
        model_path = models / scenario / fold_id / "model.json"
        acceptance = _load_acceptance(acceptance_root, scenario, fold_id, model_path)
        if str(model["branch_name"]) != branch_name:
            raise ValueError(f"Branch mismatch for {scenario}/{fold_id}: model={model['branch_name']} requested={branch_name}")
        takes = {str(r["take_id"]) for r in read_csv_rows(fold_json.parent / "test_takes.csv")}
        selected.append((fold_json, scenario, fold_id, model, arrays, acceptance, takes))
        required_takes.update(takes)
    if not selected:
        raise ValueError("No active folds with frozen routing and acceptance models were selected.")
    missing_proc = sorted(required_takes - set(manifest))
    missing_gt = sorted(required_takes - set(gt_by_take))
    if missing_proc:
        raise ValueError(f"Processing report is missing {len(missing_proc)} required takes; first={missing_proc[:10]}")
    if missing_gt:
        raise ValueError(f"Ground truth is missing {len(missing_gt)} required takes; first={missing_gt[:10]}")

    out.mkdir(parents=True)
    frame_fields = [
        "scenario", "fold_id", "take_id", "public_subject_id", "background", "gesture_id", "frame_index", "frame_relative_path",
        "gt_state", "gt_gesture", "cached_status", "prediction_input_available", "baseline_gesture", "after_os_gesture", "after_iy_gesture",
        "final_gesture", "acceptance_route", "acceptance_score", "acceptance_threshold", "accepted", "predicted_state", "prediction_reason",
        "global_top1_gesture", "global_top2_gesture", "global_top3_gesture", "raw_ranking_correct", "operational_state_correct", "error_type",
    ]
    frame_handle = (out / "frame_predictions.csv").open("w", encoding="utf-8", newline="")
    error_handle = (out / "error_frames.csv").open("w", encoding="utf-8", newline="")
    frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields, extrasaction="ignore")
    error_writer = csv.DictWriter(error_handle, fieldnames=frame_fields, extrasaction="ignore")
    frame_writer.writeheader(); error_writer.writeheader()

    fold_rows: list[dict[str, Any]] = []
    gesture_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    error_type_rows: list[dict[str, Any]] = []
    confusion_fold: Counter[tuple[str, str, str, str]] = Counter()
    scenario_acc: dict[str, E2EAccumulator] = defaultdict(E2EAccumulator)
    problem_counts: Counter[tuple[str, str, str]] = Counter()
    guards: list[dict[str, Any]] = []

    try:
        for fold_json, scenario, fold_id, model, arrays, acceptance, takes in selected:
            gestures = tuple(str(g) for g in model["gesture_ids"])
            acc = E2EAccumulator()
            for take_idx, take in enumerate(sorted(takes), start=1):
                mr = manifest[take]
                run = Path(str(mr.get("run_path", "")))
                run = run if run.is_absolute() else dataset / run
                if not run.is_dir():
                    raise FileNotFoundError(f"Cached run for take {take} does not exist: {run}")
                frame_indices, statuses, values = _load_run_features(run, branch_name)
                frame_pos = {int(v): i for i, v in enumerate(frame_indices.tolist())}
                imgmat, imgok, imgids = _run_image_features(run, frame_indices)
                expected_imgids = tuple(str(v) for v in arrays["image_feature_ids"].tolist())
                if imgids and tuple(imgids) != expected_imgids:
                    raise ValueError(f"IMAGE feature schema mismatch for {take}")
                gt_rows = [
                    row for _, row in sorted(gt_by_take[take].items())
                    if str(row.get("evaluation_state", "")) != UNANNOTATED
                ]
                local_rows: list[dict[str, Any]] = []

                # Predict complete cached inputs in bounded vectorized batches while
                # preserving the exact fixed-model mathematics.
                details_by_idx: dict[int, dict[str, Any]] = {}
                eligible_indices: list[int] = []
                for gt_row_pre in gt_rows:
                    frame_index_pre = int(gt_row_pre["frame_index"])
                    idx_pre = frame_pos.get(frame_index_pre)
                    if idx_pre is None:
                        continue
                    input_ok_pre = bool(
                        np.isfinite(values[idx_pre]).all()
                        and idx_pre < len(imgok)
                        and bool(imgok[idx_pre])
                        and imgmat.shape[1] == len(expected_imgids)
                        and np.isfinite(imgmat[idx_pre]).all()
                    )
                    if input_ok_pre:
                        eligible_indices.append(int(idx_pre))
                for start in range(0, len(eligible_indices), int(prediction_batch_size)):
                    chunk = eligible_indices[start : start + int(prediction_batch_size)]
                    qidx = np.asarray(chunk, dtype=np.int64)
                    detailed_chunk = predict_frozen_detailed(model, arrays, values[qidx], imgmat[qidx])
                    thresholds_chunk = _acceptance_thresholds_for_predictions(
                        acceptance, gestures, detailed_chunk["route"], detailed_chunk["final"]
                    )
                    for local_i, source_idx in enumerate(chunk):
                        details_by_idx[int(source_idx)] = {
                            "baseline": int(detailed_chunk["baseline"][local_i]),
                            "after_os": int(detailed_chunk["after_os"][local_i]),
                            "after_iy": int(detailed_chunk["after_iy"][local_i]),
                            "final": int(detailed_chunk["final"][local_i]),
                            "route": str(detailed_chunk["route"][local_i]),
                            "score": float(detailed_chunk["acceptance_score"][local_i]),
                            "threshold": float(thresholds_chunk[local_i]),
                            "order": np.asarray(detailed_chunk["global_order"][local_i], dtype=np.int64),
                        }

                for gt_row in gt_rows:
                    frame_index = int(gt_row["frame_index"])
                    gt_state = str(gt_row["evaluation_state"])
                    gt_gesture = state_gesture_id(gt_state) or ""
                    idx = frame_pos.get(frame_index)
                    status = str(statuses[idx]) if idx is not None else "missing_frame_index"
                    input_ok = bool(
                        idx is not None
                        and np.isfinite(values[idx]).all()
                        and idx < len(imgok)
                        and bool(imgok[idx])
                        and imgmat.shape[1] == len(expected_imgids)
                        and np.isfinite(imgmat[idx]).all()
                    )
                    if not input_ok:
                        predicted_state, reason = _prediction_from_missing(status)
                        row = {
                            "scenario": scenario, "fold_id": fold_id, "take_id": take,
                            "public_subject_id": gt_row.get("public_subject_id", ""), "background": gt_row.get("background", ""),
                            "gesture_id": gt_row.get("gesture_id", ""), "frame_index": frame_index,
                            "frame_relative_path": gt_row.get("frame_relative_path", ""), "gt_state": gt_state, "gt_gesture": gt_gesture,
                            "cached_status": status, "prediction_input_available": 0,
                            "baseline_gesture": "", "after_os_gesture": "", "after_iy_gesture": "", "final_gesture": "",
                            "acceptance_route": "", "acceptance_score": "", "acceptance_threshold": "", "accepted": 0,
                            "predicted_state": predicted_state, "prediction_reason": reason,
                            "global_top1_gesture": "", "global_top2_gesture": "", "global_top3_gesture": "", "raw_ranking_correct": 0,
                        }
                    else:
                        cached_detail = details_by_idx[int(idx)]
                        base_i = int(cached_detail["baseline"]); os_i = int(cached_detail["after_os"]); iy_i = int(cached_detail["after_iy"]); final_i = int(cached_detail["final"])
                        route = str(cached_detail["route"]); score = float(cached_detail["score"])
                        threshold = float(cached_detail["threshold"]); accepted = bool(score <= threshold)
                        final_gesture = gestures[final_i]
                        predicted_state = f"GESTURE_{final_gesture}" if accepted else NO_GESTURE
                        order = np.asarray(cached_detail["order"], dtype=np.int64)
                        row = {
                            "scenario": scenario, "fold_id": fold_id, "take_id": take,
                            "public_subject_id": gt_row.get("public_subject_id", ""), "background": gt_row.get("background", ""),
                            "gesture_id": gt_row.get("gesture_id", ""), "frame_index": frame_index,
                            "frame_relative_path": gt_row.get("frame_relative_path", ""), "gt_state": gt_state, "gt_gesture": gt_gesture,
                            "cached_status": status, "prediction_input_available": 1,
                            "baseline_gesture": gestures[base_i], "after_os_gesture": gestures[os_i], "after_iy_gesture": gestures[iy_i], "final_gesture": final_gesture,
                            "acceptance_route": route, "acceptance_score": score, "acceptance_threshold": threshold, "accepted": int(accepted),
                            "predicted_state": predicted_state, "prediction_reason": "accepted_frozen_route" if accepted else "rejected_by_train_only_acceptance",
                            "global_top1_gesture": gestures[int(order[0])], "global_top2_gesture": gestures[int(order[1])], "global_top3_gesture": gestures[int(order[2])],
                            "raw_ranking_correct": int(bool(gt_gesture) and final_gesture == gt_gesture),
                        }
                    row["operational_state_correct"] = int(_operational_state_match(gt_state, str(row["predicted_state"])))
                    row["error_type"] = _error_type(gt_state, str(row["predicted_state"]))
                    frame_writer.writerow(row)
                    if row["error_type"]:
                        error_writer.writerow(row)
                    _update_acc(acc, gestures, row)
                    confusion_fold[(scenario, fold_id, gt_state, str(row["predicted_state"]))] += 1
                    local_rows.append(row)

                    if copy_problem_cases and row["error_type"]:
                        key = (scenario, fold_id, str(row["error_type"]))
                        if problem_counts[key] < int(max_problem_cases_per_error_type_fold):
                            case = out / "problem_cases" / scenario / fold_id / str(row["error_type"]) / f"{take}__frame_{frame_index:06d}"
                            case.mkdir(parents=True, exist_ok=True)
                            rel = str(gt_row.get("frame_relative_path", ""))
                            src = dataset / rel if rel else None
                            if src is not None and src.is_file():
                                shutil.copy2(src, case / f"query_original{src.suffix.lower() or '.jpg'}")
                                image = cv2.imread(str(src), cv2.IMREAD_COLOR)
                                if image is not None and idx is not None:
                                    try:
                                        from tsgr.evaluation.training_support import LandmarkArchiveCache
                                        lm = LandmarkArchiveCache().get(run, frame_index, "raw_image_landmarks")
                                        cv2.imwrite(str(case / "query_image_landmarks.jpg"), draw_landmarks_overlay(image, lm, title=f"GT {gt_state} | PRED {row['predicted_state']}"))
                                    except Exception:
                                        pass
                            write_json(case / "summary.json", row)
                            problem_counts[key] += 1

                target_gesture = next((str(r.get("gesture_id", "")) for r in gt_rows if str(r.get("gesture_id", ""))), "")
                if target_gesture and local_rows:
                    video_rows.append({
                        "scenario": scenario, "fold_id": fold_id, "take_id": take,
                        "public_subject_id": gt_rows[0].get("public_subject_id", ""), "background": gt_rows[0].get("background", ""),
                        "gesture_id": target_gesture,
                        **_video_metrics(local_rows, target_gesture, max_lag_frames=sequence_max_lag_frames),
                    })
                if progress and (take_idx == len(takes) or take_idx % 100 == 0):
                    print(f"[{scenario}/{fold_id}] takes {take_idx}/{len(takes)}", flush=True)

            fold_summary, fold_gesture_rows = _summary_row({"scenario": scenario, "fold_id": fold_id}, acc, gestures)
            fold_rows.append(fold_summary)
            gesture_rows.extend(fold_gesture_rows)
            scenario_acc[scenario].merge(acc)
            for (route, gesture), count in sorted(acc.route_usage.items()):
                route_rows.append({"scenario": scenario, "fold_id": fold_id, "route": route, "gesture_id": gesture, "route_prediction_count": int(count)})
            for error_type, count in sorted(acc.error_types.items()):
                error_type_rows.append({"scenario": scenario, "fold_id": fold_id, "error_type": error_type, "count": int(count)})
            guards.append({
                "scenario": scenario, "fold_id": fold_id, "phase": "HELD_OUT_END_TO_END_EVALUATION",
                "ranking_refit_on_test": 0, "acceptance_refit_on_test": 0, "feature_selection_on_test": 0,
                "ground_truth_used_for_scoring_only": 1,
            })
            if progress:
                print(f"[{scenario}/{fold_id}] exact={fold_summary['exact_state_accuracy']:.4f} | macroF1={fold_summary['macro_gesture_f1']:.4f}", flush=True)
    finally:
        frame_handle.close(); error_handle.close()

    scenario_rows: list[dict[str, Any]] = []
    scenario_gesture_rows: list[dict[str, Any]] = []
    for scenario in sorted(scenario_acc):
        # All folds in one scenario use the same gesture vocabulary by contract.
        sample_model = next(item[3] for item in selected if item[1] == scenario)
        gestures = tuple(str(g) for g in sample_model["gesture_ids"])
        summary, rows = _summary_row({"scenario": scenario}, scenario_acc[scenario], gestures)
        scenario_rows.append(summary)
        scenario_gesture_rows.extend(rows)

    confusion_rows: list[dict[str, Any]] = []
    for (scenario, fold_id, gt, pred), count in sorted(confusion_fold.items()):
        confusion_rows.append({"scenario": scenario, "fold_id": fold_id, "gt_state": gt, "predicted_state": pred, "count": int(count)})

    _write_csv(out / "fold_summary.csv", fold_rows)
    _write_csv(out / "scenario_summary.csv", scenario_rows)
    _write_csv(out / "gesture_metrics_by_fold.csv", gesture_rows)
    _write_csv(out / "gesture_metrics_by_scenario.csv", scenario_gesture_rows)
    _write_csv(out / "video_temporal_summary.csv", video_rows)
    _write_csv(out / "acceptance_route_usage.csv", route_rows)
    _write_csv(out / "error_type_summary.csv", error_type_rows)
    _write_csv(out / "confusion_matrix_long.csv", confusion_rows)
    _write_csv(out / "leakage_guard.csv", guards)
    write_workbook(
        out / "frozen_end_to_end_evaluation.xlsx",
        {
            "scenario_summary": (list(scenario_rows[0]) if scenario_rows else [], scenario_rows),
            "fold_summary": (list(fold_rows[0]) if fold_rows else [], fold_rows),
            "gesture_scenario": (list(scenario_gesture_rows[0]) if scenario_gesture_rows else [], scenario_gesture_rows),
            "gesture_fold": (list(gesture_rows[0]) if gesture_rows else [], gesture_rows),
            "temporal_video": (list(video_rows[0]) if video_rows else [], video_rows),
            "route_usage": (list(route_rows[0]) if route_rows else [], route_rows),
            "error_types": (list(error_type_rows[0]) if error_type_rows else [], error_type_rows),
            "confusion_long": (list(confusion_rows[0]) if confusion_rows else [], confusion_rows),
            "leakage_guard": (list(guards[0]) if guards else [], guards),
        },
    )
    report = {
        "schema_version": EVALUATION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_plan": str(plan),
        "frozen_model_dir": str(models),
        "acceptance_dir": str(acceptance_root),
        "test_processing_report": str(proc),
        "ground_truth_report": str(gtroot),
        "branch_name": branch_name,
        "fold_count": len(fold_rows),
        "sequence_max_lag_frames": int(sequence_max_lag_frames),
        "prediction_batch_size": int(prediction_batch_size),
        "problem_case_copy_limit_per_error_type_fold": int(max_problem_cases_per_error_type_fold),
        "notes": [
            "Full operational evaluation covers GESTURE_*, NO_GESTURE and NO_HAND states.",
            "Ranking/routing and acceptance thresholds are immutable during held-out scoring.",
            "NO_HAND is operationally correct when the framework emits MISSING_HAND.",
            "No MediaPipe inference is performed; cached IMAGE/VIDEO processing reports are reused.",
        ],
    }
    write_json(out / "frozen_end_to_end_evaluation_report.json", report)
    return out / "frozen_end_to_end_evaluation_report.json"
