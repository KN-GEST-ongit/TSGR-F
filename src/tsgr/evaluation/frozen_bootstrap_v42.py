"""Clustered bootstrap summaries for frozen TSGR-F end-to-end evaluations.

The bootstrap unit is a complete take/video, never an individual frame. Frame-level
reports are streamed once into compact per-take sufficient statistics, so publication
runs with thousands of resamples do not repeatedly materialize millions of frame rows.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tsgr.evaluation.experiment_evaluator import MISSING_HAND, _binary_metrics, _operational_state_match
from tsgr.evaluation.ground_truth import NO_GESTURE, NO_HAND
from tsgr.evaluation.io_utils import read_csv_rows, write_workbook
from tsgr.utils.serialization import write_json

SCHEMA = "tsgrf_frozen_e2e_cluster_bootstrap_v42"
DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 2026
GESTURE_ORDER = ("A", "B", "C", "E", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")
TEMPORAL_METRICS = (
    "target_gesture_sequence_mcc",
    "target_gesture_temporal_iou",
    "target_best_lag_mcc",
    "target_onset_error_frames",
    "target_offset_error_frames",
    "target_predicted_segment_count",
    "any_gesture_sequence_mcc",
    "any_gesture_temporal_iou",
    "any_best_lag_mcc",
    "any_onset_error_frames",
    "any_offset_error_frames",
    "any_predicted_segment_count",
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _metric_bundle(rows: list[dict[str, str]], gesture_ids: list[str] | None = None) -> dict[str, float]:
    """Small-row reference implementation retained for tests and local diagnostics."""
    if not rows:
        return {
            "exact_state_accuracy": float("nan"),
            "gesture_frame_end_to_end_accuracy": float("nan"),
            "any_gesture_f1": float("nan"),
            "macro_gesture_f1": float("nan"),
            "no_gesture_correct_rejection_rate": float("nan"),
            "no_hand_recall": float("nan"),
        }
    exact = sum(_operational_state_match(str(r["gt_state"]), str(r["predicted_state"])) for r in rows)
    gesture_rows = [r for r in rows if str(r["gt_state"]).startswith("GESTURE_")]
    gesture_correct = sum(str(r["predicted_state"]) == str(r["gt_state"]) for r in gesture_rows)
    gt_any = np.asarray([str(r["gt_state"]).startswith("GESTURE_") for r in rows], dtype=bool)
    pr_any = np.asarray([str(r["predicted_state"]).startswith("GESTURE_") for r in rows], dtype=bool)
    tp = int(np.sum(gt_any & pr_any))
    fp = int(np.sum((~gt_any) & pr_any))
    fn = int(np.sum(gt_any & (~pr_any)))
    tn = len(rows) - tp - fp - fn
    any_f1 = float(_binary_metrics(tp, fp, tn, fn)["f1"])
    gestures = list(gesture_ids) if gesture_ids is not None else sorted(
        {str(r["gt_state"]).removeprefix("GESTURE_") for r in rows if str(r["gt_state"]).startswith("GESTURE_")}
    )
    per_f1: list[float] = []
    for gesture in gestures:
        gt = np.asarray([str(r["gt_state"]) == f"GESTURE_{gesture}" for r in rows], dtype=bool)
        pr = np.asarray([str(r["predicted_state"]) == f"GESTURE_{gesture}" for r in rows], dtype=bool)
        ctp = int(np.sum(gt & pr))
        cfp = int(np.sum((~gt) & pr))
        cfn = int(np.sum(gt & (~pr)))
        ctn = len(rows) - ctp - cfp - cfn
        per_f1.append(float(_binary_metrics(ctp, cfp, ctn, cfn)["f1"]))
    ng = [r for r in rows if str(r["gt_state"]) == NO_GESTURE]
    nh = [r for r in rows if str(r["gt_state"]) == NO_HAND]
    return {
        "exact_state_accuracy": _safe_div(exact, len(rows)),
        "gesture_frame_end_to_end_accuracy": _safe_div(gesture_correct, len(gesture_rows)),
        "any_gesture_f1": any_f1,
        "macro_gesture_f1": float(np.mean(per_f1)) if per_f1 else float("nan"),
        "no_gesture_correct_rejection_rate": _safe_div(sum(str(r["predicted_state"]) == NO_GESTURE for r in ng), len(ng)),
        "no_hand_recall": _safe_div(sum(str(r["predicted_state"]) == MISSING_HAND for r in nh), len(nh)),
    }


# Compact count-vector layout used by the scalable bootstrap.
_BASE_FIELDS = (
    "evaluated",
    "exact_correct",
    "gesture_frames",
    "gesture_correct",
    "any_tp",
    "any_fp",
    "any_tn",
    "any_fn",
    "no_gesture_frames",
    "no_gesture_correct",
    "no_hand_frames",
    "no_hand_correct",
)
_BASE_INDEX = {name: i for i, name in enumerate(_BASE_FIELDS)}
_GESTURE_OFFSET = len(_BASE_FIELDS)
_VECTOR_SIZE = _GESTURE_OFFSET + 4 * len(GESTURE_ORDER)


def _gesture_offset(gesture_index: int) -> int:
    return _GESTURE_OFFSET + 4 * int(gesture_index)


def _cluster_vector_from_confusion(confusion: Counter[tuple[str, str]], base: np.ndarray) -> np.ndarray:
    vector = np.asarray(base, dtype=np.float64).copy()
    evaluated = float(vector[_BASE_INDEX["evaluated"]])
    for gi, gesture in enumerate(GESTURE_ORDER):
        state = f"GESTURE_{gesture}"
        tp = float(confusion[(state, state)])
        fp = float(sum(v for (gt, pred), v in confusion.items() if gt != state and pred == state))
        fn = float(sum(v for (gt, pred), v in confusion.items() if gt == state and pred != state))
        tn = evaluated - tp - fp - fn
        off = _gesture_offset(gi)
        vector[off : off + 4] = (tp, fp, tn, fn)
    return vector


def _stream_cluster_vectors(path: Path) -> dict[str, dict[tuple[str, str, str], np.ndarray]]:
    """Stream frame predictions into per-scenario/per-take sufficient statistics."""
    if not path.is_file():
        raise ValueError(f"Missing frame_predictions.csv: {path}")
    raw: dict[tuple[str, str, str], tuple[np.ndarray, Counter[tuple[str, str]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scenario", "fold_id", "take_id", "gt_state", "predicted_state"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid frame_predictions schema in {path}; required={sorted(required)}")
        for row in reader:
            scenario = str(row["scenario"])
            fold_id = str(row["fold_id"])
            take_id = str(row["take_id"])
            key = (scenario, fold_id, take_id)
            if key not in raw:
                raw[key] = (np.zeros(_VECTOR_SIZE, dtype=np.float64), Counter())
            base, confusion = raw[key]
            gt = str(row["gt_state"])
            pred = str(row["predicted_state"])
            base[_BASE_INDEX["evaluated"]] += 1.0
            base[_BASE_INDEX["exact_correct"]] += float(_operational_state_match(gt, pred))
            gt_gesture = gt.startswith("GESTURE_")
            pred_gesture = pred.startswith("GESTURE_")
            if gt_gesture:
                base[_BASE_INDEX["gesture_frames"]] += 1.0
                base[_BASE_INDEX["gesture_correct"]] += float(gt == pred)
            if gt_gesture and pred_gesture:
                base[_BASE_INDEX["any_tp"]] += 1.0
            elif (not gt_gesture) and pred_gesture:
                base[_BASE_INDEX["any_fp"]] += 1.0
            elif gt_gesture and (not pred_gesture):
                base[_BASE_INDEX["any_fn"]] += 1.0
            else:
                base[_BASE_INDEX["any_tn"]] += 1.0
            if gt == NO_GESTURE:
                base[_BASE_INDEX["no_gesture_frames"]] += 1.0
                base[_BASE_INDEX["no_gesture_correct"]] += float(pred == NO_GESTURE)
            elif gt == NO_HAND:
                base[_BASE_INDEX["no_hand_frames"]] += 1.0
                base[_BASE_INDEX["no_hand_correct"]] += float(pred == MISSING_HAND)
            confusion[(gt, pred)] += 1
    grouped: dict[str, dict[tuple[str, str, str], np.ndarray]] = defaultdict(dict)
    for key, (base, confusion) in raw.items():
        grouped[key[0]][key] = _cluster_vector_from_confusion(confusion, base)
    if not grouped:
        raise ValueError(f"No frame predictions found in {path}")
    return dict(grouped)


def _metrics_from_count_vector(vector: np.ndarray) -> dict[str, float]:
    v = np.asarray(vector, dtype=np.float64)
    anym = _binary_metrics(
        int(round(v[_BASE_INDEX["any_tp"]])),
        int(round(v[_BASE_INDEX["any_fp"]])),
        int(round(v[_BASE_INDEX["any_tn"]])),
        int(round(v[_BASE_INDEX["any_fn"]])),
    )
    per_f1: list[float] = []
    for gi in range(len(GESTURE_ORDER)):
        off = _gesture_offset(gi)
        tp, fp, tn, fn = (int(round(x)) for x in v[off : off + 4])
        per_f1.append(float(_binary_metrics(tp, fp, tn, fn)["f1"]))
    return {
        "exact_state_accuracy": _safe_div(v[_BASE_INDEX["exact_correct"]], v[_BASE_INDEX["evaluated"]]),
        "gesture_frame_end_to_end_accuracy": _safe_div(v[_BASE_INDEX["gesture_correct"]], v[_BASE_INDEX["gesture_frames"]]),
        "any_gesture_f1": float(anym["f1"]),
        "macro_gesture_f1": float(np.mean(per_f1)),
        "no_gesture_correct_rejection_rate": _safe_div(v[_BASE_INDEX["no_gesture_correct"]], v[_BASE_INDEX["no_gesture_frames"]]),
        "no_hand_recall": _safe_div(v[_BASE_INDEX["no_hand_correct"]], v[_BASE_INDEX["no_hand_frames"]]),
    }


def _ci(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _bootstrap_cluster_matrix(
    matrix: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    batch_draws: int = 25,
) -> dict[str, list[float]]:
    metrics = tuple(_metrics_from_count_vector(np.sum(matrix, axis=0)).keys())
    draws: dict[str, list[float]] = {metric: [] for metric in metrics}
    n = int(matrix.shape[0])
    for start in range(0, int(samples), int(batch_draws)):
        b = min(int(batch_draws), int(samples) - start)
        indices = rng.integers(0, n, size=(b, n), endpoint=False)
        summed = np.sum(matrix[indices], axis=1)
        for row in summed:
            bundle = _metrics_from_count_vector(row)
            for metric, value in bundle.items():
                draws[metric].append(float(value))
    return draws


def _bootstrap_one_mode(
    clusters_by_scenario: dict[str, dict[tuple[str, str, str], np.ndarray]],
    *,
    samples: int,
    rng: np.random.Generator,
    mode: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario, clusters in sorted(clusters_by_scenario.items()):
        ids = sorted(clusters)
        matrix = np.vstack([clusters[cid] for cid in ids])
        observed = _metrics_from_count_vector(np.sum(matrix, axis=0))
        draws = _bootstrap_cluster_matrix(matrix, samples=samples, rng=rng)
        for metric in observed:
            mean, median, low, high = _ci(draws[metric])
            output.append({
                "mode": mode,
                "scenario": scenario,
                "metric": metric,
                "cluster_unit": "take_id",
                "cluster_count": len(ids),
                "bootstrap_samples": int(samples),
                "observed": float(observed[metric]),
                "bootstrap_mean": mean,
                "bootstrap_median": median,
                "ci95_low": low,
                "ci95_high": high,
            })
    return output


def _paired_delta(
    image_clusters: dict[str, dict[tuple[str, str, str], np.ndarray]],
    video_clusters: dict[str, dict[tuple[str, str, str], np.ndarray]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    metrics = ("exact_state_accuracy", "gesture_frame_end_to_end_accuracy", "any_gesture_f1", "macro_gesture_f1")
    output: list[dict[str, Any]] = []
    for scenario in sorted(set(image_clusters) & set(video_clusters)):
        common = sorted(set(image_clusters[scenario]) & set(video_clusters[scenario]))
        if not common:
            continue
        im = np.vstack([image_clusters[scenario][key] for key in common])
        vm = np.vstack([video_clusters[scenario][key] for key in common])
        # Full E2E reports should contain the same GT frame universe. Fail rather than
        # silently pair differently-sized clusters.
        if not np.array_equal(im[:, _BASE_INDEX["evaluated"]], vm[:, _BASE_INDEX["evaluated"]]):
            raise ValueError(f"IMAGE/VIDEO paired cluster frame-count mismatch in scenario {scenario}")
        obs_i = _metrics_from_count_vector(np.sum(im, axis=0))
        obs_v = _metrics_from_count_vector(np.sum(vm, axis=0))
        draws: dict[str, list[float]] = {m: [] for m in metrics}
        n = len(common)
        for start in range(0, int(samples), 25):
            b = min(25, int(samples) - start)
            indices = rng.integers(0, n, size=(b, n), endpoint=False)
            isum = np.sum(im[indices], axis=1)
            vsum = np.sum(vm[indices], axis=1)
            for a, bvec in zip(isum, vsum):
                bi = _metrics_from_count_vector(a)
                bv = _metrics_from_count_vector(bvec)
                for metric in metrics:
                    if math.isfinite(float(bi[metric])) and math.isfinite(float(bv[metric])):
                        draws[metric].append(float(bv[metric]) - float(bi[metric]))
        for metric in metrics:
            mean, median, low, high = _ci(draws[metric])
            output.append({
                "scenario": scenario,
                "metric": metric,
                "cluster_unit": "take_id",
                "cluster_count": len(common),
                "paired_frame_count": int(np.sum(im[:, _BASE_INDEX["evaluated"]])),
                "bootstrap_samples": int(samples),
                "image_observed": float(obs_i[metric]),
                "video_observed": float(obs_v[metric]),
                "observed_delta_video_minus_image": float(obs_v[metric]) - float(obs_i[metric]),
                "bootstrap_delta_mean": mean,
                "bootstrap_delta_median": median,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "ci_excludes_zero": int(math.isfinite(low) and math.isfinite(high) and (low > 0.0 or high < 0.0)),
            })
    return output


def _mean_finite(rows: list[dict[str, str]], metric: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(metric, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _bootstrap_temporal(rows: list[dict[str, str]], *, samples: int, rng: np.random.Generator, mode: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scenarios = sorted({str(r.get("scenario", "")) for r in rows if str(r.get("scenario", ""))})
    for scenario in scenarios:
        scenario_rows = [r for r in rows if str(r.get("scenario")) == scenario]
        by_take = {str(r.get("take_id", "")): r for r in scenario_rows if str(r.get("take_id", ""))}
        ids = sorted(by_take)
        if not ids:
            continue
        values = np.asarray(
            [[float(by_take[cid].get(metric, "nan")) for metric in TEMPORAL_METRICS] for cid in ids],
            dtype=np.float64,
        )
        observed = np.nanmean(values, axis=0)
        draws: list[np.ndarray] = []
        n = len(ids)
        for start in range(0, int(samples), 100):
            b = min(100, int(samples) - start)
            indices = rng.integers(0, n, size=(b, n), endpoint=False)
            with np.errstate(invalid="ignore"):
                draws.append(np.nanmean(values[indices], axis=1))
        draw_matrix = np.vstack(draws)
        for mi, metric in enumerate(TEMPORAL_METRICS):
            mean, median, low, high = _ci(draw_matrix[:, mi].tolist())
            output.append({
                "mode": mode,
                "scenario": scenario,
                "metric": metric,
                "aggregation": "mean_across_take_level_temporal_metrics",
                "cluster_unit": "take_id",
                "cluster_count": len(ids),
                "bootstrap_samples": int(samples),
                "observed": float(observed[mi]),
                "bootstrap_mean": mean,
                "bootstrap_median": median,
                "ci95_low": low,
                "ci95_high": high,
            })
    return output


def _paired_temporal_delta(
    image_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    key_fields = ("scenario", "fold_id", "take_id")
    image = {tuple(str(r.get(k, "")) for k in key_fields): r for r in image_rows}
    video = {tuple(str(r.get(k, "")) for k in key_fields): r for r in video_rows}
    by_scenario: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for key in sorted(set(image) & set(video)):
        by_scenario[key[0]].append((image[key], video[key]))
    output: list[dict[str, Any]] = []
    for scenario, pairs in sorted(by_scenario.items()):
        if not pairs:
            continue
        ivals = np.asarray([[float(p[0].get(m, "nan")) for m in TEMPORAL_METRICS] for p in pairs], dtype=np.float64)
        vvals = np.asarray([[float(p[1].get(m, "nan")) for m in TEMPORAL_METRICS] for p in pairs], dtype=np.float64)
        obs_i = np.nanmean(ivals, axis=0)
        obs_v = np.nanmean(vvals, axis=0)
        draws: list[np.ndarray] = []
        n = len(pairs)
        for start in range(0, int(samples), 100):
            b = min(100, int(samples) - start)
            indices = rng.integers(0, n, size=(b, n), endpoint=False)
            with np.errstate(invalid="ignore"):
                bi = np.nanmean(ivals[indices], axis=1)
                bv = np.nanmean(vvals[indices], axis=1)
            draws.append(bv - bi)
        draw_matrix = np.vstack(draws)
        for mi, metric in enumerate(TEMPORAL_METRICS):
            mean, median, low, high = _ci(draw_matrix[:, mi].tolist())
            output.append({
                "scenario": scenario,
                "metric": metric,
                "aggregation": "mean_across_paired_take_level_temporal_metrics",
                "cluster_unit": "take_id",
                "paired_take_count": len(pairs),
                "bootstrap_samples": int(samples),
                "image_observed": float(obs_i[mi]),
                "video_observed": float(obs_v[mi]),
                "observed_delta_video_minus_image": float(obs_v[mi] - obs_i[mi]),
                "bootstrap_delta_mean": mean,
                "bootstrap_delta_median": median,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "ci_excludes_zero": int(math.isfinite(low) and math.isfinite(high) and (low > 0.0 or high < 0.0)),
            })
    return output


def bootstrap_frozen_end_to_end(
    *,
    image_report: str | Path,
    video_report: str | Path,
    output_dir: str | Path,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> Path:
    if int(samples) < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    image_root = Path(image_report)
    video_root = Path(video_report)
    image_clusters = _stream_cluster_vectors(image_root / "frame_predictions.csv")
    video_clusters = _stream_cluster_vectors(video_root / "frame_predictions.csv")
    image_temporal = read_csv_rows(image_root / "video_temporal_summary.csv")
    video_temporal = read_csv_rows(video_root / "video_temporal_summary.csv")
    if not image_temporal or not video_temporal:
        raise ValueError("Both IMAGE and VIDEO end-to-end reports must contain video_temporal_summary.csv")

    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    rng = np.random.default_rng(int(seed))

    ci_rows = _bootstrap_one_mode(image_clusters, samples=samples, rng=rng, mode="IMAGE")
    ci_rows.extend(_bootstrap_one_mode(video_clusters, samples=samples, rng=rng, mode="VIDEO"))
    paired = _paired_delta(image_clusters, video_clusters, samples=samples, rng=rng)
    temporal_ci = _bootstrap_temporal(image_temporal, samples=samples, rng=rng, mode="IMAGE")
    temporal_ci.extend(_bootstrap_temporal(video_temporal, samples=samples, rng=rng, mode="VIDEO"))
    paired_temporal = _paired_temporal_delta(image_temporal, video_temporal, samples=samples, rng=rng)

    _write_csv(out / "scenario_clustered_bootstrap_ci.csv", ci_rows)
    _write_csv(out / "paired_image_video_delta_ci.csv", paired)
    _write_csv(out / "temporal_clustered_bootstrap_ci.csv", temporal_ci)
    _write_csv(out / "paired_image_video_temporal_delta_ci.csv", paired_temporal)
    write_workbook(
        out / "frozen_end_to_end_bootstrap.xlsx",
        {
            "scenario_ci": (list(ci_rows[0]) if ci_rows else [], ci_rows),
            "paired_mode_delta": (list(paired[0]) if paired else [], paired),
            "temporal_ci": (list(temporal_ci[0]) if temporal_ci else [], temporal_ci),
            "paired_temporal_delta": (list(paired_temporal[0]) if paired_temporal else [], paired_temporal),
        },
    )
    report = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster_unit": "take_id",
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "image_report": str(image_root),
        "video_report": str(video_root),
        "notes": [
            "Whole takes/videos are resampled as clusters; frames inside a take are never treated as independent bootstrap units.",
            "Frame-level CSVs are streamed once into per-take sufficient statistics, so large 5000-draw publication runs remain tractable.",
            "Paired IMAGE-VIDEO deltas use matched scenario/fold/take clusters and the same resampled take indices in both modes.",
            "Temporal confidence intervals operate on per-take temporal metrics; paired temporal deltas use take IDs present in both modes.",
        ],
    }
    write_json(out / "frozen_end_to_end_bootstrap_report.json", report)
    return out / "frozen_end_to_end_bootstrap_report.json"
