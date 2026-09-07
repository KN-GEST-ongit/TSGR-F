"""Describe annotated test-video ground truth after applying the hand-presence reference."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tsgr.evaluation.ground_truth import (
    NO_GESTURE,
    NO_HAND,
    UNANNOTATED,
    evaluation_state,
    hand_reference_bool,
    is_gesture_label,
)
from tsgr.dataset.contract import discover_public_test_frames, discover_public_test_takes
from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows, write_workbook
from tsgr.utils.serialization import write_json
from tsgr.visualization.matplotlib_backend import configure_headless_backend

configure_headless_backend()
import matplotlib.pyplot as plt


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_q1": math.nan,
            f"{prefix}_q3": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_q1": float(np.quantile(array, 0.25)),
        f"{prefix}_q3": float(np.quantile(array, 0.75)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _reference_index(reference_rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {(str(row["take_id"]), int(row["frame_index"])): row for row in reference_rows}


def _aggregate_take_rows(take_rows: list[dict[str, Any]], group_fields: tuple[str, ...], level: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in take_rows:
        groups[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        total = sum(int(row["total_frames"]) for row in rows)
        gesture = sum(int(row["gesture_frames"]) for row in rows)
        no_gesture = sum(int(row["no_gesture_frames"]) for row in rows)
        no_hand = sum(int(row["no_hand_frames"]) for row in rows)
        unannotated = sum(int(row["unannotated_frames"]) for row in rows)
        evaluated = gesture + no_gesture + no_hand
        payload: dict[str, Any] = {
            "aggregation_level": level,
            **{field: value for field, value in zip(group_fields, key)},
            "take_count": len(rows),
            "total_frames": total,
            "evaluated_frames": evaluated,
            "gesture_frames": gesture,
            "no_gesture_frames": no_gesture,
            "no_hand_frames": no_hand,
            "unannotated_frames": unannotated,
            "gesture_fraction": _safe_ratio(gesture, evaluated),
            "no_gesture_fraction": _safe_ratio(no_gesture, evaluated),
            "no_hand_fraction": _safe_ratio(no_hand, evaluated),
        }
        payload.update(_stats((float(row["gesture_duration_frames"]) for row in rows if int(row["gesture_frames"]) > 0), "gesture_duration_frames"))
        payload.update(_stats((float(row["gesture_duration_s"]) for row in rows if int(row["gesture_frames"]) > 0), "gesture_duration_s"))
        output.append(payload)
    return output


def _plot_gesture_composition(take_rows: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"gesture": 0, "no_gesture": 0, "no_hand": 0})
    for row in take_rows:
        gesture = str(row["gesture_id"])
        grouped[gesture]["gesture"] += int(row["gesture_frames"])
        grouped[gesture]["no_gesture"] += int(row["no_gesture_frames"])
        grouped[gesture]["no_hand"] += int(row["no_hand_frames"])
    labels = sorted(grouped)
    x = np.arange(len(labels))
    gesture_values = np.asarray([grouped[label]["gesture"] for label in labels], dtype=float)
    no_gesture_values = np.asarray([grouped[label]["no_gesture"] for label in labels], dtype=float)
    no_hand_values = np.asarray([grouped[label]["no_hand"] for label in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, gesture_values, label="GESTURE")
    ax.bar(x, no_gesture_values, bottom=gesture_values, label="NO_GESTURE")
    ax.bar(x, no_hand_values, bottom=gesture_values + no_gesture_values, label="NO_HAND")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Frames")
    ax.set_title("Test-dataset frame composition by target gesture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_subject_background_no_hand(take_rows: list[dict[str, Any]], output: Path) -> None:
    subjects = sorted({str(row["public_subject_id"]) for row in take_rows})
    backgrounds = sorted({str(row["background"]) for row in take_rows})
    matrix = np.zeros((len(subjects), len(backgrounds)), dtype=np.float64)
    for i, subject in enumerate(subjects):
        for j, background in enumerate(backgrounds):
            rows = [row for row in take_rows if row["public_subject_id"] == subject and row["background"] == background]
            evaluated = sum(int(row["gesture_frames"]) + int(row["no_gesture_frames"]) + int(row["no_hand_frames"]) for row in rows)
            missing = sum(int(row["no_hand_frames"]) for row in rows)
            matrix[i, j] = _safe_ratio(missing, evaluated)
    fig, ax = plt.subplots(figsize=(7, max(4, len(subjects) * 0.7 + 2)))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.01, float(matrix.max(initial=0.0))))
    ax.set_xticks(range(len(backgrounds)), backgrounds)
    ax.set_yticks(range(len(subjects)), subjects)
    ax.set_title("NO_HAND fraction by subject and background")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{100.0 * matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="NO_HAND fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_duration_boxplot(take_rows: list[dict[str, Any]], output: Path) -> None:
    gestures = sorted({str(row["gesture_id"]) for row in take_rows if int(row["gesture_frames"]) > 0})
    values = [
        [float(row["gesture_duration_frames"]) for row in take_rows if row["gesture_id"] == gesture and int(row["gesture_frames"]) > 0]
        for gesture in gestures
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, tick_labels=gestures, showfliers=True)
    ax.set_ylabel("Annotated gesture duration [frames]")
    ax.set_title("Distribution of annotated gesture duration")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def analyze_test_ground_truth(
    dataset_root: str | Path,
    *,
    hand_presence_reference_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    root = Path(dataset_root)
    frames = discover_public_test_frames(root)
    takes = discover_public_test_takes(root, read_video_fps=True)
    reference_dir = Path(hand_presence_reference_dir)
    reference_rows = read_csv_rows(reference_dir / "frame_hand_presence_reference.csv")
    if not frames or not takes:
        raise ValueError("No test frames/takes were found in the dataset contract.")
    if not reference_rows:
        raise ValueError("frame_hand_presence_reference.csv is required before ground-truth analysis.")
    take_metadata = {str(row["take_id"]): row for row in takes}
    reference = _reference_index(reference_rows)
    report_dir = Path(output_dir) if output_dir else root / "reports" / "test_ground_truth" / datetime.now().strftime("ground_truth_%Y%m%d_%H%M%S_%f")
    report_dir.mkdir(parents=True, exist_ok=False)

    frame_output: list[dict[str, Any]] = []
    by_take: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        take_id = str(row["take_id"])
        frame_index = int(row["frame_index"])
        true_label = str(row.get("true_label", UNANNOTATED)).upper()
        reference_row = reference.get((take_id, frame_index))
        hand_present = hand_reference_bool(reference_row)
        decision = evaluation_state(true_label, hand_present)
        output = {
            "take_id": take_id,
            "public_subject_id": row.get("public_subject_id", ""),
            "background": row.get("background", ""),
            "gesture_id": row.get("gesture_id", ""),
            "frame_index": frame_index,
            "frame_relative_path": row.get("relative_path", ""),
            "annotation_label": true_label,
            "hand_present_reference": "" if decision.hand_present is None else int(decision.hand_present),
            "evaluation_state": decision.evaluation_state,
            "evaluation_source": decision.source,
            "include_in_metrics": int(decision.evaluation_state != UNANNOTATED),
        }
        frame_output.append(output)
        by_take[take_id].append(output)
    frame_fields = list(frame_output[0]) if frame_output else []
    write_csv_rows(report_dir / "frame_ground_truth.csv", frame_output, frame_fields)

    take_output: list[dict[str, Any]] = []
    for take_id, rows in sorted(by_take.items()):
        metadata = take_metadata.get(take_id, {})
        target = str(metadata.get("gesture_id", rows[0].get("gesture_id", ""))).upper()
        states = [str(row["evaluation_state"]) for row in rows]
        gesture_frames = sum(state == f"GESTURE_{target}" for state in states)
        no_gesture_frames = sum(state == NO_GESTURE for state in states)
        no_hand_frames = sum(state == NO_HAND for state in states)
        unannotated_frames = sum(state == UNANNOTATED for state in states)
        fps = float(metadata.get("source_video_fps") or metadata.get("fps") or 0.0)
        if fps <= 0:
            fps = 30.0
        evaluated = gesture_frames + no_gesture_frames + no_hand_frames
        take_output.append(
            {
                "take_id": take_id,
                "public_subject_id": metadata.get("public_subject_id", rows[0].get("public_subject_id", "")),
                "background": str(metadata.get("background", rows[0].get("background", ""))).upper(),
                "gesture_id": target,
                "total_frames": len(rows),
                "evaluated_frames": evaluated,
                "gesture_frames": gesture_frames,
                "no_gesture_frames": no_gesture_frames,
                "no_hand_frames": no_hand_frames,
                "unannotated_frames": unannotated_frames,
                "gesture_fraction": _safe_ratio(gesture_frames, evaluated),
                "no_gesture_fraction": _safe_ratio(no_gesture_frames, evaluated),
                "no_hand_fraction": _safe_ratio(no_hand_frames, evaluated),
                "gesture_duration_frames": gesture_frames,
                "gesture_duration_s": gesture_frames / fps,
                "fps": fps,
            }
        )
    take_fields = list(take_output[0]) if take_output else []
    write_csv_rows(report_dir / "take_ground_truth_summary.csv", take_output, take_fields)

    aggregate_specs = [
        ((), "global"),
        (("public_subject_id",), "subject"),
        (("background",), "background"),
        (("gesture_id",), "gesture"),
        (("public_subject_id", "background"), "subject_background"),
        (("public_subject_id", "gesture_id"), "subject_gesture"),
        (("background", "gesture_id"), "background_gesture"),
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for fields, level in aggregate_specs:
        aggregate_rows.extend(_aggregate_take_rows(take_output, fields, level))
    aggregate_fields: list[str] = []
    for row in aggregate_rows:
        for key in row:
            if key not in aggregate_fields:
                aggregate_fields.append(key)
    write_csv_rows(report_dir / "ground_truth_aggregates.csv", aggregate_rows, aggregate_fields)

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_gesture_composition(take_output, plots_dir / "frame_composition_by_gesture.png")
    _plot_subject_background_no_hand(take_output, plots_dir / "no_hand_fraction_subject_background.png")
    _plot_duration_boxplot(take_output, plots_dir / "gesture_duration_boxplot.png")

    aggregate_by_level = defaultdict(list)
    for row in aggregate_rows:
        aggregate_by_level[str(row["aggregation_level"])].append(row)
    workbook_sheets: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "takes": (take_fields, take_output),
    }
    for level in ("global", "subject", "background", "gesture", "subject_background", "subject_gesture", "background_gesture"):
        rows = aggregate_by_level[level]
        if rows:
            fields: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            workbook_sheets[level] = (fields, rows)
    write_workbook(report_dir / "dataset_ground_truth_summary.xlsx", workbook_sheets)

    annotated_takes = sum(int(row["unannotated_frames"]) == 0 for row in take_output)
    write_json(
        report_dir / "ground_truth_report.json",
        {
            "schema_version": "tsgr_test_ground_truth_report_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": ".",
            "dataset_contract": "tsgr_public_dataset_v1",
            "hand_presence_reference_dir": reference_dir.name,
            "rule": "gesture annotation overrides hand presence; only NO_GESTURE frames use detector-derived hand presence",
            "source_annotation_hand_missing_frames_used": False,
            "take_count": len(take_output),
            "fully_annotated_take_count": annotated_takes,
            "frame_count": len(frame_output),
            "evaluated_frame_count": sum(int(row["include_in_metrics"]) for row in frame_output),
            "unannotated_frame_count": sum(not int(row["include_in_metrics"]) for row in frame_output),
        },
    )
    return report_dir
