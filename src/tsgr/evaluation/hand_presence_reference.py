"""Build a detector-derived reference mask for hand presence in test-video frames."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from tsgr.detection.mediapipe_hand_landmarker import MediaPipeHandLandmarker
from tsgr.dataset.contract import discover_public_test_frames, discover_public_test_takes
from tsgr.evaluation.ground_truth import NO_GESTURE, UNANNOTATED, is_gesture_label
from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows
from tsgr.utils.serialization import write_json

_WORKER_DETECTOR: MediaPipeHandLandmarker | None = None


def resolve_hand_reference_worker_count(requested_workers: int, take_count: int) -> int:
    if requested_workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if take_count <= 0:
        return 0
    if requested_workers > 0:
        return min(requested_workers, take_count)
    cpu_count = os.cpu_count() or 1
    return min(max(1, min(8, cpu_count)), take_count)


def _normalized_relative(root: Path, relative: str) -> Path:
    return root / Path(str(relative).replace("\\", "/"))


def _reference_mediapipe_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config.get("mediapipe", {}))
    result["running_mode"] = "image"
    result["handedness_policy"] = "ignore_handedness"
    result["strict_right_hand_only"] = False
    result.setdefault("image_detection", {})["active_profile"] = "high_recall"
    result.setdefault("video_recovery", {})["enabled"] = False
    return result


def _worker_init(mediapipe_config: dict[str, Any], model_path: str) -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = MediaPipeHandLandmarker(mediapipe_config, model_path)


def _worker_close() -> None:
    global _WORKER_DETECTOR
    if _WORKER_DETECTOR is not None:
        _WORKER_DETECTOR.close()
        _WORKER_DETECTOR = None


def _process_take(root_text: str, take_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    if _WORKER_DETECTOR is None:
        raise RuntimeError("Hand-presence worker was not initialized.")
    root = Path(root_text)
    output: list[dict[str, Any]] = []
    started = time.perf_counter()
    detector_frames = 0
    detector_elapsed_ms = 0.0
    for row in sorted(rows, key=lambda item: int(item["frame_index"])):
        frame_index = int(row["frame_index"])
        label = str(row.get("true_label", UNANNOTATED)).strip().upper()
        base = {
            "take_id": take_id,
            "public_subject_id": row.get("public_subject_id", ""),
            "background": row.get("background", ""),
            "gesture_id": row.get("gesture_id", ""),
            "frame_index": frame_index,
            "frame_relative_path": row.get("relative_path", ""),
            "annotation_label": label,
        }
        if is_gesture_label(label):
            output.append(
                {
                    **base,
                    "hand_present_reference": 1,
                    "reference_state": "HAND_PRESENT",
                    "reference_source": "gesture_annotation_override",
                    "mediapipe_executed": 0,
                    "selected_attempt": "",
                    "attempt_count": 0,
                    "detection_final_outcome": "annotation_override",
                    "detection_elapsed_ms": 0.0,
                    "error": "",
                }
            )
            continue
        if label == UNANNOTATED:
            output.append(
                {
                    **base,
                    "hand_present_reference": "",
                    "reference_state": UNANNOTATED,
                    "reference_source": "unannotated_excluded",
                    "mediapipe_executed": 0,
                    "selected_attempt": "",
                    "attempt_count": 0,
                    "detection_final_outcome": "",
                    "detection_elapsed_ms": 0.0,
                    "error": "",
                }
            )
            continue
        if label != NO_GESTURE:
            raise ValueError(f"Unsupported frame annotation label {label!r} in take {take_id}.")
        image_path = _normalized_relative(root, row.get("relative_path", ""))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            output.append(
                {
                    **base,
                    "hand_present_reference": "",
                    "reference_state": "REFERENCE_ERROR",
                    "reference_source": "mediapipe_high_recall_4pass",
                    "mediapipe_executed": 0,
                    "selected_attempt": "",
                    "attempt_count": 0,
                    "detection_final_outcome": "",
                    "detection_elapsed_ms": 0.0,
                    "error": f"Cannot read canonical frame: {image_path}",
                }
            )
            continue
        detection_started = time.perf_counter()
        candidates = _WORKER_DETECTOR.detect(image, frame_index)
        elapsed_ms = (time.perf_counter() - detection_started) * 1000.0
        detector_frames += 1
        detector_elapsed_ms += elapsed_ms
        diagnostics = dict(_WORKER_DETECTOR.last_diagnostics or {})
        present = bool(candidates)
        output.append(
            {
                **base,
                "hand_present_reference": int(present),
                "reference_state": "HAND_PRESENT" if present else "HAND_MISSING",
                "reference_source": "mediapipe_high_recall_4pass",
                "mediapipe_executed": 1,
                "selected_attempt": diagnostics.get("selected_attempt", ""),
                "attempt_count": int(diagnostics.get("attempt_count", 0) or 0),
                "detection_final_outcome": diagnostics.get("final_outcome", ""),
                "detection_elapsed_ms": elapsed_ms,
                "error": "",
            }
        )
    return {
        "take_id": take_id,
        "rows": output,
        "detector_frames": detector_frames,
        "detector_elapsed_ms": detector_elapsed_ms,
        "elapsed_wall_time_s": time.perf_counter() - started,
        "worker_pid": os.getpid(),
    }


def _process_chunk(
    root_text: str,
    mediapipe_config: dict[str, Any],
    model_path: str,
    chunk: list[tuple[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    _worker_init(mediapipe_config, model_path)
    results: list[dict[str, Any]] = []
    close_error = ""
    try:
        for take_id, rows in chunk:
            results.append(_process_take(root_text, take_id, rows))
    finally:
        try:
            _worker_close()
        except BaseException as error:  # pragma: no cover - native close guard
            close_error = f"{type(error).__name__}: {error}"
    return {"results": results, "close_error": close_error, "worker_pid": os.getpid()}


def _partition(items: list[tuple[str, list[dict[str, str]]]], count: int) -> list[list[tuple[str, list[dict[str, str]]]]]:
    chunks: list[list[tuple[str, list[dict[str, str]]]]] = [[] for _ in range(max(1, count))]
    loads = [0] * len(chunks)
    ordered = sorted(items, key=lambda item: -len(item[1]))
    for item in ordered:
        index = min(range(len(chunks)), key=lambda value: loads[value])
        chunks[index].append(item)
        loads[index] += len(item[1])
    return [chunk for chunk in chunks if chunk]


def _update_canonical_annotations(root: Path, reference_rows: list[dict[str, Any]], take_rows: list[dict[str, str]], report_dir: Path) -> int:
    missing_by_take: dict[str, list[int]] = defaultdict(list)
    for row in reference_rows:
        if row.get("reference_state") == "HAND_MISSING" and str(row.get("annotation_label")) == NO_GESTURE:
            missing_by_take[str(row["take_id"])].append(int(row["frame_index"]))
    take_map = {str(row["take_id"]): row for row in take_rows}
    updated = 0
    for take_id, take_row in take_map.items():
        relative = str(take_row.get("relative_path", ""))
        take_dir = _normalized_relative(root, relative) if relative else None
        if take_dir is None or not take_dir.is_dir():
            continue
        annotation_path = take_dir / "annotation.json"
        if not annotation_path.is_file():
            continue
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not bool(payload.get("source_annotation_valid", False)):
            continue
        payload["hand_missing_frames"] = sorted(missing_by_take.get(take_id, []))
        payload["use_hand_missing_frames_in_evaluation"] = True
        payload["hand_missing_frames_source"] = "mediapipe_image_high_recall_4pass_reference"
        payload["hand_presence_reference_report"] = str(report_dir.resolve())
        payload["hand_presence_reference_rule"] = (
            "gesture_interval_frames_are_hand_present_by_annotation; only NO_GESTURE frames are detector-evaluated"
        )
        annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1
    return updated


def build_hand_presence_reference(
    dataset_root: str | Path,
    *,
    config: dict[str, Any],
    model_path: str | Path,
    output_dir: str | Path | None = None,
    workers: int = 0,
    progress: bool = True,
    update_annotations: bool = True,
) -> Path:
    """Build the reference hand-presence mask once for the complete test database."""
    root = Path(dataset_root)
    frame_rows = discover_public_test_frames(root)
    take_rows = discover_public_test_takes(root, read_video_fps=False)
    # Hand presence is a derived evaluation artifact and must not modify the
    # released annotation contract.
    update_annotations = False
    if not frame_rows or not take_rows:
        raise ValueError("No test frames/takes were found in the dataset contract.")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row["take_id"])].append(row)
    items = sorted(grouped.items())
    resolved_workers = resolve_hand_reference_worker_count(workers, len(items))
    report_dir = Path(output_dir) if output_dir else root / "reports" / "hand_presence_reference" / datetime.now().strftime("reference_%Y%m%d_%H%M%S_%f")
    report_dir.mkdir(parents=True, exist_ok=False)
    mediapipe_config = _reference_mediapipe_config(config)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if resolved_workers <= 1:
        _worker_init(mediapipe_config, str(Path(model_path).resolve()))
        try:
            for index, (take_id, rows) in enumerate(items, start=1):
                results.append(_process_take(str(root.resolve()), take_id, rows))
                if progress and (index % 25 == 0 or index == len(items)):
                    print(f"[hand-reference {index}/{len(items)}] takes", flush=True)
        finally:
            _worker_close()
    else:
        chunks = _partition(items, resolved_workers)
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    str(root.resolve()),
                    mediapipe_config,
                    str(Path(model_path).resolve()),
                    chunk,
                )
                for chunk in chunks
            ]
            completed = 0
            for future in as_completed(futures):
                payload = future.result()
                if payload.get("close_error"):
                    raise RuntimeError(f"MediaPipe worker close failed: {payload['close_error']}")
                for result in payload["results"]:
                    results.append(result)
                    completed += 1
                    if progress and (completed % 25 == 0 or completed == len(items)):
                        print(f"[hand-reference {completed}/{len(items)}] takes", flush=True)
    results.sort(key=lambda item: item["take_id"])
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (str(row["take_id"]), int(row["frame_index"])))
    fields = [
        "take_id", "public_subject_id", "background", "gesture_id", "frame_index",
        "frame_relative_path", "annotation_label",
        "hand_present_reference", "reference_state", "reference_source",
        "mediapipe_executed", "selected_attempt", "attempt_count",
        "detection_final_outcome", "detection_elapsed_ms", "error",
    ]
    write_csv_rows(report_dir / "frame_hand_presence_reference.csv", rows, fields)
    take_summary: list[dict[str, Any]] = []
    for result in results:
        local_rows = result["rows"]
        states = Counter(str(row["reference_state"]) for row in local_rows)
        take_summary.append(
            {
                "take_id": result["take_id"],
                "frame_count": len(local_rows),
                "annotation_override_frames": states.get("HAND_PRESENT", 0) - sum(
                    int(row.get("mediapipe_executed", 0)) for row in local_rows if row.get("reference_state") == "HAND_PRESENT"
                ),
                "detector_evaluated_frames": result["detector_frames"],
                "detector_hand_present_frames": sum(
                    int(row.get("hand_present_reference") == 1) for row in local_rows if int(row.get("mediapipe_executed", 0))
                ),
                "detector_hand_missing_frames": states.get("HAND_MISSING", 0),
                "unannotated_frames": states.get(UNANNOTATED, 0),
                "reference_error_frames": states.get("REFERENCE_ERROR", 0),
                "detector_elapsed_ms": result["detector_elapsed_ms"],
                "elapsed_wall_time_s": result["elapsed_wall_time_s"],
                "worker_pid": result["worker_pid"],
            }
        )
    take_fields = list(take_summary[0]) if take_summary else []
    write_csv_rows(report_dir / "take_hand_presence_summary.csv", take_summary, take_fields)
    updated = _update_canonical_annotations(root, rows, take_rows, report_dir) if update_annotations else 0
    errors = [row for row in rows if row.get("reference_state") == "REFERENCE_ERROR"]
    hash_payload = hashlib.sha256(
        json.dumps(
            [(row["take_id"], row["frame_index"], row["reference_state"]) for row in rows],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    elapsed = time.perf_counter() - started
    write_json(
        report_dir / "hand_presence_reference.json",
        {
            "schema_version": "tsgr_hand_presence_reference_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": ".",
            "dataset_contract": "tsgr_public_dataset_v1",
            "reference_method": "MediaPipe IMAGE high_recall ordered 4-pass chain on NO_GESTURE frames",
            "gesture_interval_rule": "annotated gesture frames are HAND_PRESENT without detector execution",
            "source_annotation_hand_missing_frames_used": False,
            "canonical_annotations_updated": bool(update_annotations),
            "canonical_annotation_count_updated": updated,
            "take_count": len(results),
            "frame_count": len(rows),
            "mediapipe_evaluated_frame_count": sum(result["detector_frames"] for result in results),
            "reference_error_frame_count": len(errors),
            "requested_workers": workers,
            "resolved_workers": resolved_workers,
            "elapsed_wall_time_s": elapsed,
            "reference_sha256": hash_payload,
            "image_detection_profile": "high_recall",
            "handedness_policy": "ignore_handedness",
            "image_detection_pass_names": [
                str(item.get("name", "")) for item in mediapipe_config.get("image_detection", {}).get("profiles", {}).get("high_recall", {}).get("passes", [])
            ],
        },
    )
    if errors:
        write_csv_rows(report_dir / "reference_errors.csv", errors, fields)
        raise RuntimeError(
            f"Hand-presence reference contains {len(errors)} unreadable/error frames. See {report_dir / 'reference_errors.csv'}"
        )
    return report_dir
