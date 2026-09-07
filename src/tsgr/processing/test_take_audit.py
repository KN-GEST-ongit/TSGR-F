"""Parallel processing and exact MediaPipe reliability audit for canonical test takes."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tsgr.analysis.audit_utils import (
    copy_source_file,
    mediapipe_failure_reason,
    read_jsonl,
    save_review_pair,
)
from tsgr.dataset.contract import discover_public_test_frames, discover_public_test_takes
from tsgr.pipeline.frame_pipeline import FramePipeline
from tsgr.processing.image_folder_run import process_image_folder_run
from tsgr.reference_models.manifest import locate_processed_run
from tsgr.utils.serialization import write_json

_WORKER_PIPELINE: FramePipeline | None = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_MODEL_PATH: str | None = None
_WORKER_INITIALIZATION_S: float = 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)




def _portable_path_value(root: Path, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def _portable_report_rows(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for field in ("take_path", "source_video_path", "run_path", "relative_path", "source_copy", "review_image"):
            if field in item:
                item[field] = _portable_path_value(root, item.get(field, ""))
        # Acquisition filenames are not part of the released runtime contract.
        if "source_filename" in item:
            item["source_filename"] = ""
        output.append(item)
    return output

def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _take_directories(dataset_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    # Runtime metadata is reconstructed from the released annotations and hierarchy.
    for row in discover_public_test_takes(dataset_root, read_video_fps=True):
        take_dir = dataset_root / str(row["relative_path"])
        yield take_dir, dict(row)


def _summarize_run(
    run_dir: Path,
    frame_manifest: list[dict[str, str]],
    minimum_success_rate: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize MediaPipe success plus raw handedness observations for one take."""
    results = {int(row["frame_index"]): row for row in read_jsonl(run_dir / "frame_results.jsonl")}
    frame_failures: list[dict[str, Any]] = []
    handedness_events: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    successful = 0
    recovered = 0
    video_recovery_used = 0
    critical_left = 0
    raw_left_observed = 0
    ignored_left = 0
    for manifest_row in frame_manifest:
        frame_index = int(manifest_row["frame_index"])
        result = results.get(frame_index)
        status = "not_processed" if result is None else str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = mediapipe_failure_reason(result)
        diagnostics = dict((result or {}).get("detection_diagnostics") or {})
        recovered += int(bool(diagnostics.get("recovered", False)))
        video_recovery_used += int(bool(diagnostics.get("video_recovery_used", False)))
        critical_left += int(status == "critical_left_hand_detected")
        left_observed = bool(
            diagnostics.get("left_observed", False)
            or diagnostics.get("left_observed_in_any_attempt", False)
        )
        raw_left_observed += int(left_observed)
        policy_action = str(diagnostics.get("handedness_policy_action", ""))
        ignored = bool(left_observed and policy_action == "accept_landmarks_ignore_label")
        ignored_left += int(ignored)
        if left_observed:
            handedness_events.append(
                {
                    "frame_index": frame_index,
                    "filename": manifest_row.get("filename", ""),
                    "relative_path": manifest_row.get("relative_path", ""),
                    "source_filename": manifest_row.get("source_filename", ""),
                    "true_label": manifest_row.get("true_label", ""),
                    "status": status,
                    "handedness_policy": diagnostics.get("handedness_policy", ""),
                    "handedness_policy_action": policy_action,
                    "selected_raw_handedness": diagnostics.get("selected_raw_handedness", ""),
                    "selected_effective_handedness": diagnostics.get("selected_effective_handedness", ""),
                    "handedness_mismatch": int(bool(diagnostics.get("handedness_mismatch", False))),
                    "left_observed_in_any_attempt": int(bool(diagnostics.get("left_observed_in_any_attempt", False))),
                    "selected_attempt": diagnostics.get("selected_attempt", ""),
                    "recovered": int(bool(diagnostics.get("recovered", False))),
                    "video_recovery_used": int(bool(diagnostics.get("video_recovery_used", False))),
                    "detection_final_outcome": diagnostics.get("final_outcome", ""),
                    "effective_input_state": (
                        "NO_GESTURE_LEFT_HAND_REJECTED"
                        if policy_action == "reject_frame"
                        else "GESTURE_CLASSIFICATION_ALLOWED"
                    ),
                }
            )
        if reason:
            frame_failures.append(
                {
                    "frame_index": frame_index,
                    "filename": manifest_row.get("filename", ""),
                    "relative_path": manifest_row.get("relative_path", ""),
                    "source_filename": manifest_row.get("source_filename", ""),
                    "true_label": manifest_row.get("true_label", ""),
                    "status": status,
                    "failure_reason": reason,
                    "quality_warning_reasons": ";".join(
                        (result or {}).get("quality_metrics", {}).get("warning_reasons", []) or []
                    ),
                    "notes": ";".join(str(value) for value in ((result or {}).get("notes") or [])),
                    "detection_profile": diagnostics.get("profile", ""),
                    "selected_attempt": diagnostics.get("selected_attempt", ""),
                    "recovered": int(bool(diagnostics.get("recovered", False))),
                    "video_recovery_used": int(bool(diagnostics.get("video_recovery_used", False))),
                    "detection_final_outcome": diagnostics.get("final_outcome", ""),
                    "handedness_policy": diagnostics.get("handedness_policy", ""),
                    "handedness_policy_action": policy_action,
                    "effective_input_state": (
                        "NO_GESTURE_LEFT_HAND_REJECTED"
                        if policy_action == "reject_frame"
                        else "GESTURE_CLASSIFICATION_ALLOWED"
                    ),
                }
            )
        else:
            successful += 1
    total = len(frame_manifest)
    rate = successful / total if total else 0.0
    if total == 0:
        take_status = "empty_take"
    elif successful == 0:
        take_status = "complete_mediapipe_failure"
    elif rate < minimum_success_rate:
        take_status = "below_mediapipe_success_threshold"
    else:
        take_status = "pass"
    return (
        {
            "processed_frames": total,
            "successful_frames": successful,
            "failed_frames": total - successful,
            "mediapipe_success_rate": rate,
            "minimum_success_rate": minimum_success_rate,
            "take_mediapipe_status": take_status,
            "frame_status_counts": status_counts,
            "recovered_frames": recovered,
            "video_recovery_used_frames": video_recovery_used,
            "critical_left_hand_frames": critical_left,
            "raw_left_observed_frames": raw_left_observed,
            "ignored_left_hand_frames": ignored_left,
        },
        frame_failures,
        handedness_events,
    )


def resolve_worker_count(requested_workers: int, take_count: int) -> int:
    """Resolve ``0`` to a conservative automatic process count."""
    if requested_workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if take_count <= 0:
        return 0
    if requested_workers > 0:
        return min(requested_workers, take_count)
    cpu_count = os.cpu_count() or 1
    automatic = max(1, min(8, max(1, cpu_count // 2)))
    return min(automatic, take_count)


def _base_worker_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(config))
    merged.setdefault("temporal_filter", {})["active"] = "none"
    merged.setdefault("classification_input", {}).update(
        {
            "landmark_source": "raw",
            "scale": "wrist_middle_mcp",
            "export_all_feature_branches": True,
        }
    )
    return merged



def _test_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("mediapipe", {}).get("running_mode", "video")).strip().lower()
    if mode not in {"image", "video"}:
        raise ValueError("Test processing requires MediaPipe mode 'image' or 'video'.")
    return mode


def _handedness_policy(config: dict[str, Any]) -> str:
    policy = str(
        config.get("mediapipe", {}).get("handedness_policy", "reject_left")
    ).strip().lower()
    if policy not in {"reject_left", "ignore_handedness"}:
        raise ValueError("Unsupported handedness policy for test processing.")
    return policy


def _detection_profile(config: dict[str, Any]) -> str:
    mediapipe = config.get("mediapipe", {})
    if _test_mode(config) == "image":
        return str(mediapipe.get("image_detection", {}).get("active_profile", "high_recall"))
    return str(mediapipe.get("video_detection", {}).get("active_profile", "balanced"))


def _recovery_tag(config: dict[str, Any]) -> str:
    if _test_mode(config) == "image":
        return "image_retry_chain"
    enabled = bool(config.get("mediapipe", {}).get("video_recovery", {}).get("enabled", False))
    return "recovery_on" if enabled else "recovery_off"


def _test_variant_parts(config: dict[str, Any]) -> tuple[str, ...]:
    mode = _test_mode(config)
    profile = _detection_profile(config)
    policy = _handedness_policy(config)
    if mode == "image":
        return (mode, profile, policy)
    return (mode, profile, _recovery_tag(config), policy)


def _take_run_root(take_dir: Path, config: dict[str, Any]) -> Path:
    return take_dir.joinpath("runs", *_test_variant_parts(config))


def _report_root(dataset_root: Path, config: dict[str, Any]) -> Path:
    return dataset_root.joinpath("reports", "test_processing", *_test_variant_parts(config))


def _close_worker_pipeline() -> None:
    global _WORKER_PIPELINE
    if _WORKER_PIPELINE is not None:
        try:
            _WORKER_PIPELINE.close()
        finally:
            _WORKER_PIPELINE = None


def _worker_initialize(config: dict[str, Any], model_path: str) -> None:
    """Create one MediaPipe pipeline for the lifetime of one worker process."""
    global _WORKER_PIPELINE, _WORKER_CONFIG, _WORKER_MODEL_PATH, _WORKER_INITIALIZATION_S
    start = time.perf_counter()
    _WORKER_CONFIG = _base_worker_config(config)
    _WORKER_MODEL_PATH = str(model_path)
    _WORKER_PIPELINE = FramePipeline(_WORKER_CONFIG, _WORKER_MODEL_PATH)
    _WORKER_INITIALIZATION_S = time.perf_counter() - start


def _task_metadata(task: dict[str, Any]) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    take_dir = Path(task["take_dir"])
    metadata = dict(task["metadata"])
    frame_manifest = list(task["frame_manifest"])
    return take_dir, metadata, frame_manifest


def _processing_error_result(
    take_dir: Path,
    metadata: dict[str, Any],
    frame_manifest: list[dict[str, str]],
    error: BaseException,
    elapsed_s: float,
) -> dict[str, Any]:
    subject = str(metadata.get("public_subject_id", ""))
    background = str(metadata.get("background", "")).upper()
    gesture = str(metadata.get("gesture_id") or metadata.get("gesture") or "").upper()
    return {
        "take_row": {
            "take_id": str(metadata.get("take_id", take_dir.name)),
            "public_subject_id": subject,
            "background": background,
            "gesture_id": gesture,
            "take_path": str(take_dir.resolve()),
            "source_video_path": str((take_dir / "source_video.avi").resolve())
            if (take_dir / "source_video.avi").is_file()
            else "",
            "run_path": "",
            "processed_frames": 0,
            "successful_frames": 0,
            "failed_frames": len(frame_manifest),
            "mediapipe_success_rate": 0.0,
            "minimum_success_rate": task_minimum_success_rate(metadata),
            "take_mediapipe_status": "processing_error",
            "frame_status_counts": "{}",
            "test_mode": "",
            "detection_profile": "",
            "handedness_policy": "",
            "recovered_frames": 0,
            "video_recovery_used_frames": 0,
            "critical_left_hand_frames": 0,
            "error": f"{type(error).__name__}: {error}",
            "worker_pid": os.getpid(),
            "worker_model_initialization_s": _WORKER_INITIALIZATION_S,
            "elapsed_wall_time_s": elapsed_s,
            "frames_per_second": 0.0,
            "run_reused": 0,
        },
        "failure_rows": [],
        "handedness_rows": [],
    }


def task_minimum_success_rate(metadata: dict[str, Any]) -> float:
    return float(metadata.get("_minimum_success_rate", 0.95))


def _worker_process_take(task: dict[str, Any]) -> dict[str, Any]:
    """Process one complete test take inside a persistent worker process."""
    if _WORKER_PIPELINE is None or _WORKER_CONFIG is None or _WORKER_MODEL_PATH is None:
        raise RuntimeError("Worker pipeline was not initialized.")
    take_dir, metadata, frame_manifest = _task_metadata(task)
    start = time.perf_counter()
    minimum_success_rate = float(task["minimum_success_rate"])
    metadata["_minimum_success_rate"] = minimum_success_rate
    partial_dir: Path | None = None
    try:
        mirrored = _as_bool(metadata.get("input_is_mirrored"), True)
        merged = json.loads(json.dumps(_WORKER_CONFIG))
        merged.setdefault("camera", {})["input_is_mirrored"] = mirrored
        merged.setdefault("mediapipe", {})["invert_handedness_labels"] = mirrored
        _WORKER_PIPELINE.config = merged

        final_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f") + f"_p{os.getpid()}"
        partial_name = f".partial_{final_name}_{uuid.uuid4().hex[:8]}"
        variant_root = _take_run_root(take_dir, merged)
        variant_root.mkdir(parents=True, exist_ok=True)
        partial_dir = variant_root / partial_name
        partial_dir = process_image_folder_run(
            take_dir / "frames",
            output_root=variant_root,
            config=merged,
            model_path=_WORKER_MODEL_PATH,
            fps=float(metadata.get("source_video_fps") or task["fps"]),
            save_overlays=bool(task["save_overlays"]),
            save_crops=False,
            recursive=False,
            run_name=partial_name,
            pipeline=_WORKER_PIPELINE,
        )
        final_dir = partial_dir.parent / final_name
        partial_dir.replace(final_dir)
        run = final_dir
        signature = dict(task.get("inventory_signature") or test_take_inventory_signature(frame_manifest))
        mode = _test_mode(merged)
        signature.update(
            {
                "schema_version": "tsgr_test_take_processing_signature_v2",
                "test_mode": mode,
                "detection_profile": _detection_profile(merged),
                "handedness_policy": _handedness_policy(merged),
                "mediapipe_running_mode": mode,
                "mediapipe_image_profile": str(merged.get("mediapipe", {}).get("image_detection", {}).get("active_profile", "high_recall")),
                "mediapipe_video_profile": str(merged.get("mediapipe", {}).get("video_detection", {}).get("active_profile", "balanced")),
                "mediapipe_video_recovery_enabled": bool(merged.get("mediapipe", {}).get("video_recovery", {}).get("enabled", False)),
                "mediapipe_video_recovery_after_consecutive_failures": int(merged.get("mediapipe", {}).get("video_recovery", {}).get("after_consecutive_failures", 1)),
                "strict_right_hand_only": _handedness_policy(merged) == "reject_left",
                "effective_fps": float(metadata.get("source_video_fps") or task["fps"]),
            }
        )
        write_json(final_dir / "test_take_processing_signature.json", signature)
        summary, failures, handedness_events = _summarize_run(run, frame_manifest, minimum_success_rate)
        elapsed = time.perf_counter() - start
        take_id = str(metadata.get("take_id", take_dir.name))
        take_row = {
            "take_id": take_id,
            "public_subject_id": str(metadata.get("public_subject_id", "")),
            "background": str(metadata.get("background", "")).upper(),
            "gesture_id": str(metadata.get("gesture_id") or metadata.get("gesture") or "").upper(),
            "take_path": str(take_dir.resolve()),
            "source_video_path": str((take_dir / "source_video.avi").resolve())
            if (take_dir / "source_video.avi").is_file()
            else "",
            "run_path": str(run.resolve()),
            **{key: value for key, value in summary.items() if key != "frame_status_counts"},
            "frame_status_counts": json.dumps(summary["frame_status_counts"], sort_keys=True),
            "test_mode": _test_mode(merged),
            "detection_profile": _detection_profile(merged),
            "handedness_policy": _handedness_policy(merged),
            "error": "",
            "worker_pid": os.getpid(),
            "worker_model_initialization_s": _WORKER_INITIALIZATION_S,
            "elapsed_wall_time_s": elapsed,
            "frames_per_second": summary["processed_frames"] / elapsed if elapsed > 0 else 0.0,
            "run_reused": 0,
        }
        failure_rows = [
            {
                "take_id": take_id,
                "public_subject_id": take_row["public_subject_id"],
                "background": take_row["background"],
                "gesture_id": take_row["gesture_id"],
                "take_path": str(take_dir.resolve()),
                "run_path": str(run.resolve()),
                **failure,
            }
            for failure in failures
        ]
        handedness_rows = [
            {
                "take_id": take_id,
                "public_subject_id": take_row["public_subject_id"],
                "background": take_row["background"],
                "gesture_id": take_row["gesture_id"],
                "take_path": str(take_dir.resolve()),
                "run_path": str(run.resolve()),
                **event,
            }
            for event in handedness_events
        ]
        return {
            "take_row": take_row,
            "failure_rows": failure_rows,
            "handedness_rows": handedness_rows,
        }
    except BaseException as error:
        elapsed = time.perf_counter() - start
        if partial_dir is not None and partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        return _processing_error_result(take_dir, metadata, frame_manifest, error, elapsed)



def _worker_process_chunk(
    config: dict[str, Any],
    model_path: str,
    tasks: list[dict[str, Any]],
    progress: bool = False,
    worker_slot: int = 0,
) -> dict[str, Any]:
    """Process a deterministic chunk and close MediaPipe before worker shutdown.

    MediaPipe Tasks owns an internal Python executor used by ``close()``.  Calling
    close from an ``atexit`` callback is too late on Python 3.12 because that
    executor may already be shut down.  The chunk worker therefore owns the
    complete detector lifetime and closes it in ``finally`` while the interpreter
    is fully operational.
    """
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    _worker_initialize(config, model_path)
    close_error = ""
    try:
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            result = _worker_process_take(task)
            results.append(result)
            if progress:
                row = result["take_row"]
                print(
                    f"[worker {worker_slot} {index}/{total}] "
                    f"{row['take_id']}: {row['take_mediapipe_status']} "
                    f"({float(row['elapsed_wall_time_s']):.2f}s, pid={row['worker_pid']})",
                    flush=True,
                )
    finally:
        try:
            _close_worker_pipeline()
        except BaseException as error:  # pragma: no cover - native shutdown safeguard
            close_error = f"{type(error).__name__}: {error}"
    return {
        "worker_slot": int(worker_slot),
        "worker_pid": os.getpid(),
        "results": results,
        "close_error": close_error,
        "elapsed_wall_time_s": time.perf_counter() - start,
    }


def partition_test_tasks(tasks: list[dict[str, Any]], worker_count: int) -> list[list[dict[str, Any]]]:
    """Partition whole takes by canonical frame count using deterministic LPT balancing."""
    if worker_count <= 0:
        return []
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    ordered = sorted(
        tasks,
        key=lambda task: (
            -int(task.get("frame_count", len(task.get("frame_manifest") or []))),
            str((task.get("metadata") or {}).get("public_subject_id", "")),
            str((task.get("metadata") or {}).get("background", "")),
            str((task.get("metadata") or {}).get("gesture_id", "")),
            str((task.get("metadata") or {}).get("take_id", "")),
        ),
    )
    for task in ordered:
        slot = min(range(worker_count), key=lambda index: (loads[index], index))
        chunks[slot].append(task)
        loads[slot] += max(1, int(task.get("frame_count", len(task.get("frame_manifest") or []))))
    return [chunk for chunk in chunks if chunk]


# Compatibility alias retained for existing callers.
_partition_tasks = partition_test_tasks


def test_take_inventory_signature(frame_manifest: list[dict[str, str]]) -> dict[str, Any]:
    """Return a stable signature of canonical frame content, excluding annotations."""
    normalized = sorted(
        (
            int(row.get("frame_index", 0)),
            str(row.get("filename", "")),
            str(row.get("sha256", "")),
        )
        for row in frame_manifest
    )
    digest = hashlib.sha256()
    for frame_index, filename, sha256 in normalized:
        digest.update(str(frame_index).encode("ascii"))
        digest.update(b"\0")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii", errors="ignore"))
        digest.update(b"\n")
    return {
        "schema_version": "tsgr_test_take_inventory_v1",
        "frame_count": len(normalized),
        "sha256": digest.hexdigest(),
    }


def _fast_locate_test_run(take_dir: Path, config: dict[str, Any]) -> Path | None:
    """Locate the newest completed run for exactly one test-processing variant."""
    runs_dir = _take_run_root(take_dir, config)
    candidates: list[Path] = []
    if runs_dir.is_dir():
        for candidate in runs_dir.iterdir():
            if not candidate.is_dir() or candidate.name.startswith(".partial_"):
                continue
            if (candidate / "features" / "feature_arrays.npz").is_file() or (candidate / "feature_arrays.npz").is_file():
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _reused_take_result(
    take_dir: Path,
    metadata: dict[str, Any],
    frame_manifest: list[dict[str, str]],
    run: Path,
    minimum_success_rate: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    summary, failures, handedness_events = _summarize_run(run, frame_manifest, minimum_success_rate)
    elapsed = time.perf_counter() - start
    take_id = str(metadata.get("take_id", take_dir.name))
    take_row = {
        "take_id": take_id,
        "public_subject_id": str(metadata.get("public_subject_id", "")),
        "background": str(metadata.get("background", "")).upper(),
        "gesture_id": str(metadata.get("gesture_id") or metadata.get("gesture") or "").upper(),
        "take_path": str(take_dir.resolve()),
        "source_video_path": str((take_dir / "source_video.avi").resolve())
        if (take_dir / "source_video.avi").is_file()
        else "",
        "run_path": str(run.resolve()),
        **{key: value for key, value in summary.items() if key != "frame_status_counts"},
        "frame_status_counts": json.dumps(summary["frame_status_counts"], sort_keys=True),
        "test_mode": str(json.loads((run / "run_summary.json").read_text(encoding="utf-8")).get("mediapipe_running_mode", "")),
        "detection_profile": (
            str(json.loads((run / "run_summary.json").read_text(encoding="utf-8")).get("mediapipe_image_profile", ""))
            if str(json.loads((run / "run_summary.json").read_text(encoding="utf-8")).get("mediapipe_running_mode", "")) == "image"
            else str(json.loads((run / "run_summary.json").read_text(encoding="utf-8")).get("mediapipe_video_profile", ""))
        ),
        "handedness_policy": str(json.loads((run / "run_summary.json").read_text(encoding="utf-8")).get("handedness_policy", "reject_left")),
        "error": "",
        "worker_pid": "",
        "worker_model_initialization_s": "",
        "elapsed_wall_time_s": elapsed,
        "frames_per_second": summary["processed_frames"] / elapsed if elapsed > 0 else 0.0,
        "run_reused": 1,
    }
    failure_rows = [
        {
            "take_id": take_id,
            "public_subject_id": take_row["public_subject_id"],
            "background": take_row["background"],
            "gesture_id": take_row["gesture_id"],
            "take_path": str(take_dir.resolve()),
            "run_path": str(run.resolve()),
            **failure,
        }
        for failure in failures
    ]
    handedness_rows = [
        {
            "take_id": take_id,
            "public_subject_id": take_row["public_subject_id"],
            "background": take_row["background"],
            "gesture_id": take_row["gesture_id"],
            "take_path": str(take_dir.resolve()),
            "run_path": str(run.resolve()),
            **event,
        }
        for event in handedness_events
    ]
    return {
        "take_row": take_row,
        "failure_rows": failure_rows,
        "handedness_rows": handedness_rows,
    }


def _materialize_failure_reviews(
    root: Path,
    report_dir: Path,
    failure_rows: list[dict[str, Any]],
) -> None:
    result_cache: dict[str, dict[int, dict[str, Any]]] = {}
    for row in failure_rows:
        run_path = str(row.get("run_path", ""))
        if run_path and run_path not in result_cache:
            result_cache[run_path] = {
                int(item["frame_index"]): item
                for item in read_jsonl(Path(run_path) / "frame_results.jsonl")
            }
        take_dir = Path(str(row["take_path"]))
        source = (
            root / str(row["relative_path"])
            if row.get("relative_path")
            else take_dir / "frames" / str(row["filename"])
        )
        base = (
            report_dir
            / "mediapipe_failures"
            / str(row["public_subject_id"])
            / str(row["background"])
            / str(row["gesture_id"])
            / take_dir.name
        )
        original_suffix = source.suffix.lower() or ".jpg"
        original = base / "originals" / f"frame_{int(row['frame_index']):06d}{original_suffix}"
        if source.is_file():
            row["source_copy"] = str(copy_source_file(source, original).resolve())
        result = result_cache.get(run_path, {}).get(int(row["frame_index"]))
        destination = base / "reviews" / f"frame_{int(row['frame_index']):06d}.png"
        saved = save_review_pair(
            source,
            destination,
            result=result,
            header_lines=[
                f"{row['public_subject_id']}/{row['background']}/{row['gesture_id']}/{take_dir.name}",
                f"frame={row['frame_index']} reason={row['failure_reason']}",
            ],
        )
        row["review_image"] = str(saved.resolve()) if saved else ""


def _selected_tasks(
    root: Path,
    subjects: set[str] | None,
    backgrounds: set[str] | None,
    gestures: set[str] | None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for take_dir, metadata in _take_directories(root):
        subject = str(metadata.get("public_subject_id", ""))
        background = str(metadata.get("background", "")).upper()
        gesture = str(metadata.get("gesture_id") or metadata.get("gesture") or "").upper()
        if subjects and subject not in subjects:
            continue
        if backgrounds and background not in backgrounds:
            continue
        if gestures and gesture not in gestures:
            continue
        frame_manifest = _read_csv(take_dir / "frames_manifest.csv")
        if not frame_manifest and (root / "annotations.csv").is_file():
            frame_manifest = [
                dict(row)
                for row in discover_public_test_frames(
                    root, take_ids={str(metadata.get("take_id", ""))}
                )
            ]
        tasks.append(
            {
                "take_dir": str(take_dir.resolve()),
                "metadata": metadata,
                "frame_manifest": frame_manifest,
                "frame_count": len(frame_manifest),
                "inventory_signature": test_take_inventory_signature(frame_manifest),
            }
        )
    return tasks



def _run_matches_test_config(
    run_dir: Path,
    config: dict[str, Any],
    *,
    inventory_signature: dict[str, Any] | None = None,
    effective_fps: float | None = None,
) -> bool:
    """Return whether a cached run exactly matches the requested test variant."""
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    mediapipe = config.get("mediapipe", {})
    mode = _test_mode(config)
    policy = _handedness_policy(config)
    if str(payload.get("mediapipe_running_mode", "")).strip().lower() != mode:
        return False
    if str(payload.get("handedness_policy", "reject_left")).strip().lower() != policy:
        return False
    if mode == "image":
        profile_matches = (
            str(payload.get("mediapipe_image_profile", ""))
            == str(mediapipe.get("image_detection", {}).get("active_profile", "high_recall"))
        )
    else:
        profile_matches = (
            str(payload.get("mediapipe_video_profile", ""))
            == str(mediapipe.get("video_detection", {}).get("active_profile", "balanced"))
            and bool(payload.get("mediapipe_video_recovery_enabled", False))
            == bool(mediapipe.get("video_recovery", {}).get("enabled", False))
        )
    if not profile_matches:
        return False
    if inventory_signature is None:
        return True
    signature_path = run_dir / "test_take_processing_signature.json"
    if not signature_path.is_file():
        return False
    try:
        stored = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        str(stored.get("schema_version", "")) == "tsgr_test_take_processing_signature_v2"
        and int(stored.get("frame_count", -1)) == int(inventory_signature.get("frame_count", -2))
        and str(stored.get("sha256", "")) == str(inventory_signature.get("sha256", ""))
        and str(stored.get("test_mode", "")) == mode
        and str(stored.get("detection_profile", "")) == _detection_profile(config)
        and str(stored.get("handedness_policy", "")) == policy
        and bool(stored.get("mediapipe_video_recovery_enabled", False))
        == bool(mediapipe.get("video_recovery", {}).get("enabled", False))
        and int(stored.get("mediapipe_video_recovery_after_consecutive_failures", 1))
        == int(mediapipe.get("video_recovery", {}).get("after_consecutive_failures", 1))
        and (effective_fps is None or abs(float(stored.get("effective_fps", -1.0)) - float(effective_fps)) < 1e-9)
    )



def _run_matches_video_config(
    run_dir: Path,
    config: dict[str, Any],
    *,
    inventory_signature: dict[str, Any] | None = None,
    effective_fps: float | None = None,
) -> bool:
    """Return a compatible processing signature for existing cached runs."""
    legacy = json.loads(json.dumps(config))
    mediapipe = legacy.setdefault("mediapipe", {})
    mediapipe.setdefault("running_mode", "video")
    mediapipe.setdefault("handedness_policy", "reject_left")
    summary_path = run_dir / "run_summary.json"
    signature_path = run_dir / "test_take_processing_signature.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # Use the strict variant-aware matcher for current processing signatures.
    if str(summary.get("handedness_policy", "")):
        return _run_matches_test_config(
            run_dir,
            legacy,
            inventory_signature=inventory_signature,
            effective_fps=effective_fps,
        )
    if str(summary.get("mediapipe_running_mode", "")) != "video":
        return False
    if str(summary.get("mediapipe_video_profile", "")) != str(
        mediapipe.get("video_detection", {}).get("active_profile", "balanced")
    ):
        return False
    if bool(summary.get("mediapipe_video_recovery_enabled", False)) != bool(
        mediapipe.get("video_recovery", {}).get("enabled", False)
    ):
        return False
    if int(summary.get("mediapipe_video_recovery_after_consecutive_failures", 1)) != int(
        mediapipe.get("video_recovery", {}).get("after_consecutive_failures", 1)
    ):
        return False
    if not bool(summary.get("strict_right_hand_only", False)):
        return False
    if inventory_signature is None:
        return True
    if not signature_path.is_file():
        return False
    try:
        stored = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(stored.get("frame_count", -1)) == int(inventory_signature.get("frame_count", -2))
        and str(stored.get("sha256", "")) == str(inventory_signature.get("sha256", ""))
        and (effective_fps is None or abs(float(stored.get("effective_fps", -1.0)) - float(effective_fps)) < 1e-9)
    )


def process_experiment_test_takes(
    dataset_root: str | Path,
    *,
    config: dict[str, Any],
    model_path: str | Path,
    fps: float = 30.0,
    force: bool = False,
    resume: bool = True,
    subjects: set[str] | None = None,
    backgrounds: set[str] | None = None,
    gestures: set[str] | None = None,
    minimum_success_rate: float = 0.95,
    save_overlays: bool = False,
    copy_failed_frames: bool = True,
    workers: int = 0,
    progress: bool = True,
    fail_fast: bool = False,
    report_name: str | None = None,
) -> Path:
    """Process canonical test sequences in worker processes and create exact audits."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not 0 < fps <= 30:
        raise ValueError("fps must be in the interval (0, 30].")
    if not 0 <= minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must be in [0, 1].")
    if workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")

    # Explicit IMAGE/VIDEO and handedness variants are physically separated so
    # multiple ablation runs can coexist without overwriting or cross-reusing cache.
    mode = _test_mode(config)
    policy = _handedness_policy(config)
    effective_report_name = report_name or datetime.now().strftime("processing_%Y%m%d_%H%M%S_%f")
    report_dir = _report_root(root, config) / effective_report_name
    report_dir.mkdir(parents=True, exist_ok=False)
    selected = _selected_tasks(root, subjects, backgrounds, gestures)
    resolved_workers = resolve_worker_count(workers, len(selected))
    take_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    handedness_rows: list[dict[str, Any]] = []
    tasks_to_process: list[dict[str, Any]] = []
    reused_count = 0

    reusable: list[tuple[Path, dict[str, Any], list[dict[str, str]], Path]] = []
    for task in selected:
        take_dir, metadata, frame_manifest = _task_metadata(task)
        existing = _fast_locate_test_run(take_dir, config)
        effective_fps = float(metadata.get("source_video_fps") or fps)
        if (
            resume
            and not force
            and existing is not None
            and _run_matches_test_config(
                existing,
                config,
                inventory_signature=dict(task.get("inventory_signature") or {}),
                effective_fps=effective_fps,
            )
        ):
            reusable.append((take_dir, metadata, frame_manifest, existing))
        else:
            task.update(
                {
                    "fps": fps,
                    "minimum_success_rate": minimum_success_rate,
                    "save_overlays": save_overlays,
                }
            )
            tasks_to_process.append(task)

    reuse_audit_start = time.perf_counter()
    if reusable:
        audit_workers = resolve_worker_count(workers, len(reusable))
        if audit_workers <= 1:
            reuse_results = [
                _reused_take_result(take_dir, metadata, frame_manifest, existing, minimum_success_rate)
                for take_dir, metadata, frame_manifest, existing in reusable
            ]
        else:
            with ThreadPoolExecutor(max_workers=audit_workers) as executor:
                futures = [
                    executor.submit(
                        _reused_take_result, take_dir, metadata, frame_manifest, existing, minimum_success_rate
                    )
                    for take_dir, metadata, frame_manifest, existing in reusable
                ]
                reuse_results = [future.result() for future in futures]
        for result in reuse_results:
            take_rows.append(result["take_row"])
            failure_rows.extend(result["failure_rows"])
            handedness_rows.extend(result.get("handedness_rows", []))
        reused_count = len(reuse_results)
    reuse_audit_elapsed = time.perf_counter() - reuse_audit_start

    processing_start = time.perf_counter()
    abort_error: RuntimeError | None = None
    completed = 0
    if tasks_to_process:
        active_workers = resolve_worker_count(workers, len(tasks_to_process))
        if active_workers == 1:
            _worker_initialize(config, str(Path(model_path).resolve()))
            try:
                for task in tasks_to_process:
                    result = _worker_process_take(task)
                    take_rows.append(result["take_row"])
                    failure_rows.extend(result["failure_rows"])
                    handedness_rows.extend(result.get("handedness_rows", []))
                    completed += 1
                    if progress:
                        print(
                            f"[{completed}/{len(tasks_to_process)}] "
                            f"{result['take_row']['take_id']}: "
                            f"{result['take_row']['take_mediapipe_status']}"
                        )
                    if fail_fast and result["take_row"]["take_mediapipe_status"] == "processing_error":
                        abort_error = RuntimeError(
                            f"Test processing stopped after {result['take_row']['take_id']}: "
                            f"{result['take_row'].get('error', 'processing error')}"
                        )
                        break
            finally:
                _close_worker_pipeline()
        else:
            context = mp.get_context("spawn")
            chunks = partition_test_tasks(tasks_to_process, active_workers)
            # One future owns one complete MediaPipe lifetime.  This guarantees
            # explicit detector.close() before the child interpreter starts its
            # atexit/futures shutdown sequence.
            with ProcessPoolExecutor(
                max_workers=len(chunks),
                mp_context=context,
            ) as executor:
                futures: dict[Future[dict[str, Any]], int] = {
                    executor.submit(
                        _worker_process_chunk,
                        config,
                        str(Path(model_path).resolve()),
                        chunk,
                        progress,
                        slot,
                    ): slot
                    for slot, chunk in enumerate(chunks, start=1)
                }
                for future in as_completed(futures):
                    chunk_result = future.result()
                    close_error = str(chunk_result.get("close_error") or "")
                    if close_error:
                        raise RuntimeError(
                            f"Worker {chunk_result.get('worker_pid')} failed to close MediaPipe cleanly: "
                            f"{close_error}"
                        )
                    for result in chunk_result["results"]:
                        take_rows.append(result["take_row"])
                        failure_rows.extend(result["failure_rows"])
                        handedness_rows.extend(result.get("handedness_rows", []))
                        completed += 1
                        if fail_fast and result["take_row"]["take_mediapipe_status"] == "processing_error":
                            abort_error = RuntimeError(
                                f"Test processing stopped after {result['take_row']['take_id']}: "
                                f"{result['take_row'].get('error', 'processing error')}"
                            )
                            for pending in futures:
                                pending.cancel()
                            break
                    if abort_error is not None:
                        break
    processing_elapsed = time.perf_counter() - processing_start

    take_rows.sort(
        key=lambda row: (
            str(row.get("public_subject_id", "")),
            str(row.get("background", "")),
            str(row.get("gesture_id", "")),
            str(row.get("take_id", "")),
        )
    )
    failure_rows.sort(
        key=lambda row: (
            str(row.get("public_subject_id", "")),
            str(row.get("background", "")),
            str(row.get("gesture_id", "")),
            str(row.get("take_id", "")),
            int(row.get("frame_index", -1)),
        )
    )
    handedness_rows.sort(
        key=lambda row: (
            str(row.get("public_subject_id", "")),
            str(row.get("background", "")),
            str(row.get("gesture_id", "")),
            str(row.get("take_id", "")),
            int(row.get("frame_index", -1)),
        )
    )
    if copy_failed_frames:
        _materialize_failure_reviews(root, report_dir, failure_rows)

    take_fields = [
        "take_id",
        "public_subject_id",
        "background",
        "gesture_id",
        "take_path",
        "source_video_path",
        "run_path",
        "processed_frames",
        "successful_frames",
        "failed_frames",
        "mediapipe_success_rate",
        "minimum_success_rate",
        "take_mediapipe_status",
        "frame_status_counts",
        "test_mode",
        "detection_profile",
        "handedness_policy",
        "recovered_frames",
        "video_recovery_used_frames",
        "critical_left_hand_frames",
        "raw_left_observed_frames",
        "ignored_left_hand_frames",
        "error",
        "worker_pid",
        "worker_model_initialization_s",
        "elapsed_wall_time_s",
        "frames_per_second",
        "run_reused",
    ]
    frame_fields = [
        "take_id",
        "public_subject_id",
        "background",
        "gesture_id",
        "take_path",
        "run_path",
        "frame_index",
        "filename",
        "relative_path",
        "source_filename",
        "true_label",
        "status",
        "failure_reason",
        "quality_warning_reasons",
        "notes",
        "detection_profile",
        "selected_attempt",
        "recovered",
        "video_recovery_used",
        "detection_final_outcome",
        "handedness_policy",
        "handedness_policy_action",
        "effective_input_state",
        "source_copy",
        "review_image",
    ]
    portable_take_rows = _portable_report_rows(root, take_rows)
    portable_failure_rows = _portable_report_rows(root, failure_rows)
    portable_handedness_rows = _portable_report_rows(root, handedness_rows)
    _write_csv(report_dir / "test_take_mediapipe_status.csv", portable_take_rows, take_fields)
    _write_csv(report_dir / "failed_test_frames.csv", portable_failure_rows, frame_fields)
    handedness_fields = [
        "take_id", "public_subject_id", "background", "gesture_id", "take_path", "run_path",
        "frame_index", "filename", "relative_path", "source_filename", "true_label", "status",
        "handedness_policy", "handedness_policy_action", "selected_raw_handedness",
        "selected_effective_handedness", "handedness_mismatch", "left_observed_in_any_attempt",
        "selected_attempt", "recovered", "video_recovery_used", "detection_final_outcome",
        "effective_input_state",
    ]
    _write_csv(report_dir / "handedness_events.csv", portable_handedness_rows, handedness_fields)
    _write_csv(
        report_dir / "failed_test_takes.csv",
        [row for row in portable_take_rows if row.get("take_mediapipe_status") != "pass"],
        take_fields,
    )

    worker_groups: dict[str, list[dict[str, Any]]] = {}
    for row in take_rows:
        pid = str(row.get("worker_pid", ""))
        if not pid:
            continue
        worker_groups.setdefault(pid, []).append(row)
    worker_rows: list[dict[str, Any]] = []
    for pid, rows in sorted(worker_groups.items(), key=lambda item: int(item[0])):
        elapsed_sum = sum(float(row.get("elapsed_wall_time_s") or 0.0) for row in rows)
        frame_sum = sum(int(row.get("processed_frames") or 0) for row in rows)
        worker_rows.append(
            {
                "worker_pid": pid,
                "take_count": len(rows),
                "frame_count": frame_sum,
                "sum_take_elapsed_s": elapsed_sum,
                "aggregate_frames_per_second": frame_sum / elapsed_sum if elapsed_sum > 0 else 0.0,
                "model_initialization_s": max(
                    float(row.get("worker_model_initialization_s") or 0.0) for row in rows
                ),
            }
        )
    _write_csv(
        report_dir / "worker_statistics.csv",
        worker_rows,
        [
            "worker_pid",
            "take_count",
            "frame_count",
            "sum_take_elapsed_s",
            "aggregate_frames_per_second",
            "model_initialization_s",
        ],
    )

    processed_rows = [row for row in take_rows if not int(row.get("run_reused") or 0)]
    serial_equivalent_s = sum(float(row.get("elapsed_wall_time_s") or 0.0) for row in processed_rows)
    total_processed_frames = sum(int(row.get("processed_frames") or 0) for row in processed_rows)
    processing_errors = sum(row.get("take_mediapipe_status") == "processing_error" for row in take_rows)
    write_json(
        report_dir / "processing_report.json",
        {
            "dataset_root": ".",
            "dataset_contract": "tsgr_public_dataset_v1",
            "fps_fallback": fps,
            "minimum_success_rate": minimum_success_rate,
            "requested_workers": workers,
            "resolved_workers": resolved_workers,
            "worker_processes_used": len(worker_rows),
            "multiprocessing_start_method": "spawn" if len(worker_rows) > 1 else "serial",
            "resume_enabled": resume,
            "force_enabled": force,
            "processed_takes": len(processed_rows),
            "reused_takes": reused_count,
            "processing_errors": processing_errors,
            "take_count": len(take_rows),
            "failed_frame_count": len(failure_rows),
            "reuse_audit_elapsed_wall_time_s": reuse_audit_elapsed,
            "processing_elapsed_wall_time_s": processing_elapsed,
            "partition_strategy": "largest_processing_time_by_canonical_frame_count",
            "serial_equivalent_sum_take_time_s": serial_equivalent_s,
            "estimated_parallel_speedup": serial_equivalent_s / processing_elapsed
            if processing_elapsed > 0
            else 0.0,
            "processed_frame_count": total_processed_frames,
            "aggregate_processing_frames_per_second": total_processed_frames / processing_elapsed
            if processing_elapsed > 0
            else 0.0,
            "test_mode": mode,
            "detection_profile": _detection_profile(config),
            "handedness_policy": policy,
            "report_variant_parts": list(_test_variant_parts(config)),
            "mediapipe_running_mode": mode,
            "mediapipe_image_profile": str(config.get("mediapipe", {}).get("image_detection", {}).get("active_profile", "high_recall")),
            "mediapipe_video_profile": str(config.get("mediapipe", {}).get("video_detection", {}).get("active_profile", "balanced")),
            "mediapipe_video_recovery_enabled": bool(config.get("mediapipe", {}).get("video_recovery", {}).get("enabled", False)),
            "mediapipe_video_recovery_after_consecutive_failures": int(config.get("mediapipe", {}).get("video_recovery", {}).get("after_consecutive_failures", 1)),
            "strict_right_hand_only": policy == "reject_left",
            "recovered_frame_count": sum(int(row.get("recovered_frames") or 0) for row in take_rows),
            "video_recovery_used_frame_count": sum(int(row.get("video_recovery_used_frames") or 0) for row in take_rows),
            "critical_left_hand_frame_count": sum(int(row.get("critical_left_hand_frames") or 0) for row in take_rows),
            "raw_left_observed_frame_count": sum(int(row.get("raw_left_observed_frames") or 0) for row in take_rows),
            "ignored_left_hand_frame_count": sum(int(row.get("ignored_left_hand_frames") or 0) for row in take_rows),
            "handedness_event_row_count": len(handedness_rows),
            "take_status_counts": {
                status: sum(row.get("take_mediapipe_status") == status for row in take_rows)
                for status in sorted({str(row.get("take_mediapipe_status")) for row in take_rows})
            },
        },
    )
    if abort_error is not None:
        raise abort_error
    return report_dir
