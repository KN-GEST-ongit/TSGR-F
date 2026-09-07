"""Parallel processing of canonical training-photo cells with MediaPipe IMAGE mode."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tsgr.analysis.training_detection_audit import audit_training_cell, write_training_detection_audit
from tsgr.dataset.contract import discover_public_training_images
from tsgr.pipeline.frame_pipeline import FramePipeline
from tsgr.processing.image_folder_run import process_image_folder_run
from tsgr.reference_models.manifest import locate_processed_run
from tsgr.utils.serialization import write_json

_WORKER_PIPELINE: FramePipeline | None = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_MODEL_PATH: str | None = None
_WORKER_INITIALIZATION_S: float = 0.0


def resolve_training_worker_count(requested_workers: int, cell_count: int) -> int:
    """Resolve ``0`` to a bounded automatic process count for IMAGE inference."""
    if requested_workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if cell_count <= 0:
        return 0
    if requested_workers > 0:
        return min(requested_workers, cell_count)
    cpu_count = os.cpu_count() or 1
    # Each MediaPipe/XNNPACK worker may use native threads internally.  Half of the
    # logical CPUs, capped at eight, gives useful parallelism without unbounded
    # oversubscription on typical workstation CPUs.
    automatic = max(1, min(8, max(1, cpu_count // 2)))
    return min(automatic, cell_count)


def partition_training_tasks(
    tasks: list[dict[str, Any]], worker_count: int
) -> list[list[dict[str, Any]]]:
    """Create deterministic approximately balanced worker chunks."""
    if worker_count <= 0:
        return []
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    # Largest cells first gives sensible balance if a future dataset contains
    # unequal photo counts.  Stable tie-breaking preserves reproducibility.
    ordered = sorted(
        tasks,
        key=lambda task: (
            -int(task.get("image_count", 0)),
            str(task.get("subject", "")),
            str(task.get("background", "")),
            str(task.get("gesture", "")),
        ),
    )
    for task in ordered:
        slot = min(range(worker_count), key=lambda index: (loads[index], index))
        chunks[slot].append(task)
        loads[slot] += max(1, int(task.get("image_count", 0)))
    return [chunk for chunk in chunks if chunk]


def training_cell_inventory_signature(cell_dir: Path) -> dict[str, Any]:
    """Return a stable content signature for one canonical training cell."""
    manifest_path = cell_dir / "images_manifest.csv"
    rows: list[dict[str, str]] = []
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        images_dir = cell_dir / "images" if (cell_dir / "images").is_dir() else cell_dir
        for ordinal, path in enumerate(sorted(
            (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}),
            key=lambda value: value.name,
        ), start=1):
            rows.append({
                "image_id": f"{cell_dir.parts[-3]}_{cell_dir.parts[-2]}_{cell_dir.parts[-1]}_{ordinal:06d}",
                "filename": path.name,
                "sha256": __import__("tsgr.utils.serialization", fromlist=["sha256_file"]).sha256_file(path),
            })
    normalized = sorted(
        (
            str(row.get("image_id", "")),
            str(row.get("filename", "")),
            str(row.get("sha256", "")),
        )
        for row in rows
    )
    digest = hashlib.sha256()
    for image_id, filename, sha256 in normalized:
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii", errors="ignore"))
        digest.update(b"\n")
    return {
        "schema_version": "tsgr_training_cell_inventory_v1",
        "image_count": len(normalized),
        "sha256": digest.hexdigest(),
        "image_ids": [item[0] for item in normalized],
    }


def _run_matches_image_profile(run_dir: Path, profile: str, cell_dir: Path | None = None) -> bool:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    profile_matches = (
        str(payload.get("mediapipe_running_mode", "")) == "image"
        and str(payload.get("mediapipe_image_profile", "")) == profile
        and bool(payload.get("strict_right_hand_only", False))
    )
    if not profile_matches:
        return False
    if cell_dir is None:
        return True
    signature_path = run_dir / "training_cell_processing_signature.json"
    if not signature_path.is_file():
        return False
    try:
        stored = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    current = training_cell_inventory_signature(cell_dir)
    return (
        str(stored.get("schema_version", "")) == str(current["schema_version"])
        and int(stored.get("image_count", -1)) == int(current["image_count"])
        and str(stored.get("sha256", "")) == str(current["sha256"])
    )


def _cells(dataset_root: Path) -> Iterable[tuple[Path, Path, dict[str, Any]]]:
    public_mode = (dataset_root / "annotations.csv").is_file() and (dataset_root / "annotations.json").is_file()
    if public_mode:
        seen: set[tuple[str, str, str]] = set()
        for row in discover_public_training_images(dataset_root, compute_sha256=False, read_dimensions=False):
            key = (str(row["public_subject_id"]), str(row["background"]), str(row["gesture_id"]))
            if key in seen:
                continue
            seen.add(key)
            subject, background, gesture = key
            cell_dir = dataset_root / "training" / subject / background / gesture
            images_dir = cell_dir / "images" if (cell_dir / "images").is_dir() else cell_dir
            yield cell_dir, images_dir, {
                "schema_version": "tsgr_public_training_cell_v1",
                "public_subject_id": subject,
                "background": background,
                "gesture_id": gesture,
                "input_is_mirrored": True,
            }
        return
    for metadata_path in sorted(dataset_root.glob("training/*/*/*/training_cell.json")):
        cell_dir = metadata_path.parent
        images_dir = cell_dir / "images"
        if images_dir.is_dir():
            yield cell_dir, images_dir, json.loads(metadata_path.read_text(encoding="utf-8"))


def _base_worker_config(config: dict[str, Any], detection_profile: str) -> dict[str, Any]:
    merged = json.loads(json.dumps(config))
    merged.setdefault("mediapipe", {})["running_mode"] = "image"
    merged["mediapipe"].setdefault("image_detection", {})["active_profile"] = detection_profile
    merged["mediapipe"].setdefault("video_recovery", {})["enabled"] = False
    merged.setdefault("temporal_filter", {})["active"] = "none"
    merged.setdefault("classification_input", {}).update(
        {
            "landmark_source": "raw",
            "scale": "wrist_middle_mcp",
            "export_all_feature_branches": True,
        }
    )
    return merged


def _worker_initialize(config: dict[str, Any], model_path: str, detection_profile: str) -> None:
    """Create one persistent IMAGE pipeline for the lifetime of one worker."""
    global _WORKER_PIPELINE, _WORKER_CONFIG, _WORKER_MODEL_PATH, _WORKER_INITIALIZATION_S
    start = time.perf_counter()
    _WORKER_CONFIG = _base_worker_config(config, detection_profile)
    _WORKER_MODEL_PATH = str(model_path)
    _WORKER_PIPELINE = FramePipeline(_WORKER_CONFIG, _WORKER_MODEL_PATH)
    _WORKER_INITIALIZATION_S = time.perf_counter() - start


def _close_worker_pipeline() -> None:
    global _WORKER_PIPELINE
    if _WORKER_PIPELINE is not None:
        try:
            _WORKER_PIPELINE.close()
        finally:
            _WORKER_PIPELINE = None


def _worker_process_cell(task: dict[str, Any]) -> dict[str, Any]:
    """Process one subject/background/gesture cell using the persistent worker model."""
    if _WORKER_PIPELINE is None or _WORKER_CONFIG is None or _WORKER_MODEL_PATH is None:
        raise RuntimeError("Worker pipeline was not initialized.")
    cell_dir = Path(str(task["cell_dir"]))
    images_dir = Path(str(task["images_dir"]))
    metadata = dict(task["metadata"])
    start = time.perf_counter()
    partial_dir: Path | None = None
    try:
        mirrored = bool(metadata.get("input_is_mirrored", True))
        merged = json.loads(json.dumps(_WORKER_CONFIG))
        merged.setdefault("camera", {})["input_is_mirrored"] = mirrored
        merged.setdefault("mediapipe", {})["invert_handedness_labels"] = mirrored
        _WORKER_PIPELINE.config = merged

        final_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f") + f"_p{os.getpid()}"
        partial_name = f".partial_{final_name}_{uuid.uuid4().hex[:8]}"
        partial_dir = process_image_folder_run(
            images_dir,
            output_root=cell_dir / "runs",
            config=merged,
            model_path=_WORKER_MODEL_PATH,
            fps=float(task["fps"]),
            save_overlays=bool(task["save_overlays"]),
            save_crops=False,
            recursive=False,
            run_name=partial_name,
            pipeline=_WORKER_PIPELINE,
        )
        final_dir = partial_dir.parent / final_name
        partial_dir.replace(final_dir)
        signature = dict(task.get("inventory_signature") or training_cell_inventory_signature(cell_dir))
        signature.update(
            {
                "mediapipe_running_mode": "image",
                "mediapipe_image_profile": str(task.get("detection_profile", "high_recall")),
                "strict_right_hand_only": True,
            }
        )
        write_json(final_dir / "training_cell_processing_signature.json", signature)
        audit_rows = audit_training_cell(
            dataset_root=Path(str(task["dataset_root"])),
            cell_dir=cell_dir,
            run_dir=final_dir,
            metadata=metadata,
        )
        elapsed = time.perf_counter() - start
        image_count = len(audit_rows)
        return {
            "status": "ok",
            "subject": str(task["subject"]),
            "background": str(task["background"]),
            "gesture": str(task["gesture"]),
            "run_path": str(final_dir.resolve()),
            "image_count": image_count,
            "audit_rows": audit_rows,
            "worker_pid": os.getpid(),
            "worker_model_initialization_s": _WORKER_INITIALIZATION_S,
            "elapsed_wall_time_s": elapsed,
            "images_per_second": image_count / elapsed if elapsed > 0 else 0.0,
        }
    except BaseException as error:
        elapsed = time.perf_counter() - start
        if partial_dir is not None and partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        return {
            "status": "error",
            "subject": str(task["subject"]),
            "background": str(task["background"]),
            "gesture": str(task["gesture"]),
            "run_path": "",
            "image_count": int(task.get("image_count", 0)),
            "audit_rows": [],
            "worker_pid": os.getpid(),
            "worker_model_initialization_s": _WORKER_INITIALIZATION_S,
            "elapsed_wall_time_s": elapsed,
            "images_per_second": 0.0,
            "error": f"{type(error).__name__}: {error}",
        }


def _worker_process_chunk(
    config: dict[str, Any],
    model_path: str,
    detection_profile: str,
    tasks: list[dict[str, Any]],
    progress: bool,
    worker_slot: int,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Own one complete MediaPipe lifetime and process a deterministic cell chunk."""
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    close_error = ""
    _worker_initialize(config, model_path, detection_profile)
    try:
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            result = _worker_process_cell(task)
            results.append(result)
            if progress:
                label = f"{result['subject']}/{result['background']}/{result['gesture']}"
                print(
                    f"[worker {worker_slot} {index}/{total}] {label}: {result['status']} "
                    f"({float(result['elapsed_wall_time_s']):.2f}s, pid={result['worker_pid']})",
                    flush=True,
                )
            if fail_fast and result.get("status") == "error":
                break
    finally:
        try:
            _close_worker_pipeline()
        except BaseException as error:  # pragma: no cover - native shutdown safeguard
            close_error = f"{type(error).__name__}: {error}"
    return {
        "worker_slot": worker_slot,
        "worker_pid": os.getpid(),
        "results": results,
        "close_error": close_error,
        "elapsed_wall_time_s": time.perf_counter() - start,
    }


def process_experiment_training_photos(
    dataset_root: str | Path,
    *,
    config: dict[str, Any],
    model_path: str | Path,
    fps: float = 1.0,
    force: bool = False,
    subjects: set[str] | None = None,
    backgrounds: set[str] | None = None,
    gestures: set[str] | None = None,
    save_overlays: bool = False,
    detection_profile: str = "high_recall",
    workers: int = 0,
    progress: bool = True,
    fail_fast: bool = False,
    report_name: str | None = None,
) -> Path:
    """Process independent training photos in persistent worker processes."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not 0 < fps <= 30:
        raise ValueError("fps must be in the interval (0, 30].")
    if workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if detection_profile not in {"fast", "balanced", "high_recall"}:
        raise ValueError("Unsupported IMAGE detection profile.")

    effective_report_name = report_name or datetime.now().strftime("processing_%Y%m%d_%H%M%S_%f")
    report_dir = root / "reports" / "training_processing" / effective_report_name
    report_dir.mkdir(parents=True, exist_ok=False)

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for cell_dir, images_dir, metadata in _cells(root):
        subject = str(metadata["public_subject_id"])
        background = str(metadata["background"]).upper()
        gesture = str(metadata["gesture_id"]).upper()
        if subjects and subject not in subjects:
            continue
        if backgrounds and background not in backgrounds:
            continue
        if gestures and gesture not in gestures:
            continue
        existing = locate_processed_run(cell_dir)
        if existing is not None and not force and _run_matches_image_profile(existing, detection_profile, cell_dir):
            skipped.append(
                {
                    "subject": subject,
                    "background": background,
                    "gesture": gesture,
                    "run_path": str(existing.resolve()),
                    "reason": "already_processed_with_matching_image_profile",
                }
            )
            audit_rows.extend(
                audit_training_cell(
                    dataset_root=root,
                    cell_dir=cell_dir,
                    run_dir=existing,
                    metadata=metadata,
                )
            )
            if progress:
                print(f"[SKIP] {subject}/{background}/{gesture}: {existing.name}")
            continue
        if existing is not None and not force and progress:
            print(
                f"[REPROCESS] {subject}/{background}/{gesture}: "
                f"existing run is not IMAGE/{detection_profile}"
            )
        selected.append(
            {
                "dataset_root": str(root.resolve()),
                "cell_dir": str(cell_dir.resolve()),
                "images_dir": str(images_dir.resolve()),
                "metadata": metadata,
                "subject": subject,
                "background": background,
                "gesture": gesture,
                "image_count": int(metadata.get("image_count", 0)),
                "inventory_signature": training_cell_inventory_signature(cell_dir),
                "detection_profile": detection_profile,
                "fps": float(fps),
                "save_overlays": bool(save_overlays),
            }
        )

    resolved_workers = resolve_training_worker_count(workers, len(selected))
    processing_start = time.perf_counter()
    processed_results: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    abort_error: RuntimeError | None = None

    if selected:
        chunks = partition_training_tasks(selected, resolved_workers)
        if len(chunks) == 1:
            chunk_result = _worker_process_chunk(
                config,
                str(Path(model_path).resolve()),
                detection_profile,
                chunks[0],
                progress,
                1,
                fail_fast,
            )
            worker_summaries.append(chunk_result)
            processed_results.extend(chunk_result["results"])
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as executor:
                futures: dict[Future[dict[str, Any]], int] = {
                    executor.submit(
                        _worker_process_chunk,
                        config,
                        str(Path(model_path).resolve()),
                        detection_profile,
                        chunk,
                        progress,
                        slot,
                        fail_fast,
                    ): slot
                    for slot, chunk in enumerate(chunks, start=1)
                }
                for future in as_completed(futures):
                    chunk_result = future.result()
                    worker_summaries.append(chunk_result)
                    if chunk_result.get("close_error"):
                        raise RuntimeError(
                            f"Worker {chunk_result.get('worker_pid')} failed to close MediaPipe cleanly: "
                            f"{chunk_result['close_error']}"
                        )
                    processed_results.extend(chunk_result["results"])
                    if fail_fast:
                        first_error = next(
                            (item for item in chunk_result["results"] if item.get("status") == "error"),
                            None,
                        )
                        if first_error is not None:
                            abort_error = RuntimeError(
                                f"Training processing stopped after {first_error['subject']}/"
                                f"{first_error['background']}/{first_error['gesture']}: "
                                f"{first_error.get('error', 'processing error')}"
                            )
                            for pending in futures:
                                pending.cancel()
                            break
    processing_elapsed = time.perf_counter() - processing_start

    processed_results.sort(
        key=lambda row: (str(row.get("subject", "")), str(row.get("background", "")), str(row.get("gesture", "")))
    )
    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for result in processed_results:
        if result.get("status") == "ok":
            processed.append(
                {
                    "subject": result["subject"],
                    "background": result["background"],
                    "gesture": result["gesture"],
                    "run_path": result["run_path"],
                    "image_count": int(result.get("image_count", 0)),
                    "worker_pid": result.get("worker_pid", ""),
                    "worker_model_initialization_s": result.get("worker_model_initialization_s", 0.0),
                    "elapsed_wall_time_s": result.get("elapsed_wall_time_s", 0.0),
                    "images_per_second": result.get("images_per_second", 0.0),
                }
            )
            audit_rows.extend(result.get("audit_rows") or [])
        else:
            failed.append(
                {
                    "subject": result.get("subject", ""),
                    "background": result.get("background", ""),
                    "gesture": result.get("gesture", ""),
                    "error": result.get("error", "processing error"),
                    "worker_pid": result.get("worker_pid", ""),
                    "elapsed_wall_time_s": result.get("elapsed_wall_time_s", 0.0),
                }
            )

    audit_rows.sort(
        key=lambda row: (
            str(row.get("public_subject_id", "")),
            str(row.get("background", "")),
            str(row.get("gesture_id", "")),
            int(row.get("frame_index", -1)),
        )
    )
    write_training_detection_audit(root, report_dir, audit_rows)

    worker_rows: list[dict[str, Any]] = []
    for summary in sorted(worker_summaries, key=lambda item: int(item.get("worker_slot", 0))):
        results = list(summary.get("results") or [])
        image_count = sum(int(item.get("image_count", 0)) for item in results if item.get("status") == "ok")
        cell_count = sum(item.get("status") == "ok" for item in results)
        failed_cell_count = sum(item.get("status") == "error" for item in results)
        sum_cell_elapsed = sum(float(item.get("elapsed_wall_time_s", 0.0)) for item in results)
        worker_rows.append(
            {
                "worker_slot": int(summary.get("worker_slot", 0)),
                "worker_pid": int(summary.get("worker_pid", 0)),
                "cell_count": cell_count,
                "failed_cell_count": failed_cell_count,
                "image_count": image_count,
                "worker_wall_time_s": float(summary.get("elapsed_wall_time_s", 0.0)),
                "sum_cell_elapsed_s": sum_cell_elapsed,
                "aggregate_images_per_second": image_count / sum_cell_elapsed if sum_cell_elapsed > 0 else 0.0,
                "model_initialization_s": max(
                    [float(item.get("worker_model_initialization_s", 0.0)) for item in results] or [0.0]
                ),
                "close_error": str(summary.get("close_error") or ""),
            }
        )

    # Keep worker statistics plain CSV for quick local inspection without pandas.
    import csv

    with (report_dir / "worker_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "worker_slot", "worker_pid", "cell_count", "failed_cell_count", "image_count",
            "worker_wall_time_s", "sum_cell_elapsed_s", "aggregate_images_per_second",
            "model_initialization_s", "close_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(worker_rows)

    total_processed_images = sum(int(item.get("image_count", 0)) for item in processed)
    serial_equivalent_s = sum(float(item.get("elapsed_wall_time_s", 0.0)) for item in processed)
    write_json(
        report_dir / "processing_report.json",
        {
            "dataset_root": str(root.resolve()),
            "fps": fps,
            "classification_branch": "raw.wrist_middle_mcp",
            "temporal_filter": "none",
            "mediapipe_running_mode": "image",
            "detection_profile": detection_profile,
            "strict_right_hand_only": True,
            "requested_workers": workers,
            "resolved_workers": resolved_workers,
            "worker_processes_used": len(worker_rows),
            "multiprocessing_start_method": "spawn" if len(worker_rows) > 1 else "serial",
            "processing_wall_time_s": processing_elapsed,
            "processed_image_count": total_processed_images,
            "aggregate_images_per_second": total_processed_images / processing_elapsed if processing_elapsed > 0 else 0.0,
            "serial_equivalent_cell_time_s": serial_equivalent_s,
            "estimated_parallel_speedup": serial_equivalent_s / processing_elapsed if processing_elapsed > 0 else 0.0,
            "processed": processed,
            "skipped": skipped,
            "failures": failed,
        },
    )
    if abort_error is not None:
        raise abort_error
    return report_dir
