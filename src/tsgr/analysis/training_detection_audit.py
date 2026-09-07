"""Audit high-recall MediaPipe IMAGE processing for training photographs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from tsgr.analysis.audit_utils import mediapipe_failure_reason, read_jsonl, save_review_pair
from tsgr.dataset.contract import discover_public_training_images
from tsgr.utils.serialization import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_training_cell(
    *,
    dataset_root: Path,
    cell_dir: Path,
    run_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return one audit row per canonical training photograph."""
    manifest = sorted(_read_csv(cell_dir / "images_manifest.csv"), key=lambda row: row.get("filename", ""))
    if not manifest and (dataset_root / "annotations.csv").is_file():
        manifest = [
            row for row in discover_public_training_images(dataset_root, compute_sha256=False, read_dimensions=False)
            if str(row.get("public_subject_id")) == str(metadata.get("public_subject_id"))
            and str(row.get("background")) == str(metadata.get("background"))
            and str(row.get("gesture_id")) == str(metadata.get("gesture_id"))
        ]
    results = {int(item.get("frame_index", -1)): item for item in read_jsonl(run_dir / "frame_results.jsonl")}
    rows: list[dict[str, Any]] = []
    for frame_index, image_row in enumerate(manifest):
        result = results.get(frame_index)
        diagnostics = dict((result or {}).get("detection_diagnostics") or {})
        attempts = diagnostics.get("attempts") or []
        selected_attempt = str(diagnostics.get("selected_attempt") or "")
        recovered = bool(diagnostics.get("recovered", False))
        status = str((result or {}).get("status", "not_processed"))
        failure = mediapipe_failure_reason(result)
        if status == "critical_left_hand_detected":
            category = "critical_left_hand"
        elif failure:
            category = "unrecovered_failure"
        elif recovered:
            category = f"recovered_{selected_attempt.replace(':', '_') or 'retry'}"
        elif diagnostics.get("handedness_retry_observation"):
            category = "recovered_after_handedness_retry"
        elif diagnostics:
            category = "standard_success"
        else:
            category = "legacy_or_missing_diagnostics"
        rows.append(
            {
                "public_subject_id": str(metadata.get("public_subject_id", "")),
                "background": str(metadata.get("background", "")).upper(),
                "gesture_id": str(metadata.get("gesture_id", "")).upper(),
                "frame_index": frame_index,
                "image_id": image_row.get("image_id", ""),
                "filename": image_row.get("filename", ""),
                "relative_path": image_row.get("relative_path", ""),
                "run_path": str(run_dir.resolve()),
                "status": status,
                "failure_reason": failure,
                "detection_profile": diagnostics.get("profile", ""),
                "selected_attempt": selected_attempt,
                "attempt_count": diagnostics.get("attempt_count", len(attempts)),
                "recovered": int(recovered),
                "final_outcome": diagnostics.get("final_outcome", ""),
                "handedness_retry_observation": int(bool(diagnostics.get("handedness_retry_observation"))),
                "left_observed_in_any_attempt": int(bool(diagnostics.get("left_observed_in_any_attempt"))),
                "category": category,
            }
        )
    return rows


def write_training_detection_audit(
    dataset_root: Path,
    report_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write CSV, recovery summaries, and review images for nontrivial outcomes."""

    fields = [
        "public_subject_id", "background", "gesture_id", "frame_index", "image_id",
        "filename", "relative_path", "run_path",
        "status", "failure_reason", "detection_profile", "selected_attempt", "attempt_count",
        "recovered", "final_outcome", "handedness_retry_observation", "left_observed_in_any_attempt",
        "category", "review_image",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    result_cache: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row["category"] == "standard_success":
            row["review_image"] = ""
            continue
        source = dataset_root / str(row.get("relative_path", ""))
        run_path = str(row["run_path"])
        if run_path not in result_cache:
            result_cache[run_path] = {
                int(item.get("frame_index", -1)): item
                for item in read_jsonl(Path(run_path) / "frame_results.jsonl")
            }
        result = result_cache[run_path].get(int(row["frame_index"]))
        destination = (
            report_dir / "detection_review" / str(row["category"]) /
            str(row["public_subject_id"]) / str(row["background"]) /
            str(row["gesture_id"]) / f"{row['image_id'] or row['filename']}.png"
        )
        saved = save_review_pair(
            source,
            destination,
            result=result,
            header_lines=[
                f"{row['public_subject_id']}/{row['background']}/{row['gesture_id']} {row['image_id']}",
                f"category={row['category']} attempt={row['selected_attempt']} failure={row['failure_reason'] or 'none'}",
            ],
        )
        row["review_image"] = str(saved.resolve()) if saved else ""
    csv_path = report_dir / "training_detection_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    cleanup_fields = [
        "image_id", "public_subject_id", "background", "gesture_id", "status",
        "failure_reason", "category", "relative_path",
        "selected_attempt", "attempt_count", "review_image",
    ]

    def write_subset(name: str, subset: list[dict[str, Any]]) -> None:
        path = report_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cleanup_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)

    critical_left = [row for row in rows if row.get("status") == "critical_left_hand_detected"]
    no_hand = [row for row in rows if row.get("status") == "no_hand"]
    no_right = [row for row in rows if row.get("status") == "no_right_hand"]
    unrecovered = [row for row in rows if row.get("category") == "unrecovered_failure"]
    cleanup_candidates = []
    for row in rows:
        status = str(row.get("status") or "")
        if status not in {"critical_left_hand_detected", "no_hand", "no_right_hand"}:
            continue
        item = dict(row)
        item["cleanup_recommendation"] = (
            "remove_or_manual_review" if status == "critical_left_hand_detected"
            else "manual_review_then_remove_if_source_is_invalid"
        )
        cleanup_candidates.append(item)
    cleanup_fields_with_action = [*cleanup_fields, "cleanup_recommendation"]
    write_subset("critical_left_hand_images.csv", critical_left)
    write_subset("no_hand_images.csv", no_hand)
    write_subset("no_right_hand_images.csv", no_right)
    write_subset("unrecovered_detection_failures.csv", unrecovered)
    with (report_dir / "detection_cleanup_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cleanup_fields_with_action, extrasaction="ignore")
        writer.writeheader(); writer.writerows(cleanup_candidates)

    counts = Counter(str(row["category"]) for row in rows)
    attempts = Counter(str(row.get("selected_attempt") or "none") for row in rows)
    write_json(
        report_dir / "training_detection_recovery_summary.json",
        {
            "image_count": len(rows),
            "category_counts": dict(sorted(counts.items())),
            "selected_attempt_counts": dict(sorted(attempts.items())),
            "recovered_count": sum(int(row.get("recovered") or 0) for row in rows),
            "critical_left_hand_count": counts.get("critical_left_hand", 0),
            "unrecovered_failure_count": counts.get("unrecovered_failure", 0),
        },
    )
