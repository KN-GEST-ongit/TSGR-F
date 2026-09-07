"""Generate reproducible S1, LOBO, LOSO, and LOSO-background experiment folds."""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tsgr.dataset.contract import (
    PUBLIC_CONTRACT_SCHEMA,
    discover_public_test_takes,
    discover_public_training_images,
    load_public_annotations,
)
from tsgr.reference_models.manifest import ReferenceManifestEntry, locate_processed_run, write_reference_manifest
from tsgr.utils.serialization import write_json

SCENARIOS = ("S1_ALL_IN_DOMAIN", "S2_LOBO", "S3_LOSO", "S4_LOSO_BACKGROUND")


@dataclass(frozen=True, slots=True)
class FoldDefinition:
    scenario: str
    fold_id: str
    held_out_subject: str | None = None
    held_out_background: str | None = None


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


def resolve_scenario_worker_count(requested_workers: int, item_count: int) -> int:
    """Resolve worker count for filesystem-heavy scenario preparation."""
    if requested_workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if item_count <= 0:
        return 0
    if requested_workers > 0:
        return min(requested_workers, item_count)
    cpu_count = os.cpu_count() or 1
    automatic = max(1, min(8, cpu_count))
    return min(automatic, item_count)


def _contains_feature_artifacts(path: Path) -> bool:
    return (path / "features" / "feature_arrays.npz").is_file() or (path / "feature_arrays.npz").is_file()


def _fast_locate_training_run(cell_dir: Path) -> Path | None:
    """Locate the newest direct run without recursively walking training images."""
    runs_dir = cell_dir / "runs"
    candidates: list[Path] = []
    if runs_dir.is_dir():
        for candidate in runs_dir.iterdir():
            if not candidate.is_dir() or candidate.name.startswith(".partial_"):
                continue
            if _contains_feature_artifacts(candidate):
                candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    # Fall back to recursive artifact discovery for portable processed-run layouts.
    return locate_processed_run(cell_dir)


def build_training_run_index(
    dataset_root: Path,
    training_rows: list[dict[str, str]],
    *,
    workers: int = 0,
) -> tuple[dict[tuple[str, str, str], Path | None], list[dict[str, Any]], float, int]:
    """Resolve each unique training cell once and reuse the result across every fold."""
    keys = sorted(
        {
            (row["gesture_id"], row["public_subject_id"], row["background"])
            for row in training_rows
        }
    )
    resolved_workers = resolve_scenario_worker_count(workers, len(keys))
    start = time.perf_counter()

    def locate(key: tuple[str, str, str]) -> tuple[tuple[str, str, str], Path | None]:
        gesture, subject, background = key
        cell_dir = dataset_root / "training" / subject / background / gesture
        return key, _fast_locate_training_run(cell_dir)

    if resolved_workers <= 1:
        located = [locate(key) for key in keys]
    else:
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            located = list(executor.map(locate, keys))
    elapsed = time.perf_counter() - start
    index = {key: run for key, run in located}
    rows = [
        {
            "gesture_id": key[0],
            "public_subject_id": key[1],
            "background": key[2],
            "run_found": int(run is not None),
            "run_path": (
                run.resolve().relative_to(dataset_root.resolve()).as_posix()
                if run is not None and dataset_root.resolve() in run.resolve().parents
                else (str(run.resolve()) if run is not None else "")
            ),
        }
        for key, run in located
    ]
    return index, rows, elapsed, resolved_workers


def build_fold_definitions(subjects: Iterable[str], backgrounds: Iterable[str]) -> list[FoldDefinition]:
    subjects = sorted(set(subjects))
    backgrounds = sorted(set(backgrounds))
    folds = [FoldDefinition("S1_ALL_IN_DOMAIN", "all")]
    folds.extend(
        FoldDefinition("S2_LOBO", f"background_{background}", held_out_background=background)
        for background in backgrounds
    )
    folds.extend(
        FoldDefinition("S3_LOSO", f"subject_{subject}", held_out_subject=subject)
        for subject in subjects
    )
    folds.extend(
        FoldDefinition(
            "S4_LOSO_BACKGROUND",
            f"subject_{subject}__background_{background}",
            held_out_subject=subject,
            held_out_background=background,
        )
        for subject in subjects
        for background in backgrounds
    )
    return folds


def _training_selected(row: dict[str, str], fold: FoldDefinition) -> bool:
    if fold.scenario == "S1_ALL_IN_DOMAIN":
        return True
    if fold.scenario == "S2_LOBO":
        return row["background"] != fold.held_out_background
    if fold.scenario == "S3_LOSO":
        return row["public_subject_id"] != fold.held_out_subject
    if fold.scenario == "S4_LOSO_BACKGROUND":
        return row["public_subject_id"] != fold.held_out_subject and row["background"] != fold.held_out_background
    raise ValueError(f"Unsupported scenario: {fold.scenario}")


def _test_selected(row: dict[str, str], fold: FoldDefinition) -> bool:
    if fold.scenario == "S1_ALL_IN_DOMAIN":
        return True
    if fold.scenario == "S2_LOBO":
        return row["background"] == fold.held_out_background
    if fold.scenario == "S3_LOSO":
        return row["public_subject_id"] == fold.held_out_subject
    if fold.scenario == "S4_LOSO_BACKGROUND":
        return row["public_subject_id"] == fold.held_out_subject and row["background"] == fold.held_out_background
    raise ValueError(f"Unsupported scenario: {fold.scenario}")


def generate_experiment_folds(
    dataset_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    scenarios: Iterable[str] | None = None,
    backgrounds: Iterable[str] = ("BLACK", "BLUE", "WHITE"),
    workers: int = 0,
    progress: bool = False,
) -> Path:
    """Generate fold manifests using one parallel training-run index for all folds."""
    total_start = time.perf_counter()
    root = Path(dataset_root)
    read_start = time.perf_counter()
    load_public_annotations(root, require_both=True)
    training = discover_public_training_images(root, compute_sha256=True, read_dimensions=False)
    takes = discover_public_test_takes(root, read_video_fps=False)
    read_elapsed = time.perf_counter() - read_start
    if not training and not takes:
        raise ValueError("No dataset records were found. Build annotations.csv/json for the public contract first.")
    if workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")

    subjects = sorted({row["public_subject_id"] for row in training + takes})
    available_backgrounds = sorted(set(backgrounds) | {row["background"] for row in training + takes})
    selected_scenarios = set(scenarios or SCENARIOS)
    unknown = selected_scenarios - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

    plan_dir = (
        Path(output_dir)
        if output_dir
        else root / "reports" / "experiments" / datetime.now().strftime("experiment_plan_%Y%m%d_%H%M%S_%f")
    )
    plan_dir.mkdir(parents=True, exist_ok=False)

    run_index, run_index_rows, run_index_elapsed, resolved_workers = build_training_run_index(
        root, training, workers=workers
    )
    _write_csv(
        plan_dir / "training_run_index.csv",
        run_index_rows,
        ["gesture_id", "public_subject_id", "background", "run_found", "run_path"],
    )

    fold_summaries: list[dict[str, Any]] = []
    fold_generation_start = time.perf_counter()
    folds = [
        fold
        for fold in build_fold_definitions(subjects, available_backgrounds)
        if fold.scenario in selected_scenarios
    ]
    for fold_number, fold in enumerate(folds, start=1):
        fold_dir = plan_dir / fold.scenario / fold.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        selected_training = [row for row in training if _training_selected(row, fold)]
        selected_takes = [row for row in takes if _test_selected(row, fold)]
        if not selected_training and not selected_takes:
            fold_status = "deferred_no_training_or_test_data"
        elif not selected_training:
            fold_status = "deferred_no_training_data"
        elif not selected_takes:
            fold_status = "deferred_no_test_data"
        else:
            fold_status = "active"

        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        manifest_entries: list[ReferenceManifestEntry] = []
        missing_runs: list[dict[str, str]] = []
        incomplete_annotations = [
            row for row in selected_takes if row.get("annotation_complete", "false").lower() != "true"
        ]
        train_gestures = sorted({row["gesture_id"] for row in selected_training})
        model_gestures: list[str] = []

        if fold_status == "active":
            _write_csv(fold_dir / "training_images.csv", selected_training, list(training[0]) if training else [])
            _write_csv(fold_dir / "test_takes.csv", selected_takes, list(takes[0]) if takes else [])
            for row in selected_training:
                grouped.setdefault(
                    (row["gesture_id"], row["public_subject_id"], row["background"]), []
                ).append(row)
            for (gesture, subject, background), rows in sorted(grouped.items()):
                cell_dir = root / "training" / subject / background / gesture
                run = run_index.get((gesture, subject, background))
                if run is None:
                    missing_runs.append(
                        {
                            "gesture_id": gesture,
                            "public_subject_id": subject,
                            "background": background,
                            "cell_path": cell_dir.relative_to(root).as_posix(),
                        }
                    )
                else:
                    manifest_entries.append(
                        ReferenceManifestEntry(
                            gesture,
                            subject,
                            f"{background}",
                            run.resolve(),
                            True,
                            f"fold={fold.scenario}/{fold.fold_id}; images={len(rows)}",
                        )
                    )
            write_reference_manifest(fold_dir / "reference_manifest.csv", manifest_entries)
            _write_csv(
                fold_dir / "missing_training_runs.csv",
                missing_runs,
                ["gesture_id", "public_subject_id", "background", "cell_path"],
            )
            model_gestures = sorted({entry.gesture_id for entry in manifest_entries})

        ready_model = (
            fold_status == "active"
            and bool(manifest_entries)
            and not missing_runs
            and train_gestures == model_gestures
        )
        ready_evaluation = ready_model and not incomplete_annotations
        payload = {
            **asdict(fold),
            "fold_status": fold_status,
            "deferred": fold_status != "active",
            "dataset_root": ".",
            "dataset_contract": PUBLIC_CONTRACT_SCHEMA,
            "training_image_count": len(selected_training),
            "test_take_count": len(selected_takes),
            "training_subjects": sorted({row["public_subject_id"] for row in selected_training}),
            "training_backgrounds": sorted({row["background"] for row in selected_training}),
            "training_gestures": train_gestures,
            "test_subjects": sorted({row["public_subject_id"] for row in selected_takes}),
            "test_backgrounds": sorted({row["background"] for row in selected_takes}),
            "test_gestures": sorted({row["gesture_id"] for row in selected_takes}),
            "processed_training_sessions": len(manifest_entries),
            "missing_training_runs": len(missing_runs),
            "incomplete_test_annotations": len(incomplete_annotations),
            "ready_for_model_building": ready_model,
            "ready_for_evaluation": ready_evaluation,
            "leakage_policy": "Formal masks, weights, covariance, thresholds, and calibration must be learned from this fold's training manifest only.",
        }
        write_json(fold_dir / "fold.json", payload)
        fold_summaries.append(payload)
        if progress:
            print(
                f"[{fold_number}/{len(folds)}] {fold.scenario}/{fold.fold_id}: {fold_status} "
                f"train={len(selected_training)} test={len(selected_takes)}",
                flush=True,
            )

    fold_generation_elapsed = time.perf_counter() - fold_generation_start
    write_json(
        plan_dir / "experiment_plan.json",
        {
            "schema_version": "tsgr_experiment_plan_v4",
            "dataset_root": ".",
            "dataset_contract": PUBLIC_CONTRACT_SCHEMA,
            "subjects": subjects,
            "backgrounds": available_backgrounds,
            "scenarios": sorted(selected_scenarios),
            "fold_count": len(fold_summaries),
            "active_fold_count": sum(row["fold_status"] == "active" for row in fold_summaries),
            "deferred_fold_count": sum(row["fold_status"] != "active" for row in fold_summaries),
            "folds": fold_summaries,
        },
    )
    write_json(
        plan_dir / "scenario_generation_performance.json",
        {
            "dataset_root": ".",
            "dataset_contract": PUBLIC_CONTRACT_SCHEMA,
            "requested_workers": workers,
            "resolved_workers": resolved_workers,
            "training_image_count": len(training),
            "test_take_count": len(takes),
            "unique_training_cell_count": len(run_index_rows),
            "fold_count": len(fold_summaries),
            "metadata_read_elapsed_s": read_elapsed,
            "training_run_index_elapsed_s": run_index_elapsed,
            "fold_generation_elapsed_s": fold_generation_elapsed,
            "total_elapsed_s": time.perf_counter() - total_start,
            "run_discovery_strategy": "parallel_unique_cell_index_with_direct_runs_fast_path",
        },
    )
    return plan_dir
