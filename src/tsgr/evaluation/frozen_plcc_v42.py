"""Post-hoc PLCC sensitivity study for the fixed TSGR-F routing architecture.

This runner is explicitly exploratory: it rebuilds TRAIN-only reference and routing models
at alternative compact-feature correlation thresholds and scores held-out ranking accuracy.
The sensitivity results must not be used to redefine the fixed final method.

Held-out folds are independent and may be evaluated concurrently in spawned worker
processes. The evaluator is summary-only for the PLCC study and predicts valid frames in
bounded NumPy batches to reduce Python-call overhead without changing any scoring
equation.
"""
from __future__ import annotations

import copy
import csv
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tsgr.dataset.contract import resolve_dataset_root_from_plan
from tsgr.evaluation.frozen_routing_v40 import (
    _load_model,
    _predict_frozen,
    _run_image_features,
    build_frozen_routing_models,
)
from tsgr.evaluation.experiment_evaluator import _load_run_features
from tsgr.evaluation.io_utils import read_csv_rows, write_workbook
from tsgr.experiments.model_building import build_experiment_models_parallel
from tsgr.utils.serialization import write_json

SCHEMA = "tsgrf_frozen_routing_plcc_sensitivity_v42_1"
STANDARD_GRID = (0.995, 0.990, 0.985, 0.980, 0.975, 0.970, 0.965, 0.960, 0.955, 0.950)

# Process-local read-only caches populated by the ProcessPool initializer.
_PLCC_MANIFEST: dict[str, str] | None = None
_PLCC_GT_GESTURE_ROWS: dict[str, list[tuple[int, str]]] | None = None
_PLCC_DATASET_ROOT: Path | None = None
_PLCC_FROZEN_MODEL_DIR: Path | None = None
_PLCC_BRANCH_NAME: str | None = None
_PLCC_BATCH_SIZE: int = 64


def variant_name(value: float) -> str:
    return f"frozen_plcc_{float(value):.3f}".replace(".", "p")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["scenario", "fold_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_evaluation_worker_count(requested_workers: int, job_count: int) -> int:
    """Resolve a bounded process count for held-out fold evaluation.

    ``0`` selects an intentionally conservative automatic value.  An explicit positive
    value is honored up to the number of independent folds.  Negative values are invalid.
    """
    if requested_workers < 0:
        raise ValueError("evaluation_workers must be zero (automatic) or a positive integer.")
    if job_count <= 0:
        return 0
    if requested_workers > 0:
        return min(int(requested_workers), int(job_count))
    cpu_count = os.cpu_count() or 1
    automatic = max(1, min(8, max(1, cpu_count // 2)))
    return min(automatic, int(job_count))


def _init_plcc_evaluation_worker(
    status_csv: str,
    gt_csv: str,
    dataset_root: str,
    frozen_model_dir: str,
    branch_name: str,
    batch_size: int,
) -> None:
    """Load immutable report tables once per spawned worker process."""
    global _PLCC_MANIFEST, _PLCC_GT_GESTURE_ROWS, _PLCC_DATASET_ROOT
    global _PLCC_FROZEN_MODEL_DIR, _PLCC_BRANCH_NAME, _PLCC_BATCH_SIZE

    manifest_rows = read_csv_rows(Path(status_csv))
    _PLCC_MANIFEST = {str(row["take_id"]): str(row.get("run_path", "")) for row in manifest_rows}

    # Keep only the two GT fields required by this ranking-only sensitivity audit.
    # This materially lowers per-worker RAM versus retaining every CSV column as a dict.
    gt_by_take: dict[str, list[tuple[int, str]]] = {}
    for row in read_csv_rows(Path(gt_csv)):
        state = str(row.get("evaluation_state", ""))
        if state.startswith("GESTURE_"):
            gesture = str(row.get("gesture_id") or state.removeprefix("GESTURE_")).upper()
            gt_by_take.setdefault(str(row["take_id"]), []).append((int(row["frame_index"]), gesture))
    for rows in gt_by_take.values():
        rows.sort(key=lambda item: item[0])
    _PLCC_GT_GESTURE_ROWS = gt_by_take

    _PLCC_DATASET_ROOT = Path(dataset_root)
    _PLCC_FROZEN_MODEL_DIR = Path(frozen_model_dir)
    _PLCC_BRANCH_NAME = str(branch_name)
    _PLCC_BATCH_SIZE = max(1, int(batch_size))


def _evaluate_one_plcc_fold(job: tuple[str, str, str]) -> dict[str, Any]:
    """Evaluate one frozen fold and return only sufficient summary statistics."""
    fold_json_text, scenario, fold_id = job
    start = time.perf_counter()
    try:
        if _PLCC_MANIFEST is None or _PLCC_GT_GESTURE_ROWS is None:
            raise RuntimeError("PLCC evaluation worker was not initialized.")
        if _PLCC_DATASET_ROOT is None or _PLCC_FROZEN_MODEL_DIR is None or _PLCC_BRANCH_NAME is None:
            raise RuntimeError("PLCC evaluation worker paths were not initialized.")

        fold_json = Path(fold_json_text)
        model, arrays = _load_model(_PLCC_FROZEN_MODEL_DIR, scenario, fold_id)
        if str(model["branch_name"]) != _PLCC_BRANCH_NAME:
            raise ValueError(f"Branch mismatch for {scenario}/{fold_id}")
        gestures = tuple(str(value) for value in model["gesture_ids"])

        test_takes = sorted(str(row["take_id"]) for row in read_csv_rows(fold_json.parent / "test_takes.csv"))
        total = 0
        covered = 0
        correct_base = 0
        correct_final = 0

        for take in test_takes:
            run_path = _PLCC_MANIFEST.get(take)
            if run_path is None:
                raise ValueError(f"Missing take {take} in processing report.")
            run = Path(run_path)
            if not run.is_absolute():
                run = _PLCC_DATASET_ROOT / run

            frame_indices, _status, values = _load_run_features(run, _PLCC_BRANCH_NAME)
            position = {int(frame_index): idx for idx, frame_index in enumerate(frame_indices.tolist())}
            image_matrix, image_ok, image_ids = _run_image_features(run, frame_indices)
            expected_image_ids = tuple(str(value) for value in arrays["image_feature_ids"].tolist())
            if image_ids and tuple(image_ids) != expected_image_ids:
                raise ValueError(f"IMAGE feature schema mismatch for {take}")

            gt_rows = _PLCC_GT_GESTURE_ROWS.get(take, [])
            total += len(gt_rows)
            if not gt_rows:
                continue

            valid_positions: list[int] = []
            valid_gt_indices: list[int] = []
            for frame_index, gt_gesture in gt_rows:
                idx = position.get(int(frame_index))
                if idx is None:
                    continue
                if not np.isfinite(values[idx]).all():
                    continue
                if idx >= len(image_ok) or not bool(image_ok[idx]) or not np.isfinite(image_matrix[idx]).all():
                    continue
                try:
                    gt_index = gestures.index(gt_gesture)
                except ValueError as error:
                    raise ValueError(f"Unknown GT gesture {gt_gesture!r} for {take}") from error
                valid_positions.append(idx)
                valid_gt_indices.append(gt_index)

            if not valid_positions:
                continue
            covered += len(valid_positions)
            valid_positions_np = np.asarray(valid_positions, dtype=np.int64)
            gt_np = np.asarray(valid_gt_indices, dtype=np.int64)

            for lo in range(0, len(valid_positions_np), _PLCC_BATCH_SIZE):
                hi = min(lo + _PLCC_BATCH_SIZE, len(valid_positions_np))
                pos_batch = valid_positions_np[lo:hi]
                gt_batch = gt_np[lo:hi]
                base, _after_os, _after_iy, final, _scores = _predict_frozen(
                    model,
                    arrays,
                    values[pos_batch],
                    image_matrix[pos_batch],
                )
                correct_base += int(np.count_nonzero(base.astype(np.int64) == gt_batch))
                correct_final += int(np.count_nonzero(final.astype(np.int64) == gt_batch))

        elapsed = time.perf_counter() - start
        return {
            "status": "ok",
            "scenario": scenario,
            "fold_id": fold_id,
            "gt_gesture_frames": int(total),
            "complete_input_frames": int(covered),
            "feature_coverage": covered / total if total else float("nan"),
            "baseline_conditional_accuracy": correct_base / covered if covered else float("nan"),
            "final_conditional_accuracy": correct_final / covered if covered else float("nan"),
            "baseline_end_to_end_accuracy": correct_base / total if total else float("nan"),
            "final_end_to_end_accuracy": correct_final / total if total else float("nan"),
            "final_errors_given_input": int(covered - correct_final),
            "worker_pid": os.getpid(),
            "elapsed_wall_time_s": elapsed,
            "error": "",
        }
    except BaseException as error:  # noqa: BLE001 - worker must serialize the failure.
        return {
            "status": "error",
            "scenario": scenario,
            "fold_id": fold_id,
            "worker_pid": os.getpid(),
            "elapsed_wall_time_s": time.perf_counter() - start,
            "error": f"{type(error).__name__}: {error}",
        }


def _discover_plcc_eval_jobs(
    plan: Path,
    frozen_dir: Path,
    scenarios: set[str],
) -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    for fold_json in sorted(plan.glob("*/*/fold.json")):
        meta = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(meta["scenario"])
        fold_id = str(meta["fold_id"])
        if scenarios and scenario not in scenarios:
            continue
        if str(meta.get("fold_status", "active")) != "active":
            continue
        model_json = frozen_dir / scenario / fold_id / "model.json"
        arrays_npz = frozen_dir / scenario / fold_id / "model_arrays.npz"
        if model_json.is_file() and arrays_npz.is_file():
            jobs.append((str(fold_json), scenario, fold_id))
    return jobs


def _fold_checkpoint_path(eval_dir: Path, scenario: str, fold_id: str) -> Path:
    return eval_dir / "fold_results" / scenario / f"{fold_id}.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_frozen_routing_plcc_parallel(
    plan: Path,
    *,
    frozen_dir: Path,
    test_processing_report: str | Path,
    ground_truth_report: str | Path,
    output_dir: Path,
    branch_name: str,
    scenarios: set[str],
    evaluation_workers: int,
    evaluation_batch_size: int,
    dataset_root_override: str | Path | None,
    resume: bool,
    progress: bool,
) -> dict[str, Any]:
    """Parallel summary-only held-out evaluator used by the PLCC sensitivity sweep."""
    dataset = resolve_dataset_root_from_plan(plan, override=dataset_root_override)
    proc = Path(test_processing_report)
    gtroot = Path(ground_truth_report)
    status_csv = proc / "test_take_mediapipe_status.csv"
    gt_csv = gtroot / "frame_ground_truth.csv"
    if not status_csv.is_file():
        raise ValueError(f"Invalid test-processing report: missing {status_csv}")
    if not gt_csv.is_file():
        raise ValueError(f"Invalid ground-truth report: missing {gt_csv}")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be a positive integer.")

    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _discover_plcc_eval_jobs(plan, frozen_dir, scenarios)
    if not jobs:
        raise ValueError("No active folds with frozen models were selected for PLCC held-out evaluation.")

    completed_rows: list[dict[str, Any]] = []
    pending_jobs: list[tuple[str, str, str]] = []
    for job in jobs:
        _fold_json, scenario, fold_id = job
        checkpoint = _fold_checkpoint_path(output_dir, scenario, fold_id)
        if resume and checkpoint.is_file():
            payload = _read_json(checkpoint)
            if payload.get("status") == "ok":
                completed_rows.append(payload)
                continue
        pending_jobs.append(job)

    resolved_workers = resolve_evaluation_worker_count(evaluation_workers, len(pending_jobs))
    eval_start = time.perf_counter()
    new_rows: list[dict[str, Any]] = []

    if pending_jobs:
        if progress and completed_rows:
            print(
                f"[PLCC eval resume] reusing {len(completed_rows)}/{len(jobs)} completed fold checkpoints; "
                f"remaining={len(pending_jobs)}",
                flush=True,
            )
        if resolved_workers <= 1:
            _init_plcc_evaluation_worker(
                str(status_csv), str(gt_csv), str(dataset), str(frozen_dir), branch_name, evaluation_batch_size
            )
            for index, job in enumerate(pending_jobs, start=1):
                row = _evaluate_one_plcc_fold(job)
                if row["status"] != "ok":
                    raise RuntimeError(
                        f"PLCC held-out evaluation failed for {row['scenario']}/{row['fold_id']}: {row['error']}"
                    )
                checkpoint = _fold_checkpoint_path(output_dir, row["scenario"], row["fold_id"])
                write_json(checkpoint, row)
                new_rows.append(row)
                if progress:
                    print(
                        f"[PLCC eval {index}/{len(pending_jobs)}] {row['scenario']}/{row['fold_id']}: ok "
                        f"({row['elapsed_wall_time_s']:.2f}s, pid={row['worker_pid']})",
                        flush=True,
                    )
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=resolved_workers,
                mp_context=context,
                initializer=_init_plcc_evaluation_worker,
                initargs=(
                    str(status_csv),
                    str(gt_csv),
                    str(dataset),
                    str(frozen_dir),
                    branch_name,
                    int(evaluation_batch_size),
                ),
            ) as executor:
                futures = {executor.submit(_evaluate_one_plcc_fold, job): job for job in pending_jobs}
                done = 0
                for future in as_completed(futures):
                    row = future.result()
                    if row["status"] != "ok":
                        for other in futures:
                            other.cancel()
                        raise RuntimeError(
                            f"PLCC held-out evaluation failed for {row['scenario']}/{row['fold_id']}: {row['error']}"
                        )
                    checkpoint = _fold_checkpoint_path(output_dir, row["scenario"], row["fold_id"])
                    write_json(checkpoint, row)
                    new_rows.append(row)
                    done += 1
                    if progress:
                        print(
                            f"[PLCC eval {done}/{len(pending_jobs)}] {row['scenario']}/{row['fold_id']}: ok "
                            f"({row['elapsed_wall_time_s']:.2f}s, pid={row['worker_pid']})",
                            flush=True,
                        )

    elapsed = time.perf_counter() - eval_start
    fold_rows = completed_rows + new_rows
    fold_rows.sort(key=lambda row: (str(row["scenario"]), str(row["fold_id"])))
    if len(fold_rows) != len(jobs):
        raise RuntimeError(f"Incomplete PLCC evaluation: expected {len(jobs)} folds, got {len(fold_rows)}.")

    scenario_rows: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in fold_rows}):
        rows = [row for row in fold_rows if str(row["scenario"]) == scenario]
        total = sum(int(row["gt_gesture_frames"]) for row in rows)
        covered = sum(int(row["complete_input_frames"]) for row in rows)
        errors = sum(int(row["final_errors_given_input"]) for row in rows)
        scenario_rows.append(
            {
                "scenario": scenario,
                "fold_count": len(rows),
                "gt_gesture_frames": total,
                "complete_input_frames": covered,
                "feature_coverage": covered / total if total else float("nan"),
                "final_conditional_accuracy": 1.0 - errors / covered if covered else float("nan"),
                "final_errors_given_input": errors,
            }
        )

    _write_csv(output_dir / "fold_summary.csv", fold_rows)
    _write_csv(output_dir / "scenario_summary.csv", scenario_rows)
    report = {
        "schema_version": "tsgrf_frozen_routing_generalization_plcc_parallel_v42_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_model_dir": str(frozen_dir),
        "test_processing_report": str(proc),
        "ground_truth_report": str(gtroot),
        "fold_count": len(fold_rows),
        "requested_evaluation_workers": int(evaluation_workers),
        "resolved_evaluation_workers": int(resolved_workers),
        "evaluation_batch_size": int(evaluation_batch_size),
        "elapsed_wall_time_s": elapsed,
        "serial_equivalent_sum_fold_time_s": sum(float(row["elapsed_wall_time_s"]) for row in new_rows),
        "estimated_parallel_speedup": (
            sum(float(row["elapsed_wall_time_s"]) for row in new_rows) / elapsed
            if elapsed > 0.0 and new_rows
            else 0.0
        ),
        "resumed_fold_count": len(completed_rows),
        "new_fold_count": len(new_rows),
        "summary_only": True,
        "notes": [
            "Fold evaluations are process-parallel and independent.",
            "Prediction equations are identical to the frozen routing evaluator; valid frames are evaluated in bounded NumPy batches.",
            "This PLCC path intentionally omits frame-level CSVs, problem-case images and workbooks because the sensitivity summary only consumes fold/scenario sufficient statistics.",
            "Per-fold JSON checkpoints make interrupted runs resumable without repeating completed held-out folds.",
        ],
    }
    write_json(output_dir / "parallel_evaluation_report.json", report)
    return {"fold_rows": fold_rows, "scenario_rows": scenario_rows, "report": report}


def _threshold_complete(variant_root: Path) -> bool:
    required = (
        variant_root / "threshold_complete.json",
        variant_root / "frozen_models" / "frozen_model_build_report.json",
        variant_root / "heldout_ranking" / "scenario_summary.csv",
    )
    return all(path.is_file() for path in required)


def run_frozen_plcc_sensitivity(
    experiment_plan_dir: str | Path,
    *,
    config: dict[str, Any],
    test_processing_report: str | Path,
    ground_truth_report: str | Path,
    output_dir: str | Path,
    branch_name: str,
    thresholds: Iterable[float],
    scenarios: set[str],
    model_workers: int = 0,
    evaluation_workers: int = 0,
    evaluation_batch_size: int = 64,
    dataset_root_override: str | Path | None = None,
    resume: bool = False,
    progress: bool = True,
) -> Path:
    plan = Path(experiment_plan_dir)
    out = Path(output_dir)
    if out.exists() and not resume:
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    values = tuple(float(v) for v in thresholds)
    if not values or any(v <= 0.0 or v > 1.0 for v in values):
        raise ValueError("PLCC thresholds must be a non-empty sequence in (0,1].")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be a positive integer.")

    build_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for i, threshold in enumerate(values, start=1):
        variant = variant_name(threshold)
        variant_root = out / variant
        frozen_dir = variant_root / "frozen_models"
        eval_dir = variant_root / "heldout_ranking"

        if resume and _threshold_complete(variant_root):
            if progress:
                print(
                    f"[frozen PLCC {i}/{len(values)}] threshold={threshold:.3f} -> {variant}: COMPLETE, reusing",
                    flush=True,
                )
            checkpoint = _read_json(variant_root / "threshold_complete.json")
            build_rows.append(dict(checkpoint["build_summary"]))
            for row in read_csv_rows(frozen_dir / "frozen_training_loo_summary.csv"):
                item = dict(row)
                item["plcc_threshold"] = threshold
                item["variant"] = variant
                item["phase"] = "TRAIN_LOO"
                fold_rows.append(item)
            for row in read_csv_rows(eval_dir / "scenario_summary.csv"):
                item = dict(row)
                item["plcc_threshold"] = threshold
                item["variant"] = variant
                scenario_rows.append(item)
            continue

        if progress:
            print(f"[frozen PLCC {i}/{len(values)}] threshold={threshold:.3f} -> {variant}", flush=True)

        variant_root.mkdir(parents=True, exist_ok=True)
        reference_config = copy.deepcopy(config["reference_models"])
        reference_config.setdefault("compact_feature_mask", {})["correlation_threshold"] = threshold
        build = build_experiment_models_parallel(
            plan,
            branches=[branch_name],
            reference_config=reference_config,
            scenarios=scenarios,
            folds=None,
            force=False,
            workers=int(model_workers),
            progress=progress,
            model_variant=variant,
        )
        if build["failed"]:
            raise RuntimeError(f"Reference-model build failed for PLCC {threshold:.3f}")

        # The TRAIN-only model build is deterministic. Reuse a complete build after
        # an interrupted run and delete only an incomplete PLCC-local build directory.
        frozen_report = frozen_dir / "frozen_model_build_report.json"
        if not frozen_report.is_file():
            if frozen_dir.exists():
                shutil.rmtree(frozen_dir)
            build_frozen_routing_models(
                plan,
                output_dir=frozen_dir,
                branch_name=branch_name,
                feature_set="compact",
                orientation_mode="camera_aware",
                scenarios=scenarios,
                folds=None,
                os_alpha=0.65,
                c_top_n=5,
                model_variant=variant,
                dataset_root_override=dataset_root_override,
                progress=progress,
            )
        elif progress and resume:
            print(f"[{variant}] reusing complete TRAIN-only frozen model build", flush=True)

        eval_result = _evaluate_frozen_routing_plcc_parallel(
            plan,
            frozen_dir=frozen_dir,
            test_processing_report=test_processing_report,
            ground_truth_report=ground_truth_report,
            output_dir=eval_dir,
            branch_name=branch_name,
            scenarios=scenarios,
            evaluation_workers=int(evaluation_workers),
            evaluation_batch_size=int(evaluation_batch_size),
            dataset_root_override=dataset_root_override,
            resume=resume,
            progress=progress,
        )

        build_summary = {
            "plcc_threshold": threshold,
            "variant": variant,
            "reference_models_built": int(build["built"]),
            "reference_models_skipped": int(build["skipped"]),
            "reference_models_failed": int(build["failed"]),
            "reference_model_build_wall_time_s": float(build["elapsed_wall_time_s"]),
            "resolved_model_workers": int(build["resolved_workers"]),
            "requested_evaluation_workers": int(evaluation_workers),
            "resolved_evaluation_workers": int(eval_result["report"]["resolved_evaluation_workers"]),
            "evaluation_batch_size": int(evaluation_batch_size),
            "heldout_evaluation_wall_time_s": float(eval_result["report"]["elapsed_wall_time_s"]),
            "heldout_resumed_fold_count": int(eval_result["report"]["resumed_fold_count"]),
        }
        build_rows.append(build_summary)

        for row in read_csv_rows(frozen_dir / "frozen_training_loo_summary.csv"):
            item = dict(row)
            item["plcc_threshold"] = threshold
            item["variant"] = variant
            item["phase"] = "TRAIN_LOO"
            fold_rows.append(item)
        for row in eval_result["scenario_rows"]:
            item = dict(row)
            item["plcc_threshold"] = threshold
            item["variant"] = variant
            scenario_rows.append(item)

        write_json(
            variant_root / "threshold_complete.json",
            {
                "schema_version": "tsgrf_frozen_plcc_threshold_checkpoint_v42_1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plcc_threshold": threshold,
                "variant": variant,
                "build_summary": build_summary,
                "status": "complete",
            },
        )

    aggregate_rows: list[dict[str, Any]] = []
    for threshold in values:
        rows = [r for r in scenario_rows if abs(float(r["plcc_threshold"]) - threshold) <= 1e-12]
        train = [r for r in fold_rows if abs(float(r["plcc_threshold"]) - threshold) <= 1e-12]
        aggregate_rows.append(
            {
                "plcc_threshold": threshold,
                "scenario_count": len(rows),
                "fold_count": len(train),
                "heldout_final_conditional_accuracy_mean_across_scenarios": float(
                    np.mean([float(r["final_conditional_accuracy"]) for r in rows])
                )
                if rows
                else float("nan"),
                "heldout_feature_coverage_mean_across_scenarios": float(
                    np.mean([float(r["feature_coverage"]) for r in rows])
                )
                if rows
                else float("nan"),
                "train_final_loo_accuracy_mean_across_folds": float(
                    np.mean([float(r["final_train_loo_accuracy"]) for r in train])
                )
                if train
                else float("nan"),
                "post_hoc_exploratory": 1,
                "eligible_for_final_model_reselection": 0,
            }
        )

    _write_csv(out / "plcc_build_summary.csv", build_rows)
    _write_csv(out / "plcc_train_fold_summary.csv", fold_rows)
    _write_csv(out / "plcc_heldout_scenario_summary.csv", scenario_rows)
    _write_csv(out / "plcc_sensitivity_summary.csv", aggregate_rows)
    write_workbook(
        out / "frozen_plcc_sensitivity.xlsx",
        {
            "summary": (list(aggregate_rows[0]) if aggregate_rows else [], aggregate_rows),
            "heldout_scenarios": (list(scenario_rows[0]) if scenario_rows else [], scenario_rows),
            "train_folds": (list(fold_rows[0]) if fold_rows else [], fold_rows),
            "build": (list(build_rows[0]) if build_rows else [], build_rows),
        },
    )
    write_json(
        out / "frozen_plcc_sensitivity_report.json",
        {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": str(plan),
            "thresholds": list(values),
            "scenarios": sorted(scenarios),
            "branch_name": branch_name,
            "test_processing_report": str(Path(test_processing_report)),
            "ground_truth_report": str(Path(ground_truth_report)),
            "requested_model_workers": int(model_workers),
            "requested_evaluation_workers": int(evaluation_workers),
            "evaluation_batch_size": int(evaluation_batch_size),
            "resume_enabled": bool(resume),
            "post_hoc_exploratory": True,
            "eligible_for_final_model_reselection": False,
            "notes": [
                "Alternative PLCC thresholds are a sensitivity/efficiency analysis only.",
                "All model fitting remains TRAIN-only; post-hoc sensitivity results must not be used to redefine the fixed final method.",
                "MediaPipe is not re-run; the existing cached test-processing report is reused.",
                "Independent held-out folds may be evaluated concurrently in spawned worker processes using bounded vectorized inference batches.",
                "Resume checkpoints are operational only and do not alter mathematical outputs.",
            ],
        },
    )
    return out / "frozen_plcc_sensitivity_report.json"
