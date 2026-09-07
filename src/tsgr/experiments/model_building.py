"""Parallel orchestration for independent experiment-fold reference-model builds."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tsgr.reference_models.builder import build_reference_models
from tsgr.reference_models.manifest import read_reference_manifest
from tsgr.utils.serialization import write_json


@dataclass(frozen=True, slots=True)
class ModelBuildJob:
    scenario: str
    fold_id: str
    fold_dir: Path
    output_root: Path


def _portable_plan_path(path: Path, plan_dir: Path) -> str:
    """Serialize a plan artifact path without leaking a machine-specific root."""
    try:
        return path.resolve().relative_to(plan_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_model_build_worker_count(requested_workers: int, job_count: int) -> int:
    """Resolve a bounded process count for memory-heavy model aggregation."""
    if requested_workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if job_count <= 0:
        return 0
    if requested_workers > 0:
        return min(requested_workers, job_count)
    cpu_count = os.cpu_count() or 1
    automatic = max(1, min(4, max(1, cpu_count // 2)))
    return min(automatic, job_count)


def discover_model_build_jobs(
    experiment_plan_dir: Path,
    *,
    scenario_filter: set[str] | None = None,
    fold_filter: set[str] | None = None,
    force: bool = False,
    expected_branches: Iterable[str] | None = None,
    expected_compact_correlation_threshold: float | None = None,
    model_variant: str | None = None,
) -> tuple[list[ModelBuildJob], list[dict[str, str]]]:
    """Return build jobs plus deterministic skip records."""
    jobs: list[ModelBuildJob] = []
    skipped: list[dict[str, str]] = []
    scenario_filter = scenario_filter or set()
    fold_filter = fold_filter or set()
    expected_branch_names = tuple(dict.fromkeys(str(branch) for branch in (expected_branches or [])))
    expected_threshold = (
        None
        if expected_compact_correlation_threshold is None
        else float(expected_compact_correlation_threshold)
    )
    for fold_json in sorted(experiment_plan_dir.glob("*/*/fold.json")):
        fold_dir = fold_json.parent
        payload = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(payload["scenario"])
        fold_id = str(payload["fold_id"])
        if scenario_filter and scenario not in scenario_filter:
            continue
        if fold_filter and fold_id not in fold_filter:
            continue
        if not payload.get("ready_for_model_building", False):
            skipped.append(
                {
                    "scenario": scenario,
                    "fold_id": fold_id,
                    "reason": "training_cells_not_processed",
                    "location": _portable_plan_path(fold_dir, experiment_plan_dir),
                }
            )
            continue
        output_root = (
            fold_dir / "reference_models"
            if not model_variant
            else fold_dir / "reference_model_variants" / model_variant
        )
        if output_root.exists() and any(output_root.iterdir()):
            if not force:
                model_set_files = sorted(output_root.glob("*/model_set.json"))
                location = _portable_plan_path(model_set_files[-1].parent if model_set_files else output_root, experiment_plan_dir)
                if expected_branch_names or expected_threshold is not None:
                    if len(model_set_files) != 1:
                        raise ValueError(
                            f"Existing reference_models for {scenario}/{fold_id} cannot be validated: "
                            f"expected exactly one model_set.json, found {len(model_set_files)}. "
                            "Use --force only if replacing the existing model set is intentional."
                        )
                    existing = json.loads(model_set_files[0].read_text(encoding="utf-8"))
                    existing_branches = tuple(str(value) for value in existing.get("branch_names", []))
                    existing_threshold_raw = existing.get("compact_correlation_threshold")
                    existing_threshold = (
                        None if existing_threshold_raw is None else float(existing_threshold_raw)
                    )
                    branches_match = not expected_branch_names or existing_branches == expected_branch_names
                    threshold_match = (
                        expected_threshold is None
                        or (existing_threshold is not None and abs(existing_threshold - expected_threshold) <= 1.0e-12)
                    )
                    if not branches_match or not threshold_match:
                        raise ValueError(
                            f"Existing model set for {scenario}/{fold_id} was built with a different configuration: "
                            f"branches={list(existing_branches)}, compact_correlation_threshold={existing_threshold}; "
                            f"requested branches={list(expected_branch_names)}, "
                            f"compact_correlation_threshold={expected_threshold}. "
                            "Refusing to reuse it. Use --force only if overwriting is intentional, or use a fresh "
                            "experiment-plan/output tree when several variants must be preserved."
                        )
                skipped.append(
                    {
                        "scenario": scenario,
                        "fold_id": fold_id,
                        "reason": "models_already_exist_compatible",
                        "location": location,
                    }
                )
                continue
            shutil.rmtree(output_root)
        jobs.append(ModelBuildJob(scenario, fold_id, fold_dir, output_root))
    return jobs, skipped


def _build_one_job(job: ModelBuildJob, branches: list[str], reference_config: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = build_reference_models(
            read_reference_manifest(job.fold_dir / "reference_manifest.csv"),
            branches=branches,
            output_root=job.output_root,
            config=reference_config,
            model_set_name=f"models_{job.scenario}_{job.fold_id}",
        )
        return {
            "status": "ok",
            "scenario": job.scenario,
            "fold_id": job.fold_id,
            "output_dir": result.output_dir.relative_to(job.fold_dir).as_posix(),
            "worker_pid": os.getpid(),
            "elapsed_wall_time_s": time.perf_counter() - start,
            "error": "",
        }
    except BaseException as error:
        return {
            "status": "error",
            "scenario": job.scenario,
            "fold_id": job.fold_id,
            "output_dir": "",
            "worker_pid": os.getpid(),
            "elapsed_wall_time_s": time.perf_counter() - start,
            "error": f"{type(error).__name__}: {error}",
        }


def build_experiment_models_parallel(
    experiment_plan_dir: str | Path,
    *,
    branches: Iterable[str],
    reference_config: dict[str, Any],
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    force: bool = False,
    workers: int = 0,
    progress: bool = True,
    model_variant: str | None = None,
) -> dict[str, Any]:
    """Build independent fold model sets in bounded worker processes."""
    plan_dir = Path(experiment_plan_dir)
    if not (plan_dir / "experiment_plan.json").is_file():
        raise ValueError(f"Not an experiment plan directory: {plan_dir}")
    branches_list = list(branches)
    compact_correlation_threshold = float(
        reference_config.get("compact_feature_mask", {}).get("correlation_threshold", 0.995)
    )
    jobs, skipped = discover_model_build_jobs(
        plan_dir,
        scenario_filter=scenarios,
        fold_filter=folds,
        force=force,
        expected_branches=branches_list,
        expected_compact_correlation_threshold=compact_correlation_threshold,
        model_variant=model_variant,
    )
    resolved_workers = resolve_model_build_worker_count(workers, len(jobs))
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    if jobs:
        if resolved_workers <= 1:
            for index, job in enumerate(jobs, start=1):
                result = _build_one_job(job, branches_list, reference_config)
                results.append(result)
                if progress:
                    print(
                        f"[{index}/{len(jobs)}] {job.scenario}/{job.fold_id}: {result['status']} "
                        f"({result['elapsed_wall_time_s']:.2f}s)",
                        flush=True,
                    )
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=resolved_workers, mp_context=context) as executor:
                futures = {
                    executor.submit(_build_one_job, job, branches_list, reference_config): job
                    for job in jobs
                }
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if progress:
                        print(
                            f"[{completed}/{len(jobs)}] {result['scenario']}/{result['fold_id']}: "
                            f"{result['status']} ({result['elapsed_wall_time_s']:.2f}s, pid={result['worker_pid']})",
                            flush=True,
                        )
    elapsed = time.perf_counter() - start
    results.sort(key=lambda row: (row["scenario"], row["fold_id"]))
    built = sum(row["status"] == "ok" for row in results)
    failed = sum(row["status"] == "error" for row in results)
    payload = {
        "experiment_plan": plan_dir.name,
        "dataset_contract": "tsgr_public_dataset_v1",
        "requested_branches": branches_list,
        "requested_scenarios": sorted(scenarios or set()),
        "requested_folds": sorted(folds or set()),
        "compact_correlation_threshold": compact_correlation_threshold,
        "force": bool(force),
        "model_variant": model_variant or "reference",
        "requested_workers": workers,
        "resolved_workers": resolved_workers,
        "job_count": len(jobs),
        "built": built,
        "skipped": len(skipped),
        "failed": failed,
        "elapsed_wall_time_s": elapsed,
        "serial_equivalent_sum_job_time_s": sum(float(row["elapsed_wall_time_s"]) for row in results),
        "estimated_parallel_speedup": (
            sum(float(row["elapsed_wall_time_s"]) for row in results) / elapsed if elapsed > 0 else 0.0
        ),
        "results": results,
        "skip_records": skipped,
    }
    performance_name = (
        "model_build_performance.json"
        if not model_variant
        else f"model_build_performance__{model_variant}.json"
    )
    write_json(plan_dir / performance_name, payload)
    return payload
