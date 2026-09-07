"""Fold-local training orchestration for the thirteen comparable SGRF methods."""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tsgr.dataset.contract import resolve_dataset_root_from_plan

from tsgr.baselines.registry import SGRFMethodSpec, normalize_method_selection, registry_rows
from tsgr.baselines.sgrf_bridge import audit_sgrf_environment, payload_sha256, run_worker, write_job
from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows
from tsgr.utils.serialization import write_json


@dataclass(frozen=True, slots=True)
class BaselineFold:
    scenario: str
    fold_id: str
    fold_dir: Path
    dataset_root: Path
    training_rows: tuple[dict[str, str], ...]


def resolve_external_worker_count(requested_workers: int, job_count: int) -> int:
    """Resolve concurrency of heavyweight external SGRF subprocesses.

    Automatic mode is deliberately conservative because several SGRF methods use
    TensorFlow/Keras and can consume substantial RAM.  Explicit values remain
    available for a dedicated benchmark workstation.
    """
    if requested_workers < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer.")
    if job_count <= 0:
        return 0
    if requested_workers > 0:
        return min(requested_workers, job_count)
    return min(1, job_count)


def _discover_folds(
    plan_dir: Path,
    *,
    scenarios: set[str] | None,
    folds: set[str] | None,
    dataset_root_override: str | Path | None = None,
) -> list[BaselineFold]:
    plan_meta_path = plan_dir / "experiment_plan.json"
    if not plan_meta_path.is_file():
        raise FileNotFoundError(f"Missing experiment_plan.json: {plan_meta_path}")
    plan_meta = json.loads(plan_meta_path.read_text(encoding="utf-8"))
    dataset_root = resolve_dataset_root_from_plan(plan_dir, override=dataset_root_override)
    result: list[BaselineFold] = []
    for fold_json in sorted(plan_dir.glob("*/*/fold.json")):
        payload = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(payload["scenario"])
        fold_id = str(payload["fold_id"])
        if scenarios and scenario not in scenarios:
            continue
        if folds and fold_id not in folds:
            continue
        if str(payload.get("fold_status", "active")) != "active":
            continue
        rows = read_csv_rows(fold_json.parent / "training_images.csv")
        if not rows:
            continue
        result.append(
            BaselineFold(
                scenario=scenario,
                fold_id=fold_id,
                fold_dir=fold_json.parent,
                dataset_root=dataset_root,
                training_rows=tuple(rows),
            )
        )
    if not result:
        raise ValueError("No active experiment folds with training_images.csv matched the requested scope.")
    return result


def _resolved_training_rows(fold: BaselineFold) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in fold.training_rows:
        relative = Path(str(row["relative_path"]).replace("\\", "/"))
        image_path = fold.dataset_root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"Canonical training image is missing: {image_path}")
        output.append(
            {
                "image_id": row.get("image_id", ""),
                "gesture_id": str(row["gesture_id"]).upper(),
                "public_subject_id": row.get("public_subject_id", ""),
                "background": row.get("background", ""),
                "image_path": str(image_path.resolve()),
                "relative_path": relative.as_posix(),
                "width": row.get("width", ""),
                "height": row.get("height", ""),
                "sha256": row.get("sha256", ""),
            }
        )
    return output


def _training_fingerprint(rows: list[dict[str, Any]], method: SGRFMethodSpec, options: dict[str, Any], seed: int) -> str:
    identity_rows = [
        {
            "image_id": row.get("image_id", ""),
            "gesture_id": row["gesture_id"],
            "sha256": row.get("sha256", ""),
            "relative_path": row.get("relative_path", ""),
        }
        for row in rows
    ]
    return payload_sha256(
        {
            "method": method.to_dict(),
            "training_rows": identity_rows,
            "custom_options": options,
            "seed": seed,
            "gestures": sorted({row["gesture_id"] for row in rows}),
            "adapter_schema": "tsgr_sgrf_training_v1",
        }
    )


def _load_method_options(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Method-options JSON must contain an object keyed by SGRF method id.")
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Options for {key} must be a JSON object.")
        result[str(key).upper()] = value
    return result


def _prepare_job(
    output_root: Path,
    fold: BaselineFold,
    method: SGRFMethodSpec,
    *,
    method_options: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], Path, str]:
    job_dir = output_root / "models" / fold.scenario / fold.fold_id / method.method_id
    job_dir.mkdir(parents=True, exist_ok=True)
    model_dir = job_dir / "model"
    rows = _resolved_training_rows(fold)
    gestures = sorted({str(row["gesture_id"]).upper() for row in rows})
    options = dict(method_options.get(method.method_id, {}))
    manifest_path = job_dir / "training_manifest.csv"
    fields = ["image_id", "gesture_id", "public_subject_id", "background", "relative_path", "image_path", "width", "height", "sha256"]
    write_csv_rows(manifest_path, rows, fields)
    fingerprint = _training_fingerprint(rows, method, options, seed)
    job = {
        "schema_version": "tsgr_sgrf_training_job_v1",
        "method_id": method.method_id,
        "method_display_name": method.display_name,
        "scenario": fold.scenario,
        "fold_id": fold.fold_id,
        "training_manifest": str(manifest_path.resolve()),
        "model_dir": str(model_dir.resolve()),
        "gestures": gestures,
        "coordinate_policy": method.coordinate_policy,
        "custom_options": options,
        "seed": seed,
        "training_fingerprint": fingerprint,
        "input_contract": "canonical_raw_image; no TSGRF landmarks or detected hand ROI",
    }
    job_path = job_dir / "train_job.json"
    write_job(job_path, job)
    return job, job_dir, fingerprint


def _run_training_job(
    sgrf_python: str | Path,
    job: dict[str, Any],
    job_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    result_path = job_dir / "train_result.json"
    started = time.perf_counter()
    result = run_worker(
        sgrf_python,
        "train",
        job_json=job_dir / "train_job.json",
        result_json=result_path,
        log_path=job_dir / "train.log",
    )
    manifest = {
        "schema_version": "tsgr_sgrf_baseline_model_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": job["method_id"],
        "method_display_name": job["method_display_name"],
        "scenario": job["scenario"],
        "fold_id": job["fold_id"],
        "training_fingerprint": fingerprint,
        "training_sample_count": result.get("training_sample_count", 0),
        "training_accuracy_internal": result.get("accuracy"),
        "training_loss_internal": result.get("loss"),
        "training_elapsed_s": result.get("training_elapsed_s"),
        "bridge_elapsed_s": result.get("bridge_elapsed_s"),
        "coordinate_policy": job["coordinate_policy"],
        "custom_options": job["custom_options"],
        "gestures": job["gestures"],
        "model_dir": job["model_dir"],
        "model_dir_relative": "model",
        "model_files": result.get("model_files", []),
        "external_environment": result.get("environment", {}),
        "input_contract": job["input_contract"],
        "elapsed_wall_time_s": time.perf_counter() - started,
    }
    write_json(job_dir / "baseline_model.json", manifest)
    return manifest


def build_sgrf_baseline_models(
    experiment_plan_dir: str | Path,
    *,
    sgrf_python: str | Path,
    output_dir: str | Path,
    methods: Iterable[str] | None = None,
    all_methods: bool = False,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    dataset_root_override: str | Path | None = None,
    method_options_json: str | Path | None = None,
    seed: int = 2026,
    workers: int = 1,
    force: bool = False,
    fail_fast: bool = False,
    progress: bool = True,
    allow_nonreference_python: bool = False,
    allow_nonreference_sgrf: bool = False,
) -> Path:
    """Train selected upstream SGRF algorithms independently for every experiment fold."""
    plan_dir = Path(experiment_plan_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_methods = normalize_method_selection(methods, all_methods=all_methods)
    selected_folds = _discover_folds(plan_dir, scenarios=scenarios, folds=folds, dataset_root_override=dataset_root_override)
    method_options = _load_method_options(method_options_json)
    audit = audit_sgrf_environment(
        sgrf_python,
        output_dir=output_root / "environment",
        allow_nonreference_python=allow_nonreference_python,
        allow_nonreference_sgrf=allow_nonreference_sgrf,
    )
    write_csv_rows(
        output_root / "method_registry.csv",
        registry_rows(),
        ["method_id", "display_name", "sort_order", "payload_kind", "learning_data_kind", "coordinate_policy", "certainty_scale", "included", "exclusion_reason"],
    )
    jobs: list[tuple[dict[str, Any], Path, str]] = []
    skipped: list[dict[str, Any]] = []
    for fold in selected_folds:
        for method in selected_methods:
            job, job_dir, fingerprint = _prepare_job(
                output_root,
                fold,
                method,
                method_options=method_options,
                seed=seed,
            )
            manifest_path = job_dir / "baseline_model.json"
            if manifest_path.is_file() and not force:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if str(existing.get("training_fingerprint", "")) != fingerprint:
                    raise RuntimeError(
                        f"Existing SGRF model does not match the requested training data/options: {manifest_path}. "
                        "Use --force only when replacement is intentional."
                    )
                skipped.append(
                    {
                        "scenario": fold.scenario,
                        "fold_id": fold.fold_id,
                        "method_id": method.method_id,
                        "status": "compatible_existing_model",
                        "model_manifest": manifest_path.relative_to(output_root).as_posix(),
                    }
                )
                continue
            if force and (job_dir / "model").exists():
                import shutil
                shutil.rmtree(job_dir / "model", ignore_errors=True)
            jobs.append((job, job_dir, fingerprint))

    resolved_workers = resolve_external_worker_count(workers, len(jobs))
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def execute(item: tuple[dict[str, Any], Path, str]) -> dict[str, Any]:
        job, job_dir, fingerprint = item
        try:
            manifest = _run_training_job(sgrf_python, job, job_dir, fingerprint)
            return {
                "scenario": job["scenario"],
                "fold_id": job["fold_id"],
                "method_id": job["method_id"],
                "status": "ok",
                "model_manifest": (job_dir / "baseline_model.json").relative_to(output_root).as_posix(),
                "training_elapsed_s": manifest.get("training_elapsed_s"),
            }
        except BaseException as exc:
            return {
                "scenario": job["scenario"],
                "fold_id": job["fold_id"],
                "method_id": job["method_id"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "model_manifest": "",
            }

    if resolved_workers <= 1:
        for index, item in enumerate(jobs, start=1):
            row = execute(item)
            results.append(row)
            if row["status"] == "error":
                failures.append(row)
                if fail_fast:
                    break
            if progress:
                print(f"[SGRF train {index}/{len(jobs)}] {row['scenario']}/{row['fold_id']} {row['method_id']}: {row['status']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            future_map = {executor.submit(execute, item): item for item in jobs}
            completed = 0
            for future in as_completed(future_map):
                row = future.result()
                results.append(row)
                completed += 1
                if row["status"] == "error":
                    failures.append(row)
                if progress:
                    print(f"[SGRF train {completed}/{len(jobs)}] {row['scenario']}/{row['fold_id']} {row['method_id']}: {row['status']}", flush=True)
                if fail_fast and failures:
                    for pending in future_map:
                        pending.cancel()
                    break

    all_rows = sorted([*results, *skipped], key=lambda row: (row["scenario"], row["fold_id"], row["method_id"]))
    fields = ["scenario", "fold_id", "method_id", "status", "model_manifest", "training_elapsed_s", "error"]
    write_csv_rows(output_root / "model_build_summary.csv", all_rows, fields)
    write_json(
        output_root / "model_build_summary.json",
        {
            "schema_version": "tsgr_sgrf_model_build_summary_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": plan_dir.name,
            "dataset_root": ".",
            "dataset_contract": "tsgr_public_dataset_v1",
            "output_dir": output_root.name,
            "methods": [item.method_id for item in selected_methods],
            "scenarios": sorted(scenarios or set()),
            "fold_filter": sorted(folds or set()),
            "seed": seed,
            "workers_requested": workers,
            "workers_resolved": resolved_workers,
            "job_count": len(jobs),
            "built_count": sum(row["status"] == "ok" for row in results),
            "skipped_count": len(skipped),
            "failure_count": len(failures),
            "elapsed_wall_time_s": time.perf_counter() - started,
            "external_environment": audit.payload,
            "method_options_json": Path(method_options_json).name if method_options_json else "",
            "input_contract": "Each SGRF method is trained from canonical raw training images. No TSGRF landmarks or detected hand ROI are injected.",
        },
    )
    if failures and fail_fast:
        raise RuntimeError(f"SGRF training stopped after {len(failures)} failure(s); see {output_root / 'model_build_summary.csv'}")
    return output_root
