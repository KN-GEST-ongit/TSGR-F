"""Distributed checkpoint tools for the external SGRF benchmark.

The SGRF comparison is intentionally checkpointed at method x fold granularity.
This module exposes small, integrity-checked shards containing only completed
prediction jobs so long-running scenario evaluations can be split across
machines and merged without copying aggregate reports or model binaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tsgr.baselines.registry import SGRFMethodSpec, normalize_method_selection
from tsgr.baselines.sgrf_portability import discover_active_folds
from tsgr.evaluation.io_utils import write_csv_rows
from tsgr.utils.serialization import write_json

SHARD_SCHEMA = "tsgr_sgrf_evaluation_shard_v1"
SHARD_ROOT = "sgrf_evaluation_shard"
CHECKPOINT_FILENAMES = ("prediction_run.json", "worker_predictions.csv")


@dataclass(frozen=True, slots=True)
class JobStatus:
    scenario: str
    fold_id: str
    method_id: str
    method_display_name: str
    method_sort_order: int
    status: str
    prediction_fingerprint: str
    worker_prediction_rows: int | str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_data_row_count(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _job_status(evaluation_dir: Path, scenario: str, fold_id: str, method: SGRFMethodSpec) -> JobStatus:
    job_dir = evaluation_dir / "jobs" / scenario / fold_id / method.method_id
    run = _safe_json(job_dir / "prediction_run.json")
    worker = job_dir / "worker_predictions.csv"
    complete = bool(
        run
        and worker.is_file()
        and worker.stat().st_size > 0
        and str(run.get("worker_status", "")).lower() != "error"
        and run.get("bridge_returncode", 0) in (0, "0", "", None)
        and str(run.get("prediction_fingerprint", ""))
    )
    return JobStatus(
        scenario=scenario,
        fold_id=fold_id,
        method_id=method.method_id,
        method_display_name=method.display_name,
        method_sort_order=method.sort_order,
        status="complete" if complete else "pending",
        prediction_fingerprint=str(run.get("prediction_fingerprint", "")) if run else "",
        worker_prediction_rows=_csv_data_row_count(worker) if complete else "",
    )


def collect_sgrf_progress(
    experiment_plan_dir: str | Path,
    *,
    evaluation_dir: str | Path,
    methods: Iterable[str] | None = None,
    all_methods: bool = False,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
) -> dict[str, Any]:
    """Return deterministic method/fold completion status for selected jobs."""
    selected_methods = normalize_method_selection(methods, all_methods=all_methods)
    selected_folds = discover_active_folds(experiment_plan_dir, scenarios=scenarios, folds=folds)
    evaluation = Path(evaluation_dir)
    jobs = [
        _job_status(evaluation, fold.scenario, fold.fold_id, method)
        for fold in selected_folds
        for method in selected_methods
    ]
    complete = [job for job in jobs if job.status == "complete"]
    by_method: list[dict[str, Any]] = []
    for method in selected_methods:
        rows = [job for job in jobs if job.method_id == method.method_id]
        done = sum(job.status == "complete" for job in rows)
        by_method.append(
            {
                "method_id": method.method_id,
                "method_display_name": method.display_name,
                "method_sort_order": method.sort_order,
                "complete_jobs": done,
                "expected_jobs": len(rows),
                "complete": int(done == len(rows) and bool(rows)),
            }
        )
    by_scenario: list[dict[str, Any]] = []
    for scenario in sorted({job.scenario for job in jobs}):
        rows = [job for job in jobs if job.scenario == scenario]
        done = sum(job.status == "complete" for job in rows)
        by_scenario.append(
            {
                "scenario": scenario,
                "complete_jobs": done,
                "expected_jobs": len(rows),
                "complete": int(done == len(rows) and bool(rows)),
            }
        )
    return {
        "schema_version": "tsgr_sgrf_evaluation_progress_v1",
        "expected_job_count": len(jobs),
        "complete_job_count": len(complete),
        "pending_job_count": len(jobs) - len(complete),
        "jobs": [asdict(job) for job in jobs],
        "by_method": by_method,
        "by_scenario": by_scenario,
    }


def write_sgrf_progress_report(progress: dict[str, Any], output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "progress_report.json", {key: value for key, value in progress.items() if key != "jobs"})
    write_csv_rows(
        target / "progress_jobs.csv",
        list(progress["jobs"]),
        [
            "scenario", "fold_id", "method_id", "method_display_name", "method_sort_order", "status",
            "prediction_fingerprint", "worker_prediction_rows",
        ],
    )
    write_csv_rows(
        target / "progress_by_method.csv",
        list(progress["by_method"]),
        ["method_id", "method_display_name", "method_sort_order", "complete_jobs", "expected_jobs", "complete"],
    )
    write_csv_rows(
        target / "progress_by_scenario.csv",
        list(progress["by_scenario"]),
        ["scenario", "complete_jobs", "expected_jobs", "complete"],
    )
    return target


def export_sgrf_evaluation_shard(
    experiment_plan_dir: str | Path,
    *,
    evaluation_dir: str | Path,
    output_zip: str | Path,
    methods: Iterable[str] | None = None,
    all_methods: bool = False,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Export only completed method x fold checkpoints into an integrity-checked ZIP."""
    evaluation = Path(evaluation_dir)
    target_zip = Path(output_zip)
    if target_zip.exists() and not overwrite:
        raise FileExistsError(f"Shard already exists: {target_zip}")
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    progress = collect_sgrf_progress(
        experiment_plan_dir,
        evaluation_dir=evaluation,
        methods=methods,
        all_methods=all_methods,
        scenarios=scenarios,
        folds=folds,
    )
    complete_jobs = [job for job in progress["jobs"] if job["status"] == "complete"]
    if not complete_jobs:
        raise RuntimeError("No completed SGRF jobs are available in the requested scope.")

    with tempfile.TemporaryDirectory(prefix="tsgr_sgrf_shard_") as temp_name:
        root = Path(temp_name) / SHARD_ROOT
        root.mkdir(parents=True)
        job_manifest: list[dict[str, Any]] = []
        for job in complete_jobs:
            relative_job = Path("jobs") / job["scenario"] / job["fold_id"] / job["method_id"]
            source_dir = evaluation / relative_job
            target_dir = root / relative_job
            target_dir.mkdir(parents=True, exist_ok=True)
            file_records: list[dict[str, Any]] = []
            for filename in CHECKPOINT_FILENAMES:
                source = source_dir / filename
                if not source.is_file():
                    raise RuntimeError(f"Completed job lost required checkpoint file: {source}")
                destination = target_dir / filename
                shutil.copy2(source, destination)
                file_records.append(
                    {
                        "filename": filename,
                        "size_bytes": destination.stat().st_size,
                        "sha256": _sha256(destination),
                    }
                )
            job_manifest.append(
                {
                    "scenario": job["scenario"],
                    "fold_id": job["fold_id"],
                    "method_id": job["method_id"],
                    "prediction_fingerprint": job["prediction_fingerprint"],
                    "worker_prediction_rows": job["worker_prediction_rows"],
                    "files": file_records,
                }
            )

        write_csv_rows(
            root / "progress_summary.csv",
            list(progress["by_method"]),
            ["method_id", "method_display_name", "method_sort_order", "complete_jobs", "expected_jobs", "complete"],
        )
        write_json(
            root / "shard_manifest.json",
            {
                "schema_version": SHARD_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_plan": Path(experiment_plan_dir).name,
                "complete_job_count": len(complete_jobs),
                "expected_job_count": progress["expected_job_count"],
                "jobs": job_manifest,
            },
        )
        checksum_lines: list[str] = []
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
        (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        if target_zip.exists():
            target_zip.unlink()
        with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                archive.write(path, arcname=(Path(SHARD_ROOT) / path.relative_to(root)).as_posix())
    return target_zip


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise RuntimeError("SGRF shard is missing checksums.sha256.")
    for raw in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = root / Path(relative)
        if not path.is_file():
            raise RuntimeError(f"SGRF shard checksum target is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"SGRF shard checksum mismatch for {relative}: {actual} != {expected}")


def import_sgrf_evaluation_shard(
    shard_zip: str | Path,
    *,
    evaluation_dir: str | Path,
) -> dict[str, Any]:
    """Import completed checkpoints idempotently; conflicting jobs are rejected."""
    source_zip = Path(shard_zip)
    evaluation = Path(evaluation_dir)
    evaluation.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="tsgr_sgrf_shard_import_") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(temp)
        root = temp / SHARD_ROOT
        manifest_path = root / "shard_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Not a TSGR-F SGRF shard: {source_zip}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARD_SCHEMA:
            raise RuntimeError(f"Unsupported SGRF shard schema: {manifest.get('schema_version')!r}")
        _verify_checksums(root)
        for job in manifest.get("jobs", []):
            scenario = str(job["scenario"])
            fold_id = str(job["fold_id"])
            method_id = str(job["method_id"])
            source_dir = root / "jobs" / scenario / fold_id / method_id
            destination = evaluation / "jobs" / scenario / fold_id / method_id
            source_run = json.loads((source_dir / "prediction_run.json").read_text(encoding="utf-8"))
            source_fp = str(source_run.get("prediction_fingerprint", ""))
            if source_fp != str(job.get("prediction_fingerprint", "")) or not source_fp:
                raise RuntimeError(f"Shard fingerprint mismatch for {scenario}/{fold_id}/{method_id}.")
            destination_run = destination / "prediction_run.json"
            destination_worker = destination / "worker_predictions.csv"
            if destination_run.is_file() or destination_worker.is_file():
                if not (destination_run.is_file() and destination_worker.is_file()):
                    raise RuntimeError(f"Partial existing checkpoint conflicts with shard: {destination}")
                existing_run = json.loads(destination_run.read_text(encoding="utf-8"))
                existing_fp = str(existing_run.get("prediction_fingerprint", ""))
                # Runtime metadata in prediction_run.json contains machine-specific
                # paths and timings. A duplicate checkpoint is equivalent when the
                # scientific fingerprint and worker predictions are identical.
                same = (
                    existing_fp == source_fp
                    and _sha256(destination_worker) == _sha256(source_dir / "worker_predictions.csv")
                )
                if not same:
                    raise RuntimeError(f"Existing SGRF checkpoint conflicts with imported shard: {destination}")
                skipped += 1
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for filename in CHECKPOINT_FILENAMES:
                shutil.copy2(source_dir / filename, destination / filename)
            imported += 1
    return {"imported_job_count": imported, "skipped_job_count": skipped, "source_zip": str(source_zip)}
