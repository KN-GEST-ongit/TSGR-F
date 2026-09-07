"""Completeness audit and portable export for the external SGRF benchmark.

The external SGRF comparison contains thirteen methods evaluated independently on
all active TSGR-F folds.  These helpers deliberately treat the presence of a
finished Python process as insufficient evidence of completeness: every selected
method x fold job is audited against its model manifest, prediction manifest,
worker output and run status.  Aggregate publication tables are checked separately.

Portable export excludes heavyweight upstream model binaries by design.  The
small ``baseline_model.json`` manifests and their fingerprints are retained so a
result bundle can be traced back to the exact fold-local training artefacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from tsgr.baselines.registry import SGRFMethodSpec, normalize_method_selection
from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows
from tsgr.utils.serialization import write_json


@dataclass(frozen=True, slots=True)
class AuditFold:
    scenario: str
    fold_id: str
    test_take_count: int


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


def discover_active_folds(
    experiment_plan_dir: str | Path,
    *,
    scenarios: set[str] | None,
    folds: set[str] | None,
) -> list[AuditFold]:
    plan = Path(experiment_plan_dir)
    if not (plan / "experiment_plan.json").is_file():
        raise FileNotFoundError(f"Missing experiment_plan.json: {plan / 'experiment_plan.json'}")
    output: list[AuditFold] = []
    for fold_json in sorted(plan.glob("*/*/fold.json")):
        meta = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(meta.get("scenario", ""))
        fold_id = str(meta.get("fold_id", ""))
        if scenarios and scenario not in scenarios:
            continue
        if folds and fold_id not in folds:
            continue
        if str(meta.get("fold_status", "active")) != "active":
            continue
        test_takes = read_csv_rows(fold_json.parent / "test_takes.csv")
        if not test_takes:
            continue
        output.append(AuditFold(scenario=scenario, fold_id=fold_id, test_take_count=len(test_takes)))
    if not output:
        raise ValueError("No active folds with test_takes.csv matched the requested scope.")
    return output


def _safe_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _audit_one_job(
    baseline_models_dir: Path,
    evaluation_dir: Path,
    fold: AuditFold,
    method: SGRFMethodSpec,
    *,
    deep_row_count: bool,
) -> dict[str, Any]:
    model_manifest = baseline_models_dir / "models" / fold.scenario / fold.fold_id / method.method_id / "baseline_model.json"
    job_dir = evaluation_dir / "jobs" / fold.scenario / fold.fold_id / method.method_id
    prediction_manifest = job_dir / "prediction_manifest.csv"
    prediction_run = job_dir / "prediction_run.json"
    worker_predictions = job_dir / "worker_predictions.csv"

    issues: list[str] = []
    model_payload, model_error = _safe_json(model_manifest)
    model_ok = model_payload is not None
    if not model_ok:
        issues.append(f"model_manifest:{model_error}")
    else:
        if str(model_payload.get("scenario", "")) != fold.scenario:
            model_ok = False
            issues.append("model_manifest:scenario_mismatch")
        if str(model_payload.get("fold_id", "")) != fold.fold_id:
            model_ok = False
            issues.append("model_manifest:fold_id_mismatch")
        if str(model_payload.get("method_id", "")) != method.method_id:
            model_ok = False
            issues.append("model_manifest:method_id_mismatch")
        if not str(model_payload.get("training_fingerprint", "")):
            model_ok = False
            issues.append("model_manifest:missing_training_fingerprint")

    manifest_ok = prediction_manifest.is_file() and prediction_manifest.stat().st_size > 0
    if not manifest_ok:
        issues.append("prediction_manifest:missing_or_empty")

    run_payload, run_error = _safe_json(prediction_run)
    run_ok = run_payload is not None
    worker_status = ""
    bridge_returncode: int | str = ""
    if not run_ok:
        issues.append(f"prediction_run:{run_error}")
    else:
        worker_status = str(run_payload.get("worker_status", ""))
        bridge_returncode = run_payload.get("bridge_returncode", "")
        if worker_status.lower() == "error":
            run_ok = False
            issues.append("prediction_run:worker_status_error")
        if bridge_returncode not in ("", None, 0, "0"):
            run_ok = False
            issues.append(f"prediction_run:bridge_returncode_{bridge_returncode}")

    worker_output_ok = worker_predictions.is_file() and worker_predictions.stat().st_size > 0
    if not worker_output_ok:
        issues.append("worker_predictions:missing_or_empty")

    manifest_rows: int | str = ""
    worker_rows: int | str = ""
    row_count_ok: bool | str = ""
    if deep_row_count and manifest_ok and worker_output_ok:
        manifest_rows = _csv_data_row_count(prediction_manifest)
        worker_rows = _csv_data_row_count(worker_predictions)
        row_count_ok = manifest_rows == worker_rows and manifest_rows >= 0
        if not row_count_ok:
            issues.append(f"row_count_mismatch:{manifest_rows}!={worker_rows}")

    complete = bool(model_ok and manifest_ok and run_ok and worker_output_ok and (row_count_ok is not False))
    return {
        "scenario": fold.scenario,
        "fold_id": fold.fold_id,
        "method_id": method.method_id,
        "method_display_name": method.display_name,
        "status": "complete" if complete else "incomplete",
        "model_manifest_ok": int(bool(model_ok)),
        "prediction_manifest_ok": int(bool(manifest_ok)),
        "prediction_run_ok": int(bool(run_ok)),
        "worker_predictions_ok": int(bool(worker_output_ok)),
        "worker_status": worker_status,
        "bridge_returncode": bridge_returncode,
        "prediction_manifest_rows": manifest_rows,
        "worker_prediction_rows": worker_rows,
        "deep_row_count_ok": row_count_ok,
        "issues": ";".join(issues),
        "model_manifest": model_manifest.relative_to(baseline_models_dir).as_posix() if model_manifest.is_file() else "",
        "job_dir": job_dir.relative_to(evaluation_dir).as_posix() if job_dir.exists() else "",
    }


def _aggregate_table_audit(
    evaluation_dir: Path,
    *,
    job_count: int,
    method_count: int,
    scenario_count: int,
    expected_video_rows: int,
) -> list[dict[str, Any]]:
    expectations: list[tuple[str, int | None]] = [
        ("fold_summary.csv", job_count),
        ("gesture_metrics.csv", job_count * 17),
        ("scenario_summary.csv", method_count * scenario_count),
        ("video_summary.csv", expected_video_rows),
        ("group_summary.csv", None),
        ("confusion_matrix_long.csv", None),
        ("evaluation_report.json", None),
        ("sgrf_baseline_results.xlsx", None),
    ]
    rows: list[dict[str, Any]] = []
    for name, expected in expectations:
        path = evaluation_dir / name
        exists = path.is_file() and path.stat().st_size > 0
        actual: int | str = ""
        count_ok: bool | str = ""
        issue = ""
        if name.endswith(".csv") and exists:
            actual = _csv_data_row_count(path)
            if expected is not None:
                count_ok = actual == expected
                if not count_ok:
                    issue = f"row_count_mismatch:{actual}!={expected}"
        elif expected is not None:
            count_ok = False
        if not exists:
            issue = "missing_or_empty"
        rows.append(
            {
                "artifact": name,
                "exists": int(exists),
                "expected_rows": "" if expected is None else expected,
                "actual_rows": actual,
                "row_count_ok": count_ok,
                "status": "complete" if exists and count_ok is not False else "incomplete",
                "issue": issue,
            }
        )
    return rows


def audit_sgrf_evaluation_completeness(
    experiment_plan_dir: str | Path,
    *,
    baseline_models_dir: str | Path,
    evaluation_dir: str | Path,
    output_dir: str | Path,
    methods: Iterable[str] | None = None,
    all_methods: bool = False,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    deep_row_count: bool = False,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Audit every selected SGRF method x fold job and the aggregate outputs."""
    models_root = Path(baseline_models_dir)
    eval_root = Path(evaluation_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_methods = normalize_method_selection(methods, all_methods=all_methods)
    selected_folds = discover_active_folds(experiment_plan_dir, scenarios=scenarios, folds=folds)

    job_rows = [
        _audit_one_job(models_root, eval_root, fold, method, deep_row_count=deep_row_count)
        for fold in selected_folds
        for method in selected_methods
    ]
    complete_jobs = sum(row["status"] == "complete" for row in job_rows)
    job_count = len(job_rows)
    expected_video_rows = sum(fold.test_take_count for fold in selected_folds) * len(selected_methods)
    scenario_count = len({fold.scenario for fold in selected_folds})
    aggregate_rows = _aggregate_table_audit(
        eval_root,
        job_count=job_count,
        method_count=len(selected_methods),
        scenario_count=scenario_count,
        expected_video_rows=expected_video_rows,
    )
    aggregates_complete = all(row["status"] == "complete" for row in aggregate_rows)

    failure_file = eval_root / "prediction_job_failures.csv"
    failure_count = _csv_data_row_count(failure_file) if failure_file.is_file() else 0
    report_payload, _ = _safe_json(eval_root / "evaluation_report.json")
    report_external_failures = int((report_payload or {}).get("external_job_failure_count", 0) or 0)
    report_frame_count = int((report_payload or {}).get("frame_result_count", 0) or 0)

    complete = bool(
        complete_jobs == job_count
        and aggregates_complete
        and failure_count == 0
        and report_external_failures == 0
    )
    fields = [
        "scenario", "fold_id", "method_id", "method_display_name", "status",
        "model_manifest_ok", "prediction_manifest_ok", "prediction_run_ok", "worker_predictions_ok",
        "worker_status", "bridge_returncode", "prediction_manifest_rows", "worker_prediction_rows",
        "deep_row_count_ok", "issues", "model_manifest", "job_dir",
    ]
    write_csv_rows(out / "sgrf_evaluation_completeness.csv", job_rows, fields)
    incomplete_rows = [row for row in job_rows if row["status"] != "complete"]
    write_csv_rows(out / "sgrf_evaluation_incomplete_jobs.csv", incomplete_rows, fields)
    write_csv_rows(
        out / "sgrf_evaluation_aggregate_artifacts.csv",
        aggregate_rows,
        ["artifact", "exists", "expected_rows", "actual_rows", "row_count_ok", "status", "issue"],
    )
    summary = {
        "schema_version": "tsgr_sgrf_evaluation_completeness_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_plan": Path(experiment_plan_dir).name,
        "method_count": len(selected_methods),
        "fold_count": len(selected_folds),
        "scenario_count": scenario_count,
        "expected_job_count": job_count,
        "complete_job_count": complete_jobs,
        "incomplete_job_count": job_count - complete_jobs,
        "expected_video_summary_rows": expected_video_rows,
        "deep_row_count": bool(deep_row_count),
        "aggregate_artifacts_complete": aggregates_complete,
        "prediction_job_failure_rows": failure_count,
        "evaluation_report_external_job_failure_count": report_external_failures,
        "evaluation_report_frame_result_count": report_frame_count,
        "complete": complete,
        "method_ids": [item.method_id for item in selected_methods],
        "scenarios": sorted({fold.scenario for fold in selected_folds}),
    }
    write_json(out / "sgrf_evaluation_completeness_summary.json", summary)
    if require_complete and not complete:
        raise RuntimeError(
            f"SGRF evaluation is incomplete: jobs={complete_jobs}/{job_count}, "
            f"aggregate_artifacts_complete={aggregates_complete}, failures={failure_count + report_external_failures}. "
            f"See {out / 'sgrf_evaluation_completeness.csv'}"
        )
    return summary


def _copy_if_file(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def export_sgrf_portable_bundle(
    experiment_plan_dir: str | Path,
    *,
    baseline_models_dir: str | Path,
    evaluation_dir: str | Path,
    output_zip: str | Path,
    methods: Iterable[str] | None = None,
    all_methods: bool = False,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    include_error_frames: bool = False,
    include_frame_predictions: bool = False,
    allow_incomplete: bool = False,
    deep_row_count: bool = False,
) -> Path:
    """Create a portable result ZIP without copying heavyweight SGRF model binaries."""
    models_root = Path(baseline_models_dir)
    eval_root = Path(evaluation_dir)
    target_zip = Path(output_zip)
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    selected_methods = normalize_method_selection(methods, all_methods=all_methods)
    selected_folds = discover_active_folds(experiment_plan_dir, scenarios=scenarios, folds=folds)

    with tempfile.TemporaryDirectory(prefix="tsgr_sgrf_portable_") as temp_name:
        temp = Path(temp_name) / "sgrf_portable_results"
        temp.mkdir(parents=True, exist_ok=True)
        audit_dir = temp / "completeness_audit"
        summary = audit_sgrf_evaluation_completeness(
            experiment_plan_dir,
            baseline_models_dir=models_root,
            evaluation_dir=eval_root,
            output_dir=audit_dir,
            methods=[item.method_id for item in selected_methods],
            all_methods=False,
            scenarios={fold.scenario for fold in selected_folds},
            folds={fold.fold_id for fold in selected_folds} if folds else None,
            deep_row_count=deep_row_count,
            require_complete=False,
        )
        if not allow_incomplete and not summary["complete"]:
            raise RuntimeError(
                "Refusing to export an incomplete SGRF benchmark. Run the completeness audit, resume evaluation, "
                "or pass allow_incomplete=True only for a diagnostic transfer."
            )

        aggregate_names = [
            "scenario_summary.csv", "fold_summary.csv", "gesture_metrics.csv", "group_summary.csv",
            "video_summary.csv", "confusion_matrix_long.csv", "sgrf_baseline_results.xlsx",
            "evaluation_report.json", "method_registry.csv", "prediction_job_failures.csv",
        ]
        if include_error_frames:
            aggregate_names.append("error_frames.csv")
        if include_frame_predictions:
            aggregate_names.append("frame_predictions.csv")
        for name in aggregate_names:
            _copy_if_file(eval_root / name, temp / "evaluation" / name)
        _copy_tree_if_exists(eval_root / "plots", temp / "evaluation" / "plots")
        _copy_tree_if_exists(eval_root / "environment", temp / "evaluation" / "environment")

        for name in ("model_build_summary.csv", "model_build_summary.json", "method_registry.csv"):
            _copy_if_file(models_root / name, temp / "baseline_models" / name)
        _copy_tree_if_exists(models_root / "environment", temp / "baseline_models" / "environment")

        copied_model_manifests = 0
        for fold in selected_folds:
            for method in selected_methods:
                source = models_root / "models" / fold.scenario / fold.fold_id / method.method_id / "baseline_model.json"
                destination = temp / "baseline_models" / "models" / fold.scenario / fold.fold_id / method.method_id / "baseline_model.json"
                copied_model_manifests += int(_copy_if_file(source, destination))

        provenance = {
            "schema_version": "tsgr_sgrf_portable_bundle_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": Path(experiment_plan_dir).name,
            "complete": bool(summary["complete"]),
            "method_ids": [item.method_id for item in selected_methods],
            "fold_count": len(selected_folds),
            "expected_job_count": len(selected_methods) * len(selected_folds),
            "baseline_model_manifest_count": copied_model_manifests,
            "include_error_frames": bool(include_error_frames),
            "include_frame_predictions": bool(include_frame_predictions),
            "heavyweight_model_binaries_included": False,
            "notes": [
                "This archive is intended for result merging and publication analysis on another computer.",
                "Heavy SGRF model files are intentionally omitted; baseline_model.json fingerprints are retained.",
                "frame_predictions.csv is optional because aggregate tables and video_summary.csv cover the normal publication workflow.",
            ],
        }
        write_json(temp / "portable_bundle.json", provenance)
        manifest = _manifest_rows(temp)
        write_csv_rows(temp / "FILE_MANIFEST.csv", manifest, ["relative_path", "size_bytes", "sha256"])

        if target_zip.exists():
            target_zip.unlink()
        with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(item for item in temp.rglob("*") if item.is_file()):
                archive.write(path, path.relative_to(temp.parent).as_posix())
    return target_zip
