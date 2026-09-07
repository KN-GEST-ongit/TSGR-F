from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from tsgr.baselines.sgrf_portability import (
    audit_sgrf_evaluation_completeness,
    export_sgrf_portable_bundle,
)
from tsgr.evaluation.io_utils import write_csv_rows


def _write_rows(path: Path, count: int, fields: list[str]) -> None:
    rows = [{field: f"{field}_{index}" for field in fields} for index in range(count)]
    write_csv_rows(path, rows, fields)


def _build_complete_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    plan = tmp_path / "plan"
    fold = plan / "S1_ALL_IN_DOMAIN" / "all"
    fold.mkdir(parents=True)
    (plan / "experiment_plan.json").write_text(json.dumps({"dataset_root": "."}), encoding="utf-8")
    (fold / "fold.json").write_text(
        json.dumps({"scenario": "S1_ALL_IN_DOMAIN", "fold_id": "all", "fold_status": "active"}),
        encoding="utf-8",
    )
    write_csv_rows(fold / "test_takes.csv", [{"take_id": "take_1"}, {"take_id": "take_2"}], ["take_id"])

    models = tmp_path / "models"
    model_dir = models / "models" / "S1_ALL_IN_DOMAIN" / "all" / "MAUNG"
    model_dir.mkdir(parents=True)
    (model_dir / "baseline_model.json").write_text(
        json.dumps({
            "scenario": "S1_ALL_IN_DOMAIN", "fold_id": "all", "method_id": "MAUNG",
            "training_fingerprint": "abc123", "model_dir_relative": "model",
        }),
        encoding="utf-8",
    )
    (model_dir / "model").mkdir()
    (model_dir / "model" / "heavy.bin").write_bytes(b"not portable")
    write_csv_rows(models / "model_build_summary.csv", [{"status": "ok"}], ["status"])
    (models / "model_build_summary.json").write_text(json.dumps({"failure_count": 0}), encoding="utf-8")

    evaluation = tmp_path / "evaluation"
    job = evaluation / "jobs" / "S1_ALL_IN_DOMAIN" / "all" / "MAUNG"
    job.mkdir(parents=True)
    write_csv_rows(
        job / "prediction_manifest.csv",
        [{"row_id": "r1"}, {"row_id": "r2"}, {"row_id": "r3"}],
        ["row_id"],
    )
    write_csv_rows(
        job / "worker_predictions.csv",
        [{"row_id": "r1"}, {"row_id": "r2"}, {"row_id": "r3"}],
        ["row_id"],
    )
    (job / "prediction_run.json").write_text(
        json.dumps({"worker_status": "ok", "bridge_returncode": 0}), encoding="utf-8"
    )
    _write_rows(evaluation / "fold_summary.csv", 1, ["method_id"])
    _write_rows(evaluation / "gesture_metrics.csv", 17, ["method_id"])
    _write_rows(evaluation / "scenario_summary.csv", 1, ["method_id"])
    _write_rows(evaluation / "video_summary.csv", 2, ["method_id"])
    _write_rows(evaluation / "group_summary.csv", 1, ["method_id"])
    _write_rows(evaluation / "confusion_matrix_long.csv", 1, ["method_id"])
    (evaluation / "sgrf_baseline_results.xlsx").write_bytes(b"xlsx")
    (evaluation / "evaluation_report.json").write_text(
        json.dumps({"external_job_failure_count": 0, "frame_result_count": 3}), encoding="utf-8"
    )
    return plan, models, evaluation


def test_completeness_audit_checks_job_and_aggregate_counts(tmp_path: Path) -> None:
    plan, models, evaluation = _build_complete_fixture(tmp_path)
    output = tmp_path / "audit"
    summary = audit_sgrf_evaluation_completeness(
        plan,
        baseline_models_dir=models,
        evaluation_dir=evaluation,
        output_dir=output,
        methods=["MAUNG"],
        all_methods=False,
        scenarios=None,
        folds=None,
        deep_row_count=True,
        require_complete=True,
    )
    assert summary["complete"] is True
    assert summary["expected_job_count"] == 1
    assert summary["complete_job_count"] == 1
    assert summary["expected_video_summary_rows"] == 2
    assert (output / "sgrf_evaluation_completeness_summary.json").is_file()


def test_completeness_audit_detects_worker_row_mismatch(tmp_path: Path) -> None:
    plan, models, evaluation = _build_complete_fixture(tmp_path)
    job = evaluation / "jobs" / "S1_ALL_IN_DOMAIN" / "all" / "MAUNG"
    write_csv_rows(job / "worker_predictions.csv", [{"row_id": "r1"}], ["row_id"])
    summary = audit_sgrf_evaluation_completeness(
        plan,
        baseline_models_dir=models,
        evaluation_dir=evaluation,
        output_dir=tmp_path / "audit",
        methods=["MAUNG"],
        all_methods=False,
        scenarios=None,
        folds=None,
        deep_row_count=True,
        require_complete=False,
    )
    assert summary["complete"] is False
    with pytest.raises(RuntimeError, match="incomplete"):
        audit_sgrf_evaluation_completeness(
            plan,
            baseline_models_dir=models,
            evaluation_dir=evaluation,
            output_dir=tmp_path / "audit2",
            methods=["MAUNG"],
            all_methods=False,
            scenarios=None,
            folds=None,
            deep_row_count=True,
            require_complete=True,
        )


def test_portable_bundle_excludes_heavy_model_binaries(tmp_path: Path) -> None:
    plan, models, evaluation = _build_complete_fixture(tmp_path)
    target = tmp_path / "portable.zip"
    export_sgrf_portable_bundle(
        plan,
        baseline_models_dir=models,
        evaluation_dir=evaluation,
        output_zip=target,
        methods=["MAUNG"],
        all_methods=False,
        scenarios=None,
        folds=None,
        include_error_frames=False,
        include_frame_predictions=False,
        allow_incomplete=False,
        deep_row_count=True,
    )
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
    assert any(name.endswith("baseline_model.json") for name in names)
    assert not any(name.endswith("heavy.bin") for name in names)
    assert any(name.endswith("portable_bundle.json") for name in names)
    assert any(name.endswith("FILE_MANIFEST.csv") for name in names)
