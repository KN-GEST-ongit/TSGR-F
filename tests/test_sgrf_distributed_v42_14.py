from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tsgr.baselines.registry import COMPARABLE_SGRF_METHODS
from tsgr.baselines.sgrf_distributed import (
    collect_sgrf_progress,
    export_sgrf_evaluation_shard,
    import_sgrf_evaluation_shard,
)
from tsgr.baselines.sgrf_evaluation import (
    SGRF_FAST_FIRST_METHOD_IDS,
    _execution_method_order,
)
from tsgr.evaluation.io_utils import write_csv_rows


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan"
    (plan / "S1_ALL_IN_DOMAIN" / "all").mkdir(parents=True)
    (plan / "S2_LOBO" / "background_BLACK").mkdir(parents=True)
    (plan / "experiment_plan.json").write_text(json.dumps({"dataset_root": "."}), encoding="utf-8")
    for scenario, fold_id in (("S1_ALL_IN_DOMAIN", "all"), ("S2_LOBO", "background_BLACK")):
        fold = plan / scenario / fold_id
        (fold / "fold.json").write_text(
            json.dumps({"scenario": scenario, "fold_id": fold_id, "fold_status": "active"}),
            encoding="utf-8",
        )
        write_csv_rows(fold / "test_takes.csv", [{"take_id": f"{scenario}_take"}], ["take_id"])
    return plan


def _complete_job(evaluation: Path, scenario: str, fold_id: str, method_id: str, fingerprint: str) -> None:
    job = evaluation / "jobs" / scenario / fold_id / method_id
    job.mkdir(parents=True, exist_ok=True)
    write_csv_rows(job / "worker_predictions.csv", [{"row_id": "r1", "predicted_state": "GESTURE_A"}], ["row_id", "predicted_state"])
    (job / "prediction_run.json").write_text(
        json.dumps({
            "worker_status": "ok",
            "bridge_returncode": 0,
            "prediction_fingerprint": fingerprint,
        }),
        encoding="utf-8",
    )


def test_fast_first_scheduler_keeps_all_thirteen_and_defers_two_heaviest() -> None:
    ordered = _execution_method_order(COMPARABLE_SGRF_METHODS, "fast_first")
    ids = tuple(method.method_id for method in ordered)
    assert len(ids) == 13
    assert set(ids) == {method.method_id for method in COMPARABLE_SGRF_METHODS}
    assert ids == SGRF_FAST_FIRST_METHOD_IDS
    assert ids[-2:] == ("MOHANTY_RAMBHATLA", "OYEDOTUN_KHASHMAN")
    assert _execution_method_order(COMPARABLE_SGRF_METHODS, "registry") == COMPARABLE_SGRF_METHODS


def test_progress_counts_selected_method_fold_jobs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    evaluation = tmp_path / "evaluation"
    _complete_job(evaluation, "S1_ALL_IN_DOMAIN", "all", "MAUNG", "fp-s1")
    progress = collect_sgrf_progress(
        plan,
        evaluation_dir=evaluation,
        methods=["MAUNG"],
        all_methods=False,
        scenarios=None,
        folds=None,
    )
    assert progress["expected_job_count"] == 2
    assert progress["complete_job_count"] == 1
    assert progress["pending_job_count"] == 1
    assert progress["by_method"][0]["complete_jobs"] == 1


def test_shard_export_import_is_incremental_and_idempotent(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = tmp_path / "source_eval"
    _complete_job(source, "S1_ALL_IN_DOMAIN", "all", "MAUNG", "fp-s1")
    shard = tmp_path / "progress.zip"
    export_sgrf_evaluation_shard(
        plan,
        evaluation_dir=source,
        output_zip=shard,
        methods=["MAUNG"],
        all_methods=False,
        scenarios=None,
        folds=None,
        overwrite=False,
    )
    with zipfile.ZipFile(shard) as archive:
        names = set(archive.namelist())
    assert "sgrf_evaluation_shard/shard_manifest.json" in names
    assert "sgrf_evaluation_shard/checksums.sha256" in names
    assert any(name.endswith("worker_predictions.csv") for name in names)

    target = tmp_path / "target_eval"
    first = import_sgrf_evaluation_shard(shard, evaluation_dir=target)
    second = import_sgrf_evaluation_shard(shard, evaluation_dir=target)
    assert first["imported_job_count"] == 1
    assert first["skipped_job_count"] == 0
    assert second["imported_job_count"] == 0
    assert second["skipped_job_count"] == 1


def test_shard_import_rejects_existing_conflicting_checkpoint(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = tmp_path / "source_eval"
    _complete_job(source, "S1_ALL_IN_DOMAIN", "all", "MAUNG", "fp-source")
    shard = tmp_path / "progress.zip"
    export_sgrf_evaluation_shard(
        plan,
        evaluation_dir=source,
        output_zip=shard,
        methods=["MAUNG"],
        all_methods=False,
        scenarios={"S1_ALL_IN_DOMAIN"},
        folds=None,
    )
    target = tmp_path / "target_eval"
    _complete_job(target, "S1_ALL_IN_DOMAIN", "all", "MAUNG", "fp-other")
    with pytest.raises(RuntimeError, match="conflicts"):
        import_sgrf_evaluation_shard(shard, evaluation_dir=target)


def test_shard_import_accepts_same_predictions_with_machine_specific_run_metadata(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = tmp_path / "source_eval"
    _complete_job(source, "S1_ALL_IN_DOMAIN", "all", "MAUNG", "fp-same")
    source_run = source / "jobs" / "S1_ALL_IN_DOMAIN" / "all" / "MAUNG" / "prediction_run.json"
    payload = json.loads(source_run.read_text(encoding="utf-8"))
    payload["bridge_elapsed_s"] = 123.0
    payload["bridge_command"] = ["machine_a/python"]
    source_run.write_text(json.dumps(payload), encoding="utf-8")
    shard = tmp_path / "progress.zip"
    export_sgrf_evaluation_shard(
        plan,
        evaluation_dir=source,
        output_zip=shard,
        methods=["MAUNG"],
        all_methods=False,
        scenarios={"S1_ALL_IN_DOMAIN"},
        folds=None,
    )
    target = tmp_path / "target_eval"
    _complete_job(target, "S1_ALL_IN_DOMAIN", "all", "MAUNG", "fp-same")
    target_run = target / "jobs" / "S1_ALL_IN_DOMAIN" / "all" / "MAUNG" / "prediction_run.json"
    target_payload = json.loads(target_run.read_text(encoding="utf-8"))
    target_payload["bridge_elapsed_s"] = 999.0
    target_payload["bridge_command"] = ["machine_b/python"]
    target_run.write_text(json.dumps(target_payload), encoding="utf-8")
    result = import_sgrf_evaluation_shard(shard, evaluation_dir=target)
    assert result["imported_job_count"] == 0
    assert result["skipped_job_count"] == 1
