from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from tsgr.baselines.registry import (
    COMPARABLE_SGRF_METHODS,
    EXCLUDED_SGRF_METHODS,
    normalize_method_selection,
    registry_rows,
)
from tsgr.baselines.result_merge import merge_method_results
from tsgr.baselines.sgrf_evaluation import (
    EvaluationFold,
    _discover_folds as discover_evaluation_folds,
    _prepare_prediction_job,
    benchmark_ground_truth_state,
    evaluate_sgrf_baselines,
)
from tsgr.baselines.sgrf_models import _discover_folds as discover_training_folds, resolve_external_worker_count
from tsgr.baselines.sgrf_worker import GESTURE_ORDER, _coordinates, _gesture_enum, _normalize_certainty
from tsgr.cli import build_sgrf_baselines as build_cli
from tsgr.cli import evaluate_sgrf_baselines as evaluate_cli
from tsgr.evaluation.io_utils import write_csv_rows


def test_registry_contains_exactly_thirteen_comparable_methods() -> None:
    assert len(COMPARABLE_SGRF_METHODS) == 13
    assert {item.method_id for item in EXCLUDED_SGRF_METHODS} == {
        "MURTHY_JADON",
        "ISLAM_HOSSAIN_ANDERSSON",
    }
    assert all(item.included for item in COMPARABLE_SGRF_METHODS)
    assert all(not item.included for item in EXCLUDED_SGRF_METHODS)


def test_registry_keeps_tsgrf_as_last_table_row() -> None:
    rows = registry_rows()
    assert rows[-1]["method_id"] == "TSGRF"
    assert rows[-1]["sort_order"] == 9999
    assert max(int(row["sort_order"]) for row in rows[:-1]) < 9999


def test_explicit_method_selection_rejects_excluded_background_method() -> None:
    with pytest.raises(ValueError, match="intentionally excluded"):
        normalize_method_selection(["MURTHY_JADON"], all_methods=False)


def test_dynamic_pjm_enum_has_seventeen_labels_in_canonical_order() -> None:
    enum_type = _gesture_enum(GESTURE_ORDER)
    assert [item.name for item in enum_type] == list(GESTURE_ORDER)
    assert [item.value for item in enum_type] == list(range(1, 18))


def test_full_frame_coordinates_are_api_compatibility_rectangle() -> None:
    assert _coordinates("none", 640, 480) is None
    assert _coordinates("full_frame", 640, 480) == [(0, 0), (640, 480)]
    with pytest.raises(ValueError):
        _coordinates("full_frame", 0, 480)



def test_upstream_certainty_scales_are_normalized_consistently() -> None:
    assert _normalize_certainty(87, "percent") == pytest.approx(0.87)
    assert _normalize_certainty(0.87, "unit") == pytest.approx(0.87)
    assert _normalize_certainty(None, "none") is None
    assert _normalize_certainty(120, "percent") == pytest.approx(1.0)
    assert _normalize_certainty(-0.2, "unit") == pytest.approx(0.0)


def test_registry_marks_maung_without_certainty_and_zhuang_as_unit_scale() -> None:
    by_id = {item.method_id: item for item in COMPARABLE_SGRF_METHODS}
    assert by_id["MAUNG"].certainty_scale == "none"
    assert by_id["ZHUANG_YANG"].certainty_scale == "unit"
    assert by_id["ADITHYA_RAJESH"].certainty_scale == "percent"

def test_external_worker_count_is_conservative_in_auto_mode() -> None:
    assert resolve_external_worker_count(0, 20) == 1
    assert resolve_external_worker_count(4, 2) == 2
    assert resolve_external_worker_count(1, 20) == 1
    with pytest.raises(ValueError):
        resolve_external_worker_count(-1, 1)


def test_no_hand_is_collapsed_only_for_common_static_benchmark() -> None:
    assert benchmark_ground_truth_state("NO_HAND") == "NO_GESTURE"
    assert benchmark_ground_truth_state("NO_GESTURE") == "NO_GESTURE"
    assert benchmark_ground_truth_state("GESTURE_M") == "GESTURE_M"




def test_dataset_root_override_makes_copied_experiment_plan_portable(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    fold = plan / "S1_ALL_IN_DOMAIN" / "all"
    fold.mkdir(parents=True)
    dataset = tmp_path / "dataset_on_second_pc"
    dataset.mkdir()
    (plan / "experiment_plan.json").write_text(
        json.dumps({"dataset_root": "__MOVED_DATASET__"}), encoding="utf-8"
    )
    (fold / "fold.json").write_text(
        json.dumps({"scenario": "S1_ALL_IN_DOMAIN", "fold_id": "all", "fold_status": "active"}),
        encoding="utf-8",
    )
    write_csv_rows(
        fold / "training_images.csv",
        [{"relative_path": "training/P01/A/example.jpg", "gesture_id": "A"}],
        ["relative_path", "gesture_id"],
    )
    write_csv_rows(
        fold / "test_takes.csv",
        [{"take_id": "take_001"}],
        ["take_id"],
    )
    train_folds = discover_training_folds(
        plan, scenarios=None, folds=None, dataset_root_override=dataset
    )
    eval_folds = discover_evaluation_folds(
        plan, scenarios=None, folds=None, dataset_root_override=dataset
    )
    assert train_folds[0].dataset_root == dataset.resolve()
    assert eval_folds[0].dataset_root == dataset.resolve()

def test_prepare_prediction_job_fingerprints_normalized_operational_gt_state(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    frame = dataset / "test" / "P01" / "BLACK" / "A" / "take_0001" / "frame_000001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame-placeholder")

    fold_dir = tmp_path / "plan" / "S1_ALL_IN_DOMAIN" / "all"
    fold_dir.mkdir(parents=True)
    fold = EvaluationFold(
        scenario="S1_ALL_IN_DOMAIN",
        fold_id="all",
        fold_dir=fold_dir,
        dataset_root=dataset,
        test_take_ids=frozenset({"P01_BLACK_A_take_0001"}),
    )

    model_manifest_path = tmp_path / "models" / "models" / "S1_ALL_IN_DOMAIN" / "all" / "MAUNG" / "baseline_model.json"
    model_manifest_path.parent.mkdir(parents=True)
    (model_manifest_path.parent / "model").mkdir()
    model_manifest = {
        "training_fingerprint": "train-fingerprint",
        "model_files": [],
        "model_dir_relative": "model",
        "gestures": list(GESTURE_ORDER),
        "custom_options": {},
    }
    gt_rows = [{
        "take_id": "P01_BLACK_A_take_0001",
        "frame_index": "1",
        "frame_relative_path": "test/P01/BLACK/A/take_0001/frame_000001.jpg",
        "public_subject_id": "P01",
        "background": "BLACK",
        "gesture_id": "A",
        "evaluation_state": "GESTURE_A",
        "include_in_metrics": "1",
    }]

    job, _, fingerprint, rows = _prepare_prediction_job(
        tmp_path / "eval", fold, COMPARABLE_SGRF_METHODS[0], model_manifest, model_manifest_path, gt_rows,
        rejection_policy="closed_set", certainty_threshold_normalized=None, model_load_policy="process_cache",
        seed=2026, fail_fast=False, progress_every=1000,
    )
    assert rows[0]["operational_gt_state"] == "GESTURE_A"
    assert "evaluation_state" not in rows[0]
    assert job["prediction_fingerprint"] == fingerprint
    assert fingerprint


def test_evaluator_rejects_non_normalized_certainty_threshold_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        evaluate_sgrf_baselines(
            tmp_path / "plan",
            sgrf_python=tmp_path / "python.exe",
            baseline_models_dir=tmp_path / "models",
            ground_truth_report=tmp_path / "gt",
            output_dir=tmp_path / "out",
            methods=["ADITHYA_RAJESH"],
            all_methods=False,
            rejection_policy="certainty_reject",
            certainty_threshold_normalized=50.0,
            model_load_policy="process_cache",
            workers=1,
        )

def test_build_cli_requires_explicit_workers(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tsgr-build-sgrf-baselines",
            str(plan),
            "--sgrf-python", str(tmp_path / "python.exe"),
            "--output-dir", str(tmp_path / "out"),
            "--all-methods",
            "--all-scenarios",
            "--all-folds",
        ],
    )
    with pytest.raises(SystemExit) as error:
        build_cli.main()
    assert error.value.code == 2


def test_evaluate_cli_requires_explicit_rejection_and_load_policy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tsgr-evaluate-sgrf-baselines",
            str(tmp_path / "plan"),
            "--sgrf-python", str(tmp_path / "python.exe"),
            "--baseline-models-dir", str(tmp_path / "models"),
            "--ground-truth-report", str(tmp_path / "gt"),
            "--output-dir", str(tmp_path / "out"),
            "--all-methods", "--all-scenarios", "--all-folds",
            "--workers", "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        evaluate_cli.main()
    assert error.value.code == 2


def test_merge_orders_tsgrf_after_all_baselines(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    tsgrf = tmp_path / "tsgrf"
    output = tmp_path / "merged"
    baseline.mkdir()
    tsgrf.mkdir()
    tables = ("scenario_summary", "fold_summary", "gesture_metrics", "group_summary", "video_summary")
    for table in tables:
        write_csv_rows(
            baseline / f"{table}.csv",
            [{"method_id": "MAUNG", "method_display_name": "Maung", "method_sort_order": 10, "scenario": "S1"}],
            ["method_id", "method_display_name", "method_sort_order", "scenario"],
        )
        write_csv_rows(
            tsgrf / f"{table}.csv",
            [{"method_id": "TSGRF", "method_sort_order": 9999, "scenario": "S1"}],
            ["method_id", "method_sort_order", "scenario"],
        )
    merge_method_results(baseline, tsgrf, output_dir=output)
    with (output / "comparison_scenario_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method_id"] for row in rows] == ["MAUNG", "TSGRF"]
    assert (output / "method_comparison_results.xlsx").is_file()
    manifest = json.loads((output / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert "TSGRF" in manifest["ordering_rule"]
