from __future__ import annotations

import csv
import json
from pathlib import Path

from tsgr.experiments import folds as folds_module
from tsgr.experiments.model_building import (
    discover_model_build_jobs,
    resolve_model_build_worker_count,
)
from tsgr.processing import test_take_audit as test_module


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_test_take_signature_ignores_annotations_but_tracks_frame_content() -> None:
    rows = [
        {"frame_index": "0", "filename": "frame_000000.jpg", "sha256": "aaa", "true_label": "A"},
        {"frame_index": "1", "filename": "frame_000001.jpg", "sha256": "bbb", "true_label": "NO_GESTURE"},
    ]
    changed_annotation = [dict(row) for row in rows]
    changed_annotation[0]["true_label"] = "NO_GESTURE"
    changed_content = [dict(row) for row in rows]
    changed_content[0]["sha256"] = "ccc"
    assert test_module.test_take_inventory_signature(rows) == test_module.test_take_inventory_signature(changed_annotation)
    assert test_module.test_take_inventory_signature(rows) != test_module.test_take_inventory_signature(changed_content)


def test_test_take_partition_balances_by_frame_count() -> None:
    tasks = [
        {"metadata": {"take_id": "long"}, "frame_count": 100},
        {"metadata": {"take_id": "m1"}, "frame_count": 60},
        {"metadata": {"take_id": "m2"}, "frame_count": 60},
        {"metadata": {"take_id": "short"}, "frame_count": 20},
    ]
    chunks = test_module.partition_test_tasks(tasks, 2)
    loads = sorted(sum(item["frame_count"] for item in chunk) for chunk in chunks)
    assert loads == [120, 120]


def test_video_run_match_checks_inventory_and_recovery_after(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "mediapipe_running_mode": "video",
                "mediapipe_video_profile": "high_recall",
                "mediapipe_video_recovery_enabled": True,
                "mediapipe_video_recovery_after_consecutive_failures": 2,
                "strict_right_hand_only": True,
            }
        ),
        encoding="utf-8",
    )
    signature = {"schema_version": "tsgr_test_take_inventory_v1", "frame_count": 2, "sha256": "abc"}
    (run / "test_take_processing_signature.json").write_text(
        json.dumps({**signature, "effective_fps": 20.0}), encoding="utf-8"
    )
    config = {
        "mediapipe": {
            "video_detection": {"active_profile": "high_recall"},
            "video_recovery": {"enabled": True, "after_consecutive_failures": 2},
        }
    }
    assert test_module._run_matches_video_config(run, config, inventory_signature=signature, effective_fps=20.0)
    config["mediapipe"]["video_recovery"]["after_consecutive_failures"] = 1
    assert not test_module._run_matches_video_config(run, config, inventory_signature=signature, effective_fps=20.0)


def test_scenario_run_index_resolves_each_unique_cell_once(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    rows = [
        {"public_subject_id": "P01", "background": "BLUE", "gesture_id": "A"},
        {"public_subject_id": "P01", "background": "BLUE", "gesture_id": "A"},
        {"public_subject_id": "P02", "background": "BLUE", "gesture_id": "A"},
    ]
    calls: list[Path] = []

    def fake_locate(path: Path):
        calls.append(path)
        return None

    monkeypatch.setattr(folds_module, "_fast_locate_training_run", fake_locate)
    index, report_rows, _, resolved = folds_module.build_training_run_index(root, rows, workers=1)
    assert len(index) == 2
    assert len(report_rows) == 2
    assert len(calls) == 2
    assert resolved == 1


def test_scenario_worker_count_auto_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    assert folds_module.resolve_scenario_worker_count(0, 100) == 8
    assert folds_module.resolve_scenario_worker_count(20, 3) == 3


def test_model_build_job_discovery_skips_existing_and_unready(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    (plan / "experiment_plan.json").parent.mkdir(parents=True)
    (plan / "experiment_plan.json").write_text("{}", encoding="utf-8")

    ready = plan / "S1_ALL_IN_DOMAIN" / "all"
    ready.mkdir(parents=True)
    (ready / "fold.json").write_text(
        json.dumps({"scenario": "S1_ALL_IN_DOMAIN", "fold_id": "all", "ready_for_model_building": True}),
        encoding="utf-8",
    )
    unready = plan / "S2_LOBO" / "background_BLUE"
    unready.mkdir(parents=True)
    (unready / "fold.json").write_text(
        json.dumps({"scenario": "S2_LOBO", "fold_id": "background_BLUE", "ready_for_model_building": False}),
        encoding="utf-8",
    )
    jobs, skipped = discover_model_build_jobs(plan)
    assert [(job.scenario, job.fold_id) for job in jobs] == [("S1_ALL_IN_DOMAIN", "all")]
    assert skipped[0]["reason"] == "training_cells_not_processed"
    assert resolve_model_build_worker_count(0, 20) <= 4
