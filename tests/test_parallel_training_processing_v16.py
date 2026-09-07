from __future__ import annotations

import os

import pytest

from tsgr.processing import training_photo_audit as module


def test_resolve_training_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    assert module.resolve_training_worker_count(0, 100) == 8
    assert module.resolve_training_worker_count(4, 100) == 4
    assert module.resolve_training_worker_count(99, 3) == 3
    assert module.resolve_training_worker_count(0, 0) == 0
    with pytest.raises(ValueError):
        module.resolve_training_worker_count(-1, 10)


def test_partition_training_tasks_balances_image_counts() -> None:
    tasks = [
        {"subject": "P01", "background": "BLUE", "gesture": "A", "image_count": 40},
        {"subject": "P01", "background": "BLUE", "gesture": "B", "image_count": 30},
        {"subject": "P01", "background": "BLUE", "gesture": "C", "image_count": 20},
        {"subject": "P01", "background": "BLUE", "gesture": "E", "image_count": 10},
    ]
    chunks = module.partition_training_tasks(tasks, 2)
    assert len(chunks) == 2
    loads = sorted(sum(int(item["image_count"]) for item in chunk) for chunk in chunks)
    assert loads == [50, 50]


def test_worker_chunk_closes_pipeline_before_return(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def fake_initialize(config, model_path, detection_profile):
        events.append("initialize")
        module._WORKER_PIPELINE = object()  # type: ignore[assignment]
        module._WORKER_CONFIG = {}
        module._WORKER_MODEL_PATH = str(model_path)

    def fake_process(task):
        events.append(f"process:{task['gesture']}")
        return {
            "status": "ok",
            "subject": "P01",
            "background": "BLUE",
            "gesture": task["gesture"],
            "run_path": "run",
            "image_count": 25,
            "audit_rows": [],
            "worker_pid": 123,
            "worker_model_initialization_s": 0.1,
            "elapsed_wall_time_s": 1.0,
            "images_per_second": 25.0,
        }

    def fake_close():
        events.append("close")
        module._WORKER_PIPELINE = None

    monkeypatch.setattr(module, "_worker_initialize", fake_initialize)
    monkeypatch.setattr(module, "_worker_process_cell", fake_process)
    monkeypatch.setattr(module, "_close_worker_pipeline", fake_close)

    result = module._worker_process_chunk(
        {}, "model.task", "high_recall", [{"gesture": "A"}, {"gesture": "B"}], False, 1
    )
    assert [item["gesture"] for item in result["results"]] == ["A", "B"]
    assert events == ["initialize", "process:A", "process:B", "close"]
    assert result["close_error"] == ""


def test_image_profile_match_requires_strict_image_mode(tmp_path) -> None:
    run = tmp_path / "run_1"
    run.mkdir()
    (run / "run_summary.json").write_text(
        '{"mediapipe_running_mode":"image","mediapipe_image_profile":"high_recall","strict_right_hand_only":true}',
        encoding="utf-8",
    )
    assert module._run_matches_image_profile(run, "high_recall")
    assert not module._run_matches_image_profile(run, "balanced")


def test_training_inventory_signature_invalidates_stale_run(tmp_path) -> None:
    import csv
    import json

    cell = tmp_path / "cell"
    cell.mkdir()
    manifest = cell / "images_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "filename", "sha256"])
        writer.writeheader()
        writer.writerow({"image_id": "P01_BLUE_A_000001", "filename": "image_000001.jpg", "sha256": "aaa"})

    run = tmp_path / "run_1"
    run.mkdir()
    (run / "run_summary.json").write_text(
        '{"mediapipe_running_mode":"image","mediapipe_image_profile":"high_recall","strict_right_hand_only":true}',
        encoding="utf-8",
    )
    signature = module.training_cell_inventory_signature(cell)
    (run / "training_cell_processing_signature.json").write_text(json.dumps(signature), encoding="utf-8")
    assert module._run_matches_image_profile(run, "high_recall", cell)

    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "filename", "sha256"])
        writer.writerow({"image_id": "P01_BLUE_A_000002", "filename": "image_000002.jpg", "sha256": "bbb"})
    assert not module._run_matches_image_profile(run, "high_recall", cell)
