from __future__ import annotations

from tsgr.processing import test_take_audit as module


def test_chunk_worker_closes_pipeline_before_return(monkeypatch) -> None:
    created = []

    class FakePipeline:
        def __init__(self, config, model_path):
            self.config = config
            self.model_path = model_path
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    def fake_process(task):
        assert module._WORKER_PIPELINE is created[0]
        return {
            "take_row": {
                "take_id": task["id"],
                "take_mediapipe_status": "pass",
                "elapsed_wall_time_s": 0.1,
                "worker_pid": 123,
            },
            "failure_rows": [],
        }

    monkeypatch.setattr(module, "FramePipeline", FakePipeline)
    monkeypatch.setattr(module, "_worker_process_take", fake_process)
    result = module._worker_process_chunk(
        {"temporal_filter": {}, "classification_input": {}},
        "model.task",
        [{"id": "one"}, {"id": "two"}],
        progress=False,
        worker_slot=1,
    )
    assert len(result["results"]) == 2
    assert result["close_error"] == ""
    assert created[0].closed is True
    assert module._WORKER_PIPELINE is None


def test_partition_tasks_is_balanced_and_complete() -> None:
    tasks = [{"id": index} for index in range(10)]
    chunks = module._partition_tasks(tasks, 3)
    assert sorted(item["id"] for chunk in chunks for item in chunk) == list(range(10))
    assert [len(chunk) for chunk in chunks] == [4, 3, 3]
