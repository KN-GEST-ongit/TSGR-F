from __future__ import annotations

from pathlib import Path

from tsgr.processing.test_take_audit import resolve_worker_count
from tsgr.reference_models.manifest import locate_processed_run


def _feature_run(path: Path) -> None:
    features = path / "features"
    features.mkdir(parents=True)
    (features / "feature_arrays.npz").write_bytes(b"placeholder")


def test_worker_count_is_bounded_and_auto_is_conservative(monkeypatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert resolve_worker_count(0, 20) == 8
    assert resolve_worker_count(8, 3) == 3
    assert resolve_worker_count(1, 10) == 1
    assert resolve_worker_count(0, 0) == 0


def test_partial_runs_are_not_reused(tmp_path: Path) -> None:
    session = tmp_path / "take_0001"
    partial = session / "runs" / ".partial_run_001"
    complete = session / "runs" / "run_002"
    _feature_run(partial)
    _feature_run(complete)
    assert locate_processed_run(session) == complete
