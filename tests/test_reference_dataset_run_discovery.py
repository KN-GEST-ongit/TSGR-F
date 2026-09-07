from pathlib import Path

from tsgr.reference_models.manifest import locate_processed_run


def test_locate_processed_run_accepts_historical_folder_run_name(tmp_path: Path) -> None:
    session = tmp_path / "A" / "person_001" / "session_001"
    old_run = session / "runs" / "folder_run_20260101"
    features = old_run / "features"
    features.mkdir(parents=True)
    (features / "feature_arrays.npz").write_bytes(b"test")

    assert locate_processed_run(session) == old_run
