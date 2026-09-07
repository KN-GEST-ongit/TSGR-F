from __future__ import annotations

from pathlib import Path

from tsgr.reference_models.manifest import (
    ReferenceManifestEntry,
    read_reference_manifest,
    scan_reference_dataset,
    write_reference_manifest,
)


def test_manifest_round_trip_and_dataset_scan(tmp_path: Path) -> None:
    run = tmp_path / "A" / "person_001" / "session_001" / "runs" / "run_001"
    (run / "features").mkdir(parents=True)
    (run / "features" / "feature_arrays.npz").write_bytes(b"placeholder")
    scanned = scan_reference_dataset(tmp_path)
    assert len(scanned) == 1
    assert scanned[0].gesture_id == "A"
    manifest = write_reference_manifest(tmp_path / "reference_manifest.csv", scanned)
    loaded = read_reference_manifest(manifest)
    assert loaded == scanned


def test_manifest_rejects_duplicate_session_keys(tmp_path: Path) -> None:
    entries = [
        ReferenceManifestEntry("A", "p1", "s1", tmp_path / "run1"),
        ReferenceManifestEntry("A", "p1", "s1", tmp_path / "run2"),
    ]
    manifest = write_reference_manifest(tmp_path / "manifest.csv", entries)
    try:
        read_reference_manifest(manifest)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Expected duplicate manifest validation failure.")


def test_manifest_rejects_path_traversal_identifier(tmp_path: Path) -> None:
    entry = ReferenceManifestEntry("..", "p1", "s1", tmp_path / "run")
    try:
        write_reference_manifest(tmp_path / "manifest.csv", [entry])
    except ValueError as error:
        assert "portable identifier" in str(error)
    else:
        raise AssertionError("Expected path-traversal identifier validation failure.")
