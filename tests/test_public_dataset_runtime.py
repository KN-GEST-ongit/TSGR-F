from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tsgr.dataset.contract import PUBLIC_ANNOTATION_FIELDS, load_public_annotations, validate_public_dataset
from tsgr.dataset.downloader import download_dataset_release


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _avi_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (8, 8))
    assert writer.isOpened()
    writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _build_release(root: Path, tmp_path: Path) -> None:
    subject_name = "tsgr_dataset_v1.0__P01.zip"
    image = _jpeg_bytes()
    video = _avi_bytes(tmp_path)
    with zipfile.ZipFile(root / subject_name, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("tsgr_dataset/training/P01/BLACK/A/images/image_000001.jpg", image)
        archive.writestr("tsgr_dataset/testing/P01/BLACK/A/take_0001/frames/frame_000000.jpg", image)
        archive.writestr("tsgr_dataset/testing/P01/BLACK/A/take_0001/take_0001.avi", video)
    digest = hashlib.sha256((root / subject_name).read_bytes()).hexdigest()
    row = {
        "public_subject_id": "P01",
        "background": "BLACK",
        "gesture": "A",
        "take_id": "take_0001",
        "frame_count": 1,
        "gesture_start_frame": 0,
        "gesture_end_frame": 0,
    }
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=PUBLIC_ANNOTATION_FIELDS)
    writer.writeheader(); writer.writerow(row)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    json_bytes = json.dumps([row]).encode("utf-8")
    manifest = {
        "schema_version": "tsgr_public_dataset_subject_release_v1",
        "release_version": "v1.0",
        "subject_packages": [{"filename": subject_name, "subject": "P01", "sha256": digest}],
    }
    with zipfile.ZipFile(root / "tsgr_dataset_v1.0__manifest.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("tsgr_dataset/annotations.csv", csv_bytes)
        archive.writestr("tsgr_dataset/annotations.json", json_bytes)
        archive.writestr("tsgr_dataset_release/release_manifest.json", json.dumps(manifest))
        archive.writestr("tsgr_dataset_release/checksums.sha256", f"{digest}  {subject_name}\n")


def test_annotations_are_limited_to_public_fields(tmp_path: Path) -> None:
    root = tmp_path / "tsgr_dataset"
    root.mkdir()
    row = {
        "public_subject_id": "P01",
        "background": "BLACK",
        "gesture": "A",
        "take_id": "take_0001",
        "frame_count": 1,
        "gesture_start_frame": 0,
        "gesture_end_frame": 0,
    }
    with (root / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_ANNOTATION_FIELDS)
        writer.writeheader(); writer.writerow(row)
    (root / "annotations.json").write_text(json.dumps([row]), encoding="utf-8")
    loaded = load_public_annotations(root)
    assert tuple(loaded[0]) == PUBLIC_ANNOTATION_FIELDS


def test_local_release_installer_reconstructs_dataset(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _build_release(release, tmp_path)
    output = tmp_path / "tsgr_dataset"
    download_dataset_release(output, archive_dir=release, version="1.0")
    audit = validate_public_dataset(output, release_layout=True)
    assert audit.valid
    assert audit.training_images == 1
    assert audit.test_takes == 1
    assert audit.test_frames == 1
    assert audit.test_videos == 1


def test_working_dataset_audit_accepts_derived_runtime_artifacts(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _build_release(release, tmp_path)
    output = tmp_path / "tsgr_dataset"
    download_dataset_release(output, archive_dir=release, version="1.0")

    run = output / "training/P01/BLACK/A/runs/run_001/features"
    run.mkdir(parents=True)
    (run / "feature_arrays.npz").write_bytes(b"derived")
    report = output / "reports/training_processing/reference_training"
    report.mkdir(parents=True)
    (report / "processing_report.json").write_text("{}", encoding="utf-8")

    release_audit = validate_public_dataset(output, release_layout=True)
    working_audit = validate_public_dataset(output, release_layout=False)
    assert not release_audit.valid
    assert working_audit.valid
    assert working_audit.training_images == 1
    assert working_audit.test_takes == 1
