"""Download and assemble the public TSGR-F dataset release."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from tsgr.dataset.contract import validate_public_dataset

DEFAULT_REPOSITORY = "KN-GEST-ongit/TSGR-F"
DEFAULT_DATASET_VERSION = "1.0"
RELEASE_SCHEMA = "tsgr_public_dataset_subject_release_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_url(repository: str, tag: str, filename: str) -> str:
    return f"https://github.com/{repository}/releases/download/{tag}/{filename}"


def _obtain_asset(
    filename: str,
    *,
    workspace: Path,
    repository: str,
    tag: str,
    archive_dir: Path | None,
) -> Path:
    destination = workspace / filename
    if archive_dir is not None:
        source = archive_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Dataset release asset does not exist: {source}")
        shutil.copy2(source, destination)
        return destination
    url = _asset_url(repository, tag, filename)
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as error:  # noqa: BLE001 - preserve the original network error context.
        raise RuntimeError(f"Failed to download {url}: {error}") from error
    return destination


def _read_manifest(manifest_zip: Path) -> tuple[dict, dict[str, str], bytes, bytes]:
    with zipfile.ZipFile(manifest_zip, "r") as archive:
        names = set(archive.namelist())
        required = {
            "tsgr_dataset/annotations.csv",
            "tsgr_dataset/annotations.json",
            "tsgr_dataset_release/release_manifest.json",
            "tsgr_dataset_release/checksums.sha256",
        }
        missing = required - names
        if missing:
            raise ValueError(f"Dataset manifest archive is missing entries: {sorted(missing)}")
        payload = json.loads(archive.read("tsgr_dataset_release/release_manifest.json").decode("utf-8"))
        checksums: dict[str, str] = {}
        for line in archive.read("tsgr_dataset_release/checksums.sha256").decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, filename = line.split(maxsplit=1)
            checksums[filename.strip()] = digest.strip().lower()
        annotations_csv = archive.read("tsgr_dataset/annotations.csv")
        annotations_json = archive.read("tsgr_dataset/annotations.json")
    if payload.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(f"Unsupported dataset release schema: {payload.get('schema_version')!r}")
    return payload, checksums, annotations_csv, annotations_json


def _safe_member_destination(output_root: Path, member_name: str) -> Path | None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {member_name}")
    if not path.parts or path.parts[0] != "tsgr_dataset":
        return None
    relative = Path(*path.parts[1:])
    destination = (output_root / relative).resolve()
    root = output_root.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Archive member escapes the dataset root: {member_name}")
    return destination


def _extract_dataset_members(archive_path: Path, output_root: Path, *, overwrite: bool) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            destination = _safe_member_destination(output_root, info.filename)
            if destination is None:
                continue
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing dataset file: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def download_dataset_release(
    output_dir: str | Path,
    *,
    version: str = DEFAULT_DATASET_VERSION,
    subjects: Iterable[str] | None = None,
    repository: str = DEFAULT_REPOSITORY,
    tag: str | None = None,
    archive_dir: str | Path | None = None,
    overwrite: bool = False,
    validate: bool = True,
) -> Path:
    """Download or install subject packages and assemble one public dataset tree."""
    version_text = str(version).strip().lstrip("v")
    if not version_text:
        raise ValueError("Dataset version cannot be empty.")
    release_tag = tag or f"dataset-v{version_text}"
    local_archive_dir = Path(archive_dir) if archive_dir is not None else None
    if local_archive_dir is not None and not local_archive_dir.is_dir():
        raise FileNotFoundError(f"Dataset archive directory does not exist: {local_archive_dir}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_filename = f"tsgr_dataset_v{version_text}__manifest.zip"

    with tempfile.TemporaryDirectory(prefix="tsgr_dataset_download_") as temporary:
        workspace = Path(temporary)
        manifest_zip = _obtain_asset(
            manifest_filename,
            workspace=workspace,
            repository=repository,
            tag=release_tag,
            archive_dir=local_archive_dir,
        )
        manifest, checksums, annotations_csv, annotations_json = _read_manifest(manifest_zip)
        available = {str(item["subject"]).upper(): dict(item) for item in manifest.get("subject_packages", [])}
        selected = [str(value).upper() for value in subjects] if subjects else sorted(available)
        unknown = [subject for subject in selected if subject not in available]
        if unknown:
            raise ValueError(f"Unknown subject package(s): {unknown}; available: {sorted(available)}")

        # Install the annotation contract from the manifest package first.
        annotations_csv_path = output_root / "annotations.csv"
        annotations_json_path = output_root / "annotations.json"
        for path, data in ((annotations_csv_path, annotations_csv), (annotations_json_path, annotations_json)):
            if path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing dataset file: {path}")
            path.write_bytes(data)

        for subject in selected:
            item = available[subject]
            filename = str(item["filename"])
            package = _obtain_asset(
                filename,
                workspace=workspace,
                repository=repository,
                tag=release_tag,
                archive_dir=local_archive_dir,
            )
            expected = str(item.get("sha256") or checksums.get(filename) or "").lower()
            if not expected:
                raise ValueError(f"No SHA-256 is available for dataset asset {filename}.")
            actual = _sha256(package)
            if actual != expected:
                raise ValueError(f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}.")
            _extract_dataset_members(package, output_root, overwrite=overwrite)

    if validate:
        full_release = set(selected) == set(available)
        audit = validate_public_dataset(
            output_root,
            require_videos=True,
            require_annotations=True,
            release_layout=full_release,
        )
        if not audit.valid:
            raise RuntimeError("Installed dataset failed validation: " + "; ".join(audit.errors[:10]))
    return output_root.resolve()
