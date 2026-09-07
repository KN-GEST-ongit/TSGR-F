"""Dataset manifest helpers for reference-model training sessions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_FIELDS = (
    "gesture_id",
    "person_id",
    "session_id",
    "run_path",
    "include",
    "notes",
)


@dataclass(frozen=True, slots=True)
class ReferenceManifestEntry:
    """One processed recording session assigned to a gesture and participant."""

    gesture_id: str
    person_id: str
    session_id: str
    run_path: Path
    include: bool = True
    notes: str = ""

    def validate(self) -> None:
        for field_name, value in (
            ("gesture_id", self.gesture_id),
            ("person_id", self.person_id),
            ("session_id", self.session_id),
        ):
            if not value.strip():
                raise ValueError(f"Manifest field {field_name} cannot be empty.")
            if any(character in value for character in ("/", "\\")):
                raise ValueError(
                    f"Manifest field {field_name} cannot contain path separators: {value!r}."
                )
            if value in {".", ".."} or any(
                character in value for character in '<>:"|?*'
            ):
                raise ValueError(
                    f"Manifest field {field_name} is not a portable identifier: {value!r}."
                )
            if value.endswith((" ", ".")):
                raise ValueError(
                    f"Manifest field {field_name} cannot end with a space or dot: {value!r}."
                )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "include"}:
        return True
    if normalized in {"0", "false", "no", "n", "exclude"}:
        return False
    raise ValueError(f"Cannot parse manifest boolean value: {value!r}.")


def read_reference_manifest(path: str | Path) -> list[ReferenceManifestEntry]:
    """Load and validate a portable CSV manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reference manifest does not exist: {manifest_path}")
    entries: list[ReferenceManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Reference manifest is missing columns: " + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            raw_path = Path((row.get("run_path") or "").strip())
            if not raw_path.is_absolute():
                raw_path = (manifest_path.parent / raw_path).resolve()
            entry = ReferenceManifestEntry(
                gesture_id=(row.get("gesture_id") or "").strip(),
                person_id=(row.get("person_id") or "").strip(),
                session_id=(row.get("session_id") or "").strip(),
                run_path=raw_path,
                include=_parse_bool(row.get("include") or "true"),
                notes=(row.get("notes") or "").strip(),
            )
            try:
                entry.validate()
            except ValueError as error:
                raise ValueError(f"Invalid manifest row {row_number}: {error}") from error
            entries.append(entry)
    if not entries:
        raise ValueError(f"Reference manifest is empty: {manifest_path}")
    duplicates: set[tuple[str, str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        key = (entry.gesture_id, entry.person_id, entry.session_id)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        detail = ", ".join("/".join(item) for item in sorted(duplicates))
        raise ValueError(f"Duplicate gesture/person/session entries: {detail}")
    return entries


def write_reference_manifest(
    path: str | Path,
    entries: Iterable[ReferenceManifestEntry],
    *,
    make_paths_relative: bool = True,
) -> Path:
    """Write a deterministic manifest sorted by gesture, person, and session."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        entries,
        key=lambda entry: (entry.gesture_id, entry.person_id, entry.session_id),
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in ordered:
            entry.validate()
            run_path = entry.run_path
            if make_paths_relative:
                try:
                    run_path = run_path.resolve().relative_to(manifest_path.parent.resolve())
                except ValueError:
                    run_path = Path(
                        __import__("os").path.relpath(
                            entry.run_path.resolve(), manifest_path.parent.resolve()
                        )
                    )
            writer.writerow(
                {
                    "gesture_id": entry.gesture_id,
                    "person_id": entry.person_id,
                    "session_id": entry.session_id,
                    "run_path": str(run_path),
                    "include": "true" if entry.include else "false",
                    "notes": entry.notes,
                }
            )
    return manifest_path


def _contains_feature_artifacts(path: Path) -> bool:
    return (
        (path / "features" / "feature_arrays.npz").is_file()
        or (path / "feature_arrays.npz").is_file()
    )


def locate_processed_run(session_path: Path) -> Path | None:
    """Locate the newest processed feature run under one session directory.

    The feature artifact itself is authoritative, including artifacts stored
    in nested processed-run directories.
    """
    if _contains_feature_artifacts(session_path):
        return session_path
    candidates: set[Path] = set()
    for artifact in session_path.rglob("feature_arrays.npz"):
        if artifact.parent.name == "features":
            candidate = artifact.parent.parent
        else:
            candidate = artifact.parent
        if any(part.startswith(".partial_") for part in candidate.parts):
            continue
        if candidate.is_dir() and _contains_feature_artifacts(candidate):
            candidates.add(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def scan_reference_dataset(dataset_root: str | Path) -> list[ReferenceManifestEntry]:
    """Scan the canonical gesture/person/session hierarchy for processed runs."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Reference dataset root does not exist: {root}")
    entries: list[ReferenceManifestEntry] = []
    for gesture_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for person_dir in sorted(path for path in gesture_dir.iterdir() if path.is_dir()):
            for session_dir in sorted(path for path in person_dir.iterdir() if path.is_dir()):
                run_path = locate_processed_run(session_dir)
                if run_path is None:
                    continue
                entries.append(
                    ReferenceManifestEntry(
                        gesture_id=gesture_dir.name,
                        person_id=person_dir.name,
                        session_id=session_dir.name,
                        run_path=run_path.resolve(),
                    )
                )
    return entries
