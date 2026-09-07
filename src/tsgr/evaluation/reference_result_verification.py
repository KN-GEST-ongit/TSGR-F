"""Verify machine-readable result trees against published reference outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".csv", ".json"}


@dataclass(frozen=True)
class ReferenceResultMismatch:
    relative_path: str
    reason: str
    expected: str = ""
    actual: str = ""


@dataclass(frozen=True)
class ReferenceResultVerification:
    reference_root: str
    candidate_root: str
    matched: bool
    compared_file_count: int
    mismatch_count: int
    mismatches: tuple[ReferenceResultMismatch, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mismatches"] = [asdict(item) for item in self.mismatches]
        return payload


def _files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, [dict(row) for row in reader]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _short(value: Any, limit: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def verify_reference_results(
    reference_root: str | Path,
    candidate_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> ReferenceResultVerification:
    """Compare CSV and JSON outputs using parsed semantic content."""
    reference = Path(reference_root).resolve()
    candidate = Path(candidate_root).resolve()
    if not reference.is_dir():
        raise FileNotFoundError(f"Reference result directory does not exist: {reference}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Candidate result directory does not exist: {candidate}")

    expected_files = _files(reference)
    actual_files = _files(candidate)
    mismatches: list[ReferenceResultMismatch] = []

    for relative_path in sorted(set(expected_files) - set(actual_files)):
        mismatches.append(ReferenceResultMismatch(relative_path, "missing_candidate_file"))
    for relative_path in sorted(set(actual_files) - set(expected_files)):
        mismatches.append(ReferenceResultMismatch(relative_path, "unexpected_candidate_file"))

    compared = 0
    for relative_path in sorted(set(expected_files) & set(actual_files)):
        expected_path = expected_files[relative_path]
        actual_path = actual_files[relative_path]
        compared += 1
        try:
            if expected_path.suffix.lower() == ".csv":
                expected_fields, expected_rows = _load_csv(expected_path)
                actual_fields, actual_rows = _load_csv(actual_path)
                if expected_fields != actual_fields:
                    mismatches.append(
                        ReferenceResultMismatch(
                            relative_path,
                            "csv_header_mismatch",
                            _short(expected_fields),
                            _short(actual_fields),
                        )
                    )
                    continue
                if expected_rows != actual_rows:
                    mismatches.append(
                        ReferenceResultMismatch(
                            relative_path,
                            "csv_rows_mismatch",
                            _short(expected_rows),
                            _short(actual_rows),
                        )
                    )
            else:
                expected_payload = _load_json(expected_path)
                actual_payload = _load_json(actual_path)
                if expected_payload != actual_payload:
                    mismatches.append(
                        ReferenceResultMismatch(
                            relative_path,
                            "json_payload_mismatch",
                            _short(expected_payload),
                            _short(actual_payload),
                        )
                    )
        except Exception as error:  # noqa: BLE001 - preserve the comparison failure in the report.
            mismatches.append(
                ReferenceResultMismatch(relative_path, f"comparison_error:{type(error).__name__}:{error}")
            )

    verification = ReferenceResultVerification(
        reference_root=str(reference),
        candidate_root=str(candidate),
        matched=not mismatches,
        compared_file_count=compared,
        mismatch_count=len(mismatches),
        mismatches=tuple(mismatches),
    )

    if output_dir is not None:
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "reference_result_verification.json").write_text(
            json.dumps(verification.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (report_dir / "reference_result_mismatches.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "reason", "expected", "actual"])
            writer.writeheader()
            for mismatch in mismatches:
                writer.writerow(asdict(mismatch))
    return verification
