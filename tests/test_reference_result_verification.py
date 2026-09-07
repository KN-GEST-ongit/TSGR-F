from __future__ import annotations

import csv
import json
from pathlib import Path

from tsgr.evaluation.reference_result_verification import verify_reference_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_reference_verifier_accepts_semantically_equal_json_and_csv(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_csv(reference / "table.csv", [{"scenario": "S4", "accuracy": "0.963"}])
    _write_csv(candidate / "table.csv", [{"scenario": "S4", "accuracy": "0.963"}])
    (reference / "report.json").write_text(json.dumps({"b": 2, "a": 1}, indent=2), encoding="utf-8")
    (candidate / "report.json").write_text('{"a":1,"b":2}\n', encoding="utf-8")

    result = verify_reference_results(reference, candidate, output_dir=tmp_path / "audit")
    assert result.matched
    assert result.mismatch_count == 0
    assert result.compared_file_count == 2


def test_reference_verifier_reports_missing_and_changed_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_csv(reference / "table.csv", [{"value": "1"}])
    _write_csv(candidate / "table.csv", [{"value": "2"}])
    (reference / "required.json").write_text("{}", encoding="utf-8")

    result = verify_reference_results(reference, candidate)
    reasons = {item.reason for item in result.mismatches}
    assert not result.matched
    assert "csv_rows_mismatch" in reasons
    assert "missing_candidate_file" in reasons


def test_reference_verifier_ignores_non_machine_readable_files(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    (reference / "README.md").write_text("reference", encoding="utf-8")
    (candidate / "README.md").write_text("different", encoding="utf-8")

    result = verify_reference_results(reference, candidate)
    assert result.matched
    assert result.compared_file_count == 0
