"""Merge external SGRF baseline tables with native TSGRF results for paper-ready comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tsgr.evaluation.io_utils import read_csv_rows, write_csv_rows, write_workbook
from tsgr.utils.serialization import write_json


def _fields(rows: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        for key in row:
            if key not in output:
                output.append(key)
    return output


def _normalize(rows: list[dict[str, str]], *, tsgrf: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        if tsgrf:
            item["method_id"] = "TSGRF"
            item["method_display_name"] = "TSGRF"
            item["method_sort_order"] = 9999
        else:
            item.setdefault("method_display_name", item.get("method_id", ""))
        try:
            item["method_sort_order"] = int(float(item.get("method_sort_order", 9998)))
        except (TypeError, ValueError):
            item["method_sort_order"] = 9998
        output.append(item)
    return output


def merge_method_results(
    baseline_evaluation_dir: str | Path,
    tsgrf_evaluation_dir: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    baseline = Path(baseline_evaluation_dir)
    tsgrf = Path(tsgrf_evaluation_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = ("scenario_summary", "fold_summary", "gesture_metrics", "group_summary", "video_summary")
    workbook: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    counts: dict[str, int] = {}
    for table in tables:
        baseline_rows = _normalize(read_csv_rows(baseline / f"{table}.csv"), tsgrf=False)
        tsgrf_rows = _normalize(read_csv_rows(tsgrf / f"{table}.csv"), tsgrf=True)
        if not baseline_rows:
            raise ValueError(f"Missing or empty baseline table: {baseline / f'{table}.csv'}")
        if not tsgrf_rows:
            raise ValueError(f"Missing or empty TSGRF table: {tsgrf / f'{table}.csv'}")
        rows = [*baseline_rows, *tsgrf_rows]
        rows.sort(
            key=lambda row: (
                int(row.get("method_sort_order", 9998)),
                str(row.get("scenario", "")),
                str(row.get("fold_id", "")),
                str(row.get("gesture_id", "")),
                str(row.get("take_id", "")),
            )
        )
        fields = _fields(rows)
        write_csv_rows(output / f"comparison_{table}.csv", rows, fields)
        workbook[table] = (fields, rows)
        counts[table] = len(rows)
    write_workbook(output / "method_comparison_results.xlsx", workbook)
    write_json(
        output / "comparison_manifest.json",
        {
            "schema_version": "tsgr_method_comparison_v1",
            "baseline_evaluation_dir": baseline.name,
            "tsgrf_evaluation_dir": tsgrf.name,
            "table_rows": counts,
            "ordering_rule": "External SGRF methods follow registry sort_order; TSGRF is always appended last with sort_order=9999.",
        },
    )
    return output
