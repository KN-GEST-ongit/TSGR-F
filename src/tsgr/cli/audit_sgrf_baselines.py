"""Audit the isolated Python 3.11 / SGRF 3.2.0 environment used for external baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.registry import registry_rows
from tsgr.baselines.sgrf_bridge import audit_sgrf_environment
from tsgr.evaluation.io_utils import write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgrf-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonreference-python", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-nonreference-sgrf", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_sgrf_environment(
        args.sgrf_python,
        output_dir=args.output_dir,
        allow_nonreference_python=args.allow_nonreference_python,
        allow_nonreference_sgrf=args.allow_nonreference_sgrf,
    )
    write_csv_rows(
        args.output_dir / "method_registry.csv",
        registry_rows(),
        ["method_id", "display_name", "sort_order", "payload_kind", "learning_data_kind", "coordinate_policy", "certainty_scale", "included", "exclusion_reason"],
    )
    print(f"Python: {audit.python_version} -> {audit.python_executable}")
    print(f"SGRF: {audit.sgrf_version}")
    print(f"Import OK: {audit.import_ok}")
    print(f"Audit saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
