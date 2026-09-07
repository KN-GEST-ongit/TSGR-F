"""Verify reproduced CSV/JSON results against published TSGR-F reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.reference_result_verification import verify_reference_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_results", type=Path)
    parser.add_argument("candidate_results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-match", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = verify_reference_results(
        args.reference_results,
        args.candidate_results,
        output_dir=args.output_dir,
    )
    status = "MATCH" if result.matched else "MISMATCH"
    print(
        f"Reference-result verification: {status} | "
        f"files={result.compared_file_count} | mismatches={result.mismatch_count}"
    )
    if args.require_match and not result.matched:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
