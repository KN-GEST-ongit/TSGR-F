"""Merge SGRF baseline evaluation tables with native TSGRF tables; TSGRF remains last."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.result_merge import merge_method_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation-dir", type=Path, required=True)
    parser.add_argument("--tsgrf-evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = merge_method_results(
        args.baseline_evaluation_dir,
        args.tsgrf_evaluation_dir,
        output_dir=args.output_dir,
    )
    print(f"Merged method-comparison tables saved to: {output}")


if __name__ == "__main__":
    main()
