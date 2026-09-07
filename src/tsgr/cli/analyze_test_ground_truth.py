"""Analyze test-video annotations plus the detector-derived hand-presence reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.dataset_ground_truth_analysis import analyze_test_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--hand-presence-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = analyze_test_ground_truth(
        args.dataset_root,
        hand_presence_reference_dir=args.hand_presence_reference,
        output_dir=args.output_dir,
    )
    print(f"Test ground-truth analysis saved to: {report}")


if __name__ == "__main__":
    main()
