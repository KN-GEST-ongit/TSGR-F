"""Compute clustered-by-take bootstrap intervals for IMAGE and VIDEO evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.frozen_bootstrap_v42 import bootstrap_frozen_end_to_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-report", type=Path, required=True)
    parser.add_argument("--video-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = bootstrap_frozen_end_to_end(
        image_report=args.image_report,
        video_report=args.video_report,
        output_dir=args.output_dir,
        samples=args.samples,
        seed=args.seed,
    )
    print(f"Bootstrap results saved to: {report}")


if __name__ == "__main__":
    main()
