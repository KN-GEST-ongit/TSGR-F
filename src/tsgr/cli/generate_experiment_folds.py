"""Generate S1, LOBO, LOSO, and combined subject-background experiment folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.experiments.folds import SCENARIOS, generate_experiment_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Threads used to build the shared training-run index. Use 0 for automatic selection (up to 8).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one line after each generated fold.",
    )
    args = parser.parse_args()
    plan = generate_experiment_folds(
        args.dataset_root,
        output_dir=args.output_dir,
        scenarios=args.scenario,
        workers=args.workers,
        progress=args.progress,
    )
    print(f"Experiment plan saved to: {plan.resolve()}")


if __name__ == "__main__":
    main()
