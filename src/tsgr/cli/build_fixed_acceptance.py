"""Build training-only acceptance thresholds for fixed TSGR-F routing models."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.frozen_end_to_end_v42 import ACCEPTANCE_OBJECTIVES, build_frozen_acceptance_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--fixed-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all-scenarios", action="store_true")
    scope.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--objective", choices=ACCEPTANCE_OBJECTIVES, default="mcc")
    parser.add_argument("--min-positive-recall", type=float, default=0.99)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = build_frozen_acceptance_calibration(
        args.experiment_plan_dir,
        frozen_model_dir=args.fixed_model_dir,
        output_dir=args.output_dir,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=set(args.fold or []),
        objective=args.objective,
        min_positive_recall=args.min_positive_recall,
        progress=args.progress,
    )
    print(f"Acceptance calibration saved to: {report}")


if __name__ == "__main__":
    main()
