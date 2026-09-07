"""Evaluate fixed ranking, routing, and training-only acceptance end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.frozen_end_to_end_v42 import evaluate_frozen_end_to_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--fixed-model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root-override", type=Path, default=None, help="Runtime dataset root for portable experiment plans.")
    parser.add_argument("--acceptance-dir", type=Path, required=True)
    parser.add_argument("--test-processing-report", type=Path, required=True)
    parser.add_argument("--ground-truth-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all-scenarios", action="store_true")
    scope.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--sequence-max-lag-frames", type=int, default=10)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--copy-problem-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-problem-cases-per-error-type-fold", type=int, default=20)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = evaluate_frozen_end_to_end(
        args.experiment_plan_dir,
        frozen_model_dir=args.fixed_model_dir,
        acceptance_dir=args.acceptance_dir,
        test_processing_report=args.test_processing_report,
        ground_truth_report=args.ground_truth_report,
        output_dir=args.output_dir,
        branch_name=args.branch,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=set(args.fold or []),
        sequence_max_lag_frames=args.sequence_max_lag_frames,
        prediction_batch_size=args.prediction_batch_size,
        copy_problem_cases=args.copy_problem_cases,
        max_problem_cases_per_error_type_fold=args.max_problem_cases_per_error_type_fold,
        dataset_root_override=args.dataset_root_override,
        progress=args.progress,
    )
    print(f"End-to-end evaluation saved to: {report}")


if __name__ == "__main__":
    main()
