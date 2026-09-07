"""Verify that the selected SGRF evaluation is complete at method x fold and aggregate level."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.registry import comparable_method_ids
from tsgr.baselines.sgrf_portability import audit_sgrf_evaluation_completeness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--baseline-models-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    method_scope = parser.add_mutually_exclusive_group(required=True)
    method_scope.add_argument("--all-methods", action="store_true")
    method_scope.add_argument("--method", action="append", choices=comparable_method_ids(), default=None)
    scenario_scope = parser.add_mutually_exclusive_group(required=True)
    scenario_scope.add_argument("--all-scenarios", action="store_true")
    scenario_scope.add_argument("--scenario", action="append", default=None)
    fold_scope = parser.add_mutually_exclusive_group(required=True)
    fold_scope.add_argument("--all-folds", action="store_true")
    fold_scope.add_argument("--fold", action="append", default=None)
    parser.add_argument("--deep-row-count", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    summary = audit_sgrf_evaluation_completeness(
        args.experiment_plan_dir,
        baseline_models_dir=args.baseline_models_dir,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        methods=args.method,
        all_methods=args.all_methods,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=None if args.all_folds else set(args.fold or []),
        deep_row_count=args.deep_row_count,
        require_complete=args.require_complete,
    )
    status = "COMPLETE" if summary["complete"] else "INCOMPLETE"
    print(
        f"SGRF evaluation audit: {status} | jobs={summary['complete_job_count']}/{summary['expected_job_count']}"
    )
    print(f"Audit saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
