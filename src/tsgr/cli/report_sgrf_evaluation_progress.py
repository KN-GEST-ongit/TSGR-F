"""Report completed SGRF method x fold checkpoints without running inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.registry import comparable_method_ids
from tsgr.baselines.sgrf_distributed import collect_sgrf_progress, write_sgrf_progress_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    method_scope = parser.add_mutually_exclusive_group(required=True)
    method_scope.add_argument("--all-methods", action="store_true")
    method_scope.add_argument("--method", action="append", choices=comparable_method_ids(), default=None)
    scenario_scope = parser.add_mutually_exclusive_group(required=True)
    scenario_scope.add_argument("--all-scenarios", action="store_true")
    scenario_scope.add_argument("--scenario", action="append", default=None)
    fold_scope = parser.add_mutually_exclusive_group(required=True)
    fold_scope.add_argument("--all-folds", action="store_true")
    fold_scope.add_argument("--fold", action="append", default=None)
    args = parser.parse_args()
    progress = collect_sgrf_progress(
        args.experiment_plan_dir,
        evaluation_dir=args.evaluation_dir,
        methods=args.method,
        all_methods=args.all_methods,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=None if args.all_folds else set(args.fold or []),
    )
    if args.output_dir is not None:
        write_sgrf_progress_report(progress, args.output_dir)
    print(
        f"SGRF progress: jobs={progress['complete_job_count']}/{progress['expected_job_count']} "
        f"pending={progress['pending_job_count']}"
    )
    for scenario in progress["by_scenario"]:
        print(f"  {scenario['scenario']}: {scenario['complete_jobs']}/{scenario['expected_jobs']}")
    print("Methods:")
    for method in progress["by_method"]:
        status = "COMPLETE" if method["complete"] else "pending"
        print(f"  {method['method_id']:<20} {method['complete_jobs']:>3}/{method['expected_jobs']:<3} {status}")
    if args.output_dir is not None:
        print(f"Progress report saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
