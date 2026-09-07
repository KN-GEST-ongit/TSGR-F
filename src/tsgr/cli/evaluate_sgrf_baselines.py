"""Run and evaluate the thirteen SGRF baselines on canonical TSGR-F test frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.registry import comparable_method_ids
from tsgr.baselines.sgrf_evaluation import evaluate_sgrf_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--sgrf-python", type=Path, required=True)
    parser.add_argument("--baseline-models-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root-override", type=Path, default=None, help="Portable override for the dataset root stored in experiment_plan.json, useful on a second computer.")
    method_scope = parser.add_mutually_exclusive_group(required=True)
    method_scope.add_argument("--all-methods", action="store_true")
    method_scope.add_argument("--method", action="append", choices=comparable_method_ids(), default=None)
    scenario_scope = parser.add_mutually_exclusive_group(required=True)
    scenario_scope.add_argument("--all-scenarios", action="store_true")
    scenario_scope.add_argument("--scenario", action="append", default=None)
    fold_scope = parser.add_mutually_exclusive_group(required=True)
    fold_scope.add_argument("--all-folds", action="store_true")
    fold_scope.add_argument("--fold", action="append", default=None)
    parser.add_argument("--rejection-policy", choices=("closed_set", "certainty_reject"), required=True)
    parser.add_argument("--certainty-threshold-normalized", type=float, default=None, help="Normalized rejection threshold in [0,1]; upstream certainty scales are normalized by the adapter.")
    parser.add_argument("--model-load-policy", choices=("process_cache", "upstream_each_call"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, required=True, help="Concurrent external SGRF inference subprocesses; start with 1 on CPU/TensorFlow workloads.")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--execution-order",
        choices=("fast_first", "registry"),
        default="fast_first",
        help="Execution scheduling only; publication/table method order is unchanged.",
    )
    parser.add_argument(
        "--aggregate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not run inference; fail if any selected method x fold checkpoint is missing, then rebuild aggregate outputs.",
    )
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonreference-python", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-nonreference-sgrf", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be a positive integer; use 1 for the reference run.")
    if args.rejection_policy == "certainty_reject" and args.certainty_threshold_normalized is None:
        parser.error("--certainty-threshold-normalized is required with --rejection-policy certainty_reject.")
    if args.rejection_policy == "closed_set" and args.certainty_threshold_normalized is not None:
        parser.error("--certainty-threshold-normalized is not used with --rejection-policy closed_set.")
    report = evaluate_sgrf_baselines(
        args.experiment_plan_dir,
        sgrf_python=args.sgrf_python,
        baseline_models_dir=args.baseline_models_dir,
        ground_truth_report=args.ground_truth_report,
        output_dir=args.output_dir,
        dataset_root_override=args.dataset_root_override,
        methods=args.method,
        all_methods=args.all_methods,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=None if args.all_folds else set(args.fold or []),
        rejection_policy=args.rejection_policy,
        certainty_threshold_normalized=args.certainty_threshold_normalized,
        model_load_policy=args.model_load_policy,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
        fail_fast=args.fail_fast,
        progress=args.progress,
        progress_every=args.progress_every,
        allow_nonreference_python=args.allow_nonreference_python,
        allow_nonreference_sgrf=args.allow_nonreference_sgrf,
        execution_order=args.execution_order,
        aggregate_only=args.aggregate_only,
    )
    print(f"SGRF baseline evaluation saved to: {report}")


if __name__ == "__main__":
    main()
