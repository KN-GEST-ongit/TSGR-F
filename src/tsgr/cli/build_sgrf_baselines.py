"""Train fold-local models for the thirteen comparable upstream SGRF methods."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.baselines.registry import comparable_method_ids
from tsgr.baselines.sgrf_models import build_sgrf_baseline_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--sgrf-python", type=Path, required=True)
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
    parser.add_argument("--method-options-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, required=True, help="Concurrent external SGRF training subprocesses; reference recommendation is 1.")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonreference-python", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-nonreference-sgrf", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be a positive integer; use 1 for the reference run.")
    report = build_sgrf_baseline_models(
        args.experiment_plan_dir,
        sgrf_python=args.sgrf_python,
        output_dir=args.output_dir,
        dataset_root_override=args.dataset_root_override,
        methods=args.method,
        all_methods=args.all_methods,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=None if args.all_folds else set(args.fold or []),
        method_options_json=args.method_options_json,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
        fail_fast=args.fail_fast,
        progress=args.progress,
        allow_nonreference_python=args.allow_nonreference_python,
        allow_nonreference_sgrf=args.allow_nonreference_sgrf,
    )
    print(f"SGRF baseline models saved to: {report}")


if __name__ == "__main__":
    main()
