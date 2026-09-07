"""Benchmark fixed TSGR-F routing runtime and model footprint on cached features."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.frozen_runtime_v42 import benchmark_frozen_routing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--fixed-model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root-override", type=Path, default=None, help="Runtime dataset root for portable experiment plans.")
    parser.add_argument("--test-processing-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all-scenarios", action="store_true")
    scope.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--max-frames-per-fold", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = benchmark_frozen_routing(
        args.experiment_plan_dir,
        frozen_model_dir=args.fixed_model_dir,
        test_processing_report=args.test_processing_report,
        output_dir=args.output_dir,
        branch_name=args.branch,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=set(args.fold or []),
        max_frames_per_fold=args.max_frames_per_fold,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup=args.warmup,
        dataset_root_override=args.dataset_root_override,
        progress=args.progress,
    )
    print(f"Routing benchmark saved to: {report}")


if __name__ == "__main__":
    main()
