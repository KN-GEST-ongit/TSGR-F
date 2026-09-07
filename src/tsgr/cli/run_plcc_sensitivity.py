"""Run post-hoc PLCC sensitivity for the fixed routing architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.config import load_config
from tsgr.evaluation.frozen_plcc_v42 import STANDARD_GRID, run_frozen_plcc_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-root-override", type=Path, default=None, help="Runtime dataset root for portable experiment plans.")
    parser.add_argument("--test-processing-report", type=Path, required=True)
    parser.add_argument("--ground-truth-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--scenario", action="append", required=True)
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--standard-grid", action="store_true")
    grid.add_argument("--plcc-threshold", action="append", type=float, default=None)
    parser.add_argument("--model-workers", type=int, default=0)
    parser.add_argument("--evaluation-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    thresholds = STANDARD_GRID if args.standard_grid else tuple(args.plcc_threshold or [])
    report = run_frozen_plcc_sensitivity(
        args.experiment_plan_dir,
        config=load_config(args.config),
        test_processing_report=args.test_processing_report,
        ground_truth_report=args.ground_truth_report,
        output_dir=args.output_dir,
        branch_name=args.branch,
        thresholds=thresholds,
        scenarios=set(args.scenario),
        model_workers=args.model_workers,
        evaluation_workers=args.evaluation_workers,
        evaluation_batch_size=args.evaluation_batch_size,
        dataset_root_override=args.dataset_root_override,
        resume=args.resume,
        progress=args.progress,
    )
    print(f"PLCC sensitivity results saved to: {report}")


if __name__ == "__main__":
    main()
