"""Build leakage-safe fixed routing models using training data only."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.evaluation.frozen_routing_v40 import build_frozen_routing_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root-override", type=Path, default=None, help="Runtime dataset root for portable experiment plans.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all-scenarios", action="store_true")
    scope.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--feature-set", choices=("active", "compact"), default="compact")
    parser.add_argument("--orientation-mode", default="camera_aware")
    parser.add_argument("--os-alpha", type=float, default=0.65)
    parser.add_argument("--c-top-n", type=int, default=5)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = build_frozen_routing_models(
        args.experiment_plan_dir,
        output_dir=args.output_dir,
        branch_name=args.branch,
        feature_set=args.feature_set,
        orientation_mode=args.orientation_mode,
        scenarios=None if args.all_scenarios else set(args.scenario or []),
        folds=set(args.fold or []),
        os_alpha=args.os_alpha,
        c_top_n=args.c_top_n,
        model_variant=args.model_variant,
        dataset_root_override=args.dataset_root_override,
        progress=args.progress,
    )
    print(f"Fixed routing models saved to: {report}")


if __name__ == "__main__":
    main()
