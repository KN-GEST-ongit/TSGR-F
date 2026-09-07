"""Build independent reference-model sets for generated experiment folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsgr.config import load_config
from tsgr.experiments.model_building import build_experiment_models_parallel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_plan_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    scenario_scope = parser.add_mutually_exclusive_group(required=True)
    scenario_scope.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Explicitly build every scenario declared by the experiment plan.",
    )
    scenario_scope.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Build only the named scenario. Repeat for multiple scenarios.",
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--all-folds",
        action="store_true",
        help="Explicitly build every active fold selected by the scenario filters.",
    )
    scope.add_argument(
        "--fold",
        action="append",
        default=None,
        help="Build only the named fold. Repeat for multiple folds.",
    )
    parser.add_argument(
        "--branch",
        action="append",
        dest="branches",
        required=True,
        help="Feature branch to build. Repeat for multiple branches; no implicit branch is selected.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help=(
            "Optional persistent variant identifier. When set, models are written under "
            "reference_model_variants/<variant> so PLCC ablation models can coexist."
        ),
    )
    parser.add_argument(
        "--compact-correlation-threshold",
        type=float,
        required=True,
        help="Absolute Pearson-correlation threshold used to build the fold-local compact feature mask.",
    )
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of fold model-building processes. Use 0 for automatic selection (up to 4).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress as fold models finish.",
    )
    args = parser.parse_args()
    if not 0.0 < args.compact_correlation_threshold <= 1.0:
        parser.error("--compact-correlation-threshold must be in (0, 1].")
    if not (args.experiment_plan_dir / "experiment_plan.json").is_file():
        parser.error(f"Not an experiment plan directory: {args.experiment_plan_dir}")
    plan_payload = json.loads((args.experiment_plan_dir / "experiment_plan.json").read_text(encoding="utf-8"))
    if args.all_scenarios:
        selected_scenarios = {str(name) for name in plan_payload.get("scenarios", [])}
        if not selected_scenarios:
            parser.error("--all-scenarios was requested but the experiment plan declares no scenarios.")
    else:
        selected_scenarios = set(args.scenario or [])
    config = load_config(args.config)
    reference_config = config["reference_models"]
    reference_config["compact_feature_mask"]["correlation_threshold"] = args.compact_correlation_threshold
    payload = build_experiment_models_parallel(
        args.experiment_plan_dir,
        branches=args.branches,
        reference_config=reference_config,
        scenarios=selected_scenarios,
        folds=set(args.fold or []),
        force=args.force,
        workers=args.workers,
        progress=args.progress,
        model_variant=args.model_variant,
    )
    for row in payload["skip_records"]:
        print(f"[SKIP] {row['scenario']}/{row['fold_id']}: {row['reason']} -> {row['location']}")
    for row in payload["results"]:
        if row["status"] == "ok":
            print(f"[OK] {row['scenario']}/{row['fold_id']} -> {row['output_dir']}")
        else:
            print(f"[ERROR] {row['scenario']}/{row['fold_id']}: {row['error']}")
    print(f"Built: {payload['built']}, skipped: {payload['skipped']}, failed: {payload['failed']}")
    if payload["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
