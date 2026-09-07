from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tsgr.evaluation.frozen_end_to_end_v42 import evaluate_frozen_end_to_end
from tsgr.evaluation.frozen_plcc_v42 import run_frozen_plcc_sensitivity
from tsgr.evaluation.frozen_routing_v40 import build_frozen_routing_models, evaluate_frozen_routing
from tsgr.evaluation.frozen_runtime_v42 import benchmark_frozen_routing


def _portable_plan_and_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "data" / "tsgr_dataset"
    (dataset / "training").mkdir(parents=True)
    (dataset / "testing").mkdir()
    (dataset / "annotations.csv").write_text("take_id\n", encoding="utf-8")
    plan = tmp_path / "results" / "experiment_plan"
    plan.mkdir(parents=True)
    (plan / "experiment_plan.json").write_text(
        json.dumps({"dataset_root": ".", "scenarios": []}), encoding="utf-8"
    )
    return plan, dataset


def test_public_fixed_pipeline_accepts_runtime_dataset_override(tmp_path: Path) -> None:
    plan, dataset = _portable_plan_and_dataset(tmp_path)
    report = build_frozen_routing_models(
        plan,
        output_dir=tmp_path / "fixed",
        branch_name="raw.wrist_middle_mcp",
        dataset_root_override=dataset,
        progress=False,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["fold_count"] == 0
    assert Path(payload["dataset_root"]).resolve() == dataset.resolve()


def test_routing_evaluator_uses_override_before_report_validation(tmp_path: Path) -> None:
    plan, dataset = _portable_plan_and_dataset(tmp_path)
    with pytest.raises(ValueError, match="Invalid test-processing report"):
        evaluate_frozen_routing(
            plan,
            frozen_model_dir=tmp_path / "models",
            test_processing_report=tmp_path / "missing_processing",
            ground_truth_report=tmp_path / "missing_ground_truth",
            output_dir=tmp_path / "evaluation",
            branch_name="raw.wrist_middle_mcp",
            dataset_root_override=dataset,
        )


def test_all_public_dataset_dependent_fixed_functions_expose_override() -> None:
    for function in (
        build_frozen_routing_models,
        evaluate_frozen_routing,
        evaluate_frozen_end_to_end,
        benchmark_frozen_routing,
        run_frozen_plcc_sensitivity,
    ):
        assert "dataset_root_override" in inspect.signature(function).parameters


def test_public_cli_help_exposes_dataset_override() -> None:
    modules = (
        "tsgr.cli.build_fixed_routing_models",
        "tsgr.cli.evaluate_fixed_routing",
        "tsgr.cli.evaluate_fixed_end_to_end",
        "tsgr.cli.benchmark_fixed_routing",
        "tsgr.cli.run_plcc_sensitivity",
    )
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--dataset-root-override" in completed.stdout
