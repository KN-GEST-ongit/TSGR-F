from pathlib import Path

import pytest

from tsgr.evaluation.frozen_routing_v40 import evaluate_frozen_routing


def _minimal_plan(tmp_path: Path) -> Path:
    dataset = tmp_path / "tsgr_dataset"
    (dataset / "training").mkdir(parents=True)
    (dataset / "testing").mkdir()
    (dataset / "annotations.csv").write_text("take_id\n", encoding="utf-8")
    plan = dataset / "reports" / "experiments" / "experiment_plan_test"
    plan.mkdir(parents=True)
    (plan / "experiment_plan.json").write_text('{"dataset_root":"../../../.."}', encoding="utf-8")
    return plan


def test_evaluator_rejects_placeholder_processing_report_before_creating_output(tmp_path: Path) -> None:
    plan = _minimal_plan(tmp_path)
    out = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="Invalid test-processing report"):
        evaluate_frozen_routing(
            plan,
            frozen_model_dir=tmp_path / "models",
            test_processing_report=tmp_path / "<IMAGE_REPORT>",
            ground_truth_report=tmp_path / "ground_truth",
            output_dir=out,
            branch_name="raw.wrist_middle_mcp",
        )
    assert not out.exists()


def test_evaluator_rejects_missing_ground_truth_before_creating_output(tmp_path: Path) -> None:
    plan = _minimal_plan(tmp_path)
    proc = tmp_path / "processing"
    proc.mkdir()
    (proc / "test_take_mediapipe_status.csv").write_text("take_id,run_path\nT1,runs/x\n", encoding="utf-8")
    out = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="Invalid ground-truth report"):
        evaluate_frozen_routing(
            plan,
            frozen_model_dir=tmp_path / "models",
            test_processing_report=proc,
            ground_truth_report=tmp_path / "<GROUND_TRUTH_REPORT>",
            output_dir=out,
            branch_name="raw.wrist_middle_mcp",
        )
    assert not out.exists()
