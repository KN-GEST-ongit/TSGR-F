from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsgr.evaluation import frozen_plcc_v42 as plcc
from tsgr.evaluation.frozen_plcc_v42 import resolve_evaluation_worker_count, variant_name


def test_frozen_plcc_variant_name_is_stable():
    assert variant_name(0.995) == "frozen_plcc_0p995"


def test_plcc_evaluation_worker_count_is_bounded(monkeypatch):
    monkeypatch.setattr(plcc.os, "cpu_count", lambda: 32)
    assert resolve_evaluation_worker_count(0, 24) == 8
    assert resolve_evaluation_worker_count(12, 5) == 5
    assert resolve_evaluation_worker_count(1, 24) == 1
    assert resolve_evaluation_worker_count(0, 0) == 0
    with pytest.raises(ValueError):
        resolve_evaluation_worker_count(-1, 2)


def test_plcc_fold_worker_batches_valid_frames(monkeypatch, tmp_path: Path):
    fold_json = tmp_path / "S" / "F" / "fold.json"
    fold_json.parent.mkdir(parents=True)
    fold_json.write_text("{}", encoding="utf-8")

    test_takes_path = fold_json.parent / "test_takes.csv"

    def fake_read_csv(path: Path):
        if Path(path) == test_takes_path:
            return [{"take_id": "take_1"}]
        raise AssertionError(path)

    monkeypatch.setattr(plcc, "read_csv_rows", fake_read_csv)
    monkeypatch.setattr(
        plcc,
        "_load_model",
        lambda *args, **kwargs: (
            {"branch_name": "branch", "gesture_ids": ["A", "C", "I", "Y", "O", "S"]},
            {"image_feature_ids": np.asarray(["img"])},
        ),
    )
    monkeypatch.setattr(
        plcc,
        "_load_run_features",
        lambda *args, **kwargs: (
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([1, 1, 1], dtype=np.int64),
            np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        ),
    )
    monkeypatch.setattr(
        plcc,
        "_run_image_features",
        lambda *args, **kwargs: (
            np.asarray([[0.0], [1.0], [2.0]], dtype=float),
            np.asarray([True, True, False]),
            ("img",),
        ),
    )

    calls: list[int] = []

    def fake_predict(model, arrays, qfull, q2):
        calls.append(len(qfull))
        # First two valid frames predict A then C, both correct for the GT below.
        pred = np.asarray([0 if float(row[0]) == 0.0 else 1 for row in qfull], dtype=np.int64)
        scores = np.zeros((len(pred), 6), dtype=float)
        return pred, pred.copy(), pred.copy(), pred.copy(), scores

    monkeypatch.setattr(plcc, "_predict_frozen", fake_predict)

    plcc._PLCC_MANIFEST = {"take_1": "run"}
    plcc._PLCC_GT_GESTURE_ROWS = {
        "take_1": [(0, "A"), (1, "C"), (2, "I")]
    }
    plcc._PLCC_DATASET_ROOT = tmp_path
    plcc._PLCC_FROZEN_MODEL_DIR = tmp_path / "models"
    plcc._PLCC_BRANCH_NAME = "branch"
    plcc._PLCC_BATCH_SIZE = 1

    row = plcc._evaluate_one_plcc_fold((str(fold_json), "S", "F"))
    assert row["status"] == "ok"
    assert row["gt_gesture_frames"] == 3
    assert row["complete_input_frames"] == 2
    assert row["final_errors_given_input"] == 0
    assert row["final_conditional_accuracy"] == 1.0
    assert calls == [1, 1]


def test_frozen_predict_batch_matches_single_frame_calls():
    from tsgr.evaluation.frozen_routing_v40 import _predict_frozen

    gestures = ["A", "C", "I", "Y", "O", "S"]
    train3 = np.asarray(
        [
            [0.0, 0.0], [0.1, 0.0],
            [1.0, 0.0], [1.1, 0.0],
            [2.0, 0.0], [2.1, 0.0],
            [3.0, 0.0], [3.1, 0.0],
            [4.0, 0.0], [4.1, 0.0],
            [5.0, 0.0], [5.1, 0.0],
        ],
        dtype=float,
    )
    train2 = train3[:, :1].copy()
    labels = np.repeat(np.arange(6, dtype=np.int64), 2)
    model = {
        "gesture_ids": gestures,
        "os_specialist": {"alpha": 0.65, "threshold": 0.0},
    }
    arrays = {
        "feature_indices": np.asarray([0, 1], dtype=np.int64),
        "labels": labels,
        "train3": train3,
        "train2": train2,
        "w3": np.asarray([0.5, 0.5], dtype=float),
        "w2": np.asarray([1.0], dtype=float),
        "iy_indices": np.asarray([0], dtype=np.int64),
        "iy_weights": np.asarray([1.0], dtype=float),
        "c_indices": np.asarray([0], dtype=np.int64),
        "c_a_center": np.asarray([0.05], dtype=float),
        "c_a_scale": np.asarray([0.1], dtype=float),
        "c_c_center": np.asarray([1.05], dtype=float),
        "c_c_scale": np.asarray([0.1], dtype=float),
    }
    q3 = np.asarray([[0.04, 0.0], [1.04, 0.0], [2.04, 0.0], [4.95, 0.0]], dtype=float)
    q2 = q3[:, :1].copy()

    batch = _predict_frozen(model, arrays, q3, q2)
    singles = [_predict_frozen(model, arrays, q3[i:i+1], q2[i:i+1]) for i in range(len(q3))]
    for output_index in range(5):
        expected = np.concatenate([item[output_index] for item in singles], axis=0)
        np.testing.assert_allclose(batch[output_index], expected)
