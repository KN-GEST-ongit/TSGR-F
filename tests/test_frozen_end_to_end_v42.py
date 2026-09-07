from __future__ import annotations

import numpy as np

from tsgr.evaluation.frozen_bootstrap_v42 import _metric_bundle
from tsgr.evaluation.frozen_end_to_end_v42 import (
    _acceptance_thresholds_for_predictions,
    select_acceptance_threshold,
)


def test_select_acceptance_threshold_perfect_separation():
    scores = np.asarray([0.10, 0.15, 0.20, 0.80, 0.90, 1.00], dtype=float)
    positive = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    result = select_acceptance_threshold(scores, positive, objective="mcc", min_positive_recall=1.0)
    assert 0.20 <= result["threshold"] < 0.80
    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["mcc"] == 1.0


def test_acceptance_threshold_lookup_is_route_specific():
    acceptance = {
        "thresholds": {
            "GLOBAL:A": {"threshold": 1.0},
            "OS:O": {"threshold": 2.0},
            "IY:Y": {"threshold": 3.0},
            "C_RESCUE:C": {"threshold": 4.0},
        }
    }
    gestures = ("A", "C", "O", "Y")
    route = np.asarray(["GLOBAL", "OS", "IY", "C_RESCUE"])
    final = np.asarray([0, 2, 3, 1], dtype=np.int64)
    got = _acceptance_thresholds_for_predictions(acceptance, gestures, route, final)
    assert got.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_metric_bundle_distinguishes_no_gesture_and_no_hand():
    rows = [
        {"gt_state": "GESTURE_A", "predicted_state": "GESTURE_A"},
        {"gt_state": "GESTURE_C", "predicted_state": "NO_GESTURE"},
        {"gt_state": "NO_GESTURE", "predicted_state": "NO_GESTURE"},
        {"gt_state": "NO_HAND", "predicted_state": "MISSING_HAND"},
    ]
    metrics = _metric_bundle(rows, ["A", "C"])
    assert metrics["exact_state_accuracy"] == 0.75
    assert metrics["gesture_frame_end_to_end_accuracy"] == 0.5
    assert metrics["no_gesture_correct_rejection_rate"] == 1.0
    assert metrics["no_hand_recall"] == 1.0


def test_streaming_cluster_bootstrap_counts_match_reference(tmp_path):
    import csv
    from tsgr.evaluation.frozen_bootstrap_v42 import _metrics_from_count_vector, _stream_cluster_vectors

    rows = [
        {"scenario": "S1", "fold_id": "all", "take_id": "t1", "gt_state": "GESTURE_A", "predicted_state": "GESTURE_A"},
        {"scenario": "S1", "fold_id": "all", "take_id": "t1", "gt_state": "NO_GESTURE", "predicted_state": "NO_GESTURE"},
        {"scenario": "S1", "fold_id": "all", "take_id": "t2", "gt_state": "GESTURE_C", "predicted_state": "NO_GESTURE"},
        {"scenario": "S1", "fold_id": "all", "take_id": "t2", "gt_state": "NO_HAND", "predicted_state": "MISSING_HAND"},
    ]
    path = tmp_path / "frame_predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    clusters = _stream_cluster_vectors(path)["S1"]
    got = _metrics_from_count_vector(np.sum(np.vstack(list(clusters.values())), axis=0))
    expected = _metric_bundle(rows, list(("A", "B", "C", "E", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")))
    for key in expected:
        assert np.isclose(got[key], expected[key], equal_nan=True)


def test_predict_frozen_detailed_batch_matches_single_rows():
    from tsgr.evaluation.frozen_end_to_end_v42 import predict_frozen_detailed

    gestures = ("A", "B", "C", "E", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(len(gestures), dtype=np.int64), 3)
    train3 = rng.normal(size=(len(labels), 4))
    train2 = rng.normal(size=(len(labels), 3))
    combined_width = train3.shape[1] + train2.shape[1]
    model = {
        "gesture_ids": list(gestures),
        "os_specialist": {"alpha": 0.65, "threshold": 0.0},
    }
    arrays = {
        "feature_indices": np.arange(4, dtype=np.int64),
        "labels": labels,
        "train3": train3,
        "train2": train2,
        "w3": np.ones(4),
        "w2": np.ones(3),
        "iy_indices": np.asarray([0, 2, 5], dtype=np.int64),
        "iy_weights": np.ones(3),
        "c_indices": np.asarray([1, 4, 6], dtype=np.int64),
        "c_a_center": np.zeros(3),
        "c_a_scale": np.ones(3),
        "c_c_center": np.ones(3) * 0.2,
        "c_c_scale": np.ones(3),
    }
    qfull = rng.normal(size=(5, 4))
    q2 = rng.normal(size=(5, 3))
    batch = predict_frozen_detailed(model, arrays, qfull, q2)
    singles = [predict_frozen_detailed(model, arrays, qfull[i:i+1], q2[i:i+1]) for i in range(5)]
    for key in ("baseline", "after_os", "after_iy", "final", "route", "acceptance_score"):
        expected = np.concatenate([np.asarray(item[key]) for item in singles], axis=0)
        if np.issubdtype(np.asarray(batch[key]).dtype, np.number):
            assert np.allclose(np.asarray(batch[key]), expected)
        else:
            assert np.asarray(batch[key]).tolist() == expected.tolist()
