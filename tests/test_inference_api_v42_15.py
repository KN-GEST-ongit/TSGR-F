from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import tsgr.inference as inference


class _FakePipeline:
    config = None
    model_path = None
    analysis = None

    def __init__(self, config, model_path):
        type(self).config = config
        type(self).model_path = model_path
        self.reset_calls = 0
        self.closed = False

    def reset_sequence(self):
        self.reset_calls += 1

    def process(self, packet):
        assert packet.image_bgr.shape == (8, 8, 3)
        return type(self).analysis

    def close(self):
        self.closed = True


def _install_fake_runtime(monkeypatch, *, score: float, status: str = "valid"):
    model = {
        "branch_name": "raw.wrist_middle_mcp",
        "gesture_ids": ["A", "B", "C"],
    }
    arrays = {"image_feature_ids": np.asarray(["img.a", "img.b"])}
    acceptance = {"thresholds": {"GLOBAL:B": {"threshold": 0.5}}}

    monkeypatch.setattr(inference, "_load_model", lambda root, scenario, fold_id: (model, arrays))
    monkeypatch.setattr(
        inference,
        "_load_acceptance",
        lambda root, scenario, fold_id, model_path: acceptance,
    )
    monkeypatch.setattr(inference, "FramePipeline", _FakePipeline)
    monkeypatch.setattr(
        inference,
        "image_features",
        lambda landmarks: ([0.25, 0.75], ["img.a", "img.b"]),
    )
    monkeypatch.setattr(
        inference,
        "predict_frozen_detailed",
        lambda model, arrays, qfull, q2: {
            "final": np.asarray([1], dtype=np.int64),
            "route": np.asarray(["GLOBAL"]),
            "acceptance_score": np.asarray([score], dtype=float),
            "global_order": np.asarray([[1, 0, 2]], dtype=np.int64),
        },
    )

    feature = SimpleNamespace(values=np.asarray([1.0, 2.0, 3.0]))
    hand = SimpleNamespace(image_landmarks=np.zeros((21, 3), dtype=float))
    _FakePipeline.analysis = SimpleNamespace(
        status=status,
        selected_right_hand=hand,
        feature_vectors={"raw.wrist_middle_mcp": feature},
    )


def test_image_recognizer_uses_reference_image_policy_and_accepts(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch, score=0.25)
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"unused")

    with inference.TSGRFImageRecognizer(
        mediapipe_model=model_file,
        routing_model_root=tmp_path / "routing",
        acceptance_root=tmp_path / "acceptance",
        scenario="S1_ALL_IN_DOMAIN",
        fold_id="all",
    ) as recognizer:
        result = recognizer.predict_bgr(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.label == "B"
    assert result.predicted_state == "GESTURE_B"
    assert result.accepted is True
    assert result.candidate_gesture == "B"
    assert result.route == "GLOBAL"
    assert result.top_gestures == ("B", "A", "C")
    assert _FakePipeline.config["mediapipe"]["running_mode"] == "image"
    assert _FakePipeline.config["mediapipe"]["handedness_policy"] == "ignore_handedness"
    assert _FakePipeline.config["mediapipe"]["image_detection"]["active_profile"] == "high_recall"
    assert _FakePipeline.config["roi"]["inference_enabled"] is False
    assert _FakePipeline.config["temporal_filter"]["active"] == "none"


def test_image_recognizer_rejects_unsupported_candidate(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch, score=0.75)
    recognizer = inference.TSGRFImageRecognizer(
        mediapipe_model=tmp_path / "hand_landmarker.task",
        routing_model_root=tmp_path / "routing",
        acceptance_root=tmp_path / "acceptance",
        scenario="S4_LOSO_BACKGROUND",
        fold_id="subject_P01__background_BLACK",
    )
    result = recognizer.predict_bgr(np.zeros((8, 8, 3), dtype=np.uint8))
    recognizer.close()

    assert result.label == "NO_GESTURE"
    assert result.candidate_gesture == "B"
    assert result.accepted is False
    assert result.acceptance_score == 0.75
    assert result.acceptance_threshold == 0.5
    assert result.reason == "rejected_by_train_only_acceptance"


def test_image_recognizer_maps_missing_hand_to_operational_state(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch, score=0.25, status="no_hand")
    _FakePipeline.analysis = SimpleNamespace(
        status="no_hand",
        selected_right_hand=None,
        feature_vectors={},
    )
    recognizer = inference.TSGRFImageRecognizer(
        mediapipe_model=tmp_path / "hand_landmarker.task",
        routing_model_root=tmp_path / "routing",
        acceptance_root=tmp_path / "acceptance",
        scenario="S1_ALL_IN_DOMAIN",
        fold_id="all",
    )
    result = recognizer.predict_bgr(np.zeros((8, 8, 3), dtype=np.uint8))
    recognizer.close()

    assert result.label == "MISSING_HAND"
    assert result.predicted_state == "MISSING_HAND"
    assert result.candidate_gesture is None
    assert result.accepted is False
    assert result.reason == "mediapipe_missing_hand"


def test_image_recognizer_rejects_invalid_bgr_shape(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch, score=0.25)
    recognizer = inference.TSGRFImageRecognizer(
        mediapipe_model=tmp_path / "hand_landmarker.task",
        routing_model_root=tmp_path / "routing",
        acceptance_root=tmp_path / "acceptance",
        scenario="S1_ALL_IN_DOMAIN",
        fold_id="all",
    )
    try:
        recognizer.predict_bgr(np.zeros((8, 8), dtype=np.uint8))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected invalid image shape to fail.")
    finally:
        recognizer.close()


def test_image_recognizer_matches_existing_fixed_core(monkeypatch, tmp_path):
    gestures = ("A", "B", "C", "E", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(len(gestures), dtype=np.int64), 2)
    train3 = rng.normal(size=(len(labels), 4))
    train2 = rng.normal(size=(len(labels), 3))
    model = {
        "branch_name": "raw.wrist_middle_mcp",
        "gesture_ids": list(gestures),
        "os_specialist": {"alpha": 0.65, "threshold": 0.0},
    }
    arrays = {
        "feature_indices": np.arange(4, dtype=np.int64),
        "image_feature_ids": np.asarray(["img.a", "img.b", "img.c"]),
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
    thresholds = {
        f"{route}:{gesture}": {"threshold": 1e9}
        for route in ("GLOBAL", "OS", "IY", "C_RESCUE")
        for gesture in gestures
    }
    acceptance = {"thresholds": thresholds}
    qfull = rng.normal(size=4)
    q2 = rng.normal(size=3)

    monkeypatch.setattr(inference, "_load_model", lambda root, scenario, fold_id: (model, arrays))
    monkeypatch.setattr(
        inference,
        "_load_acceptance",
        lambda root, scenario, fold_id, model_path: acceptance,
    )
    monkeypatch.setattr(inference, "FramePipeline", _FakePipeline)
    monkeypatch.setattr(
        inference,
        "image_features",
        lambda landmarks: (q2.tolist(), ["img.a", "img.b", "img.c"]),
    )
    _FakePipeline.analysis = SimpleNamespace(
        status="valid",
        selected_right_hand=SimpleNamespace(image_landmarks=np.zeros((21, 3))),
        feature_vectors={"raw.wrist_middle_mcp": SimpleNamespace(values=qfull)},
    )

    expected = inference.predict_frozen_detailed(model, arrays, qfull[None, :], q2[None, :])
    recognizer = inference.TSGRFImageRecognizer(
        mediapipe_model=tmp_path / "hand_landmarker.task",
        routing_model_root=tmp_path / "routing",
        acceptance_root=tmp_path / "acceptance",
        scenario="S4_LOSO_BACKGROUND",
        fold_id="subject_P01__background_BLACK",
    )
    result = recognizer.predict_bgr(np.zeros((8, 8, 3), dtype=np.uint8))
    recognizer.close()

    expected_index = int(expected["final"][0])
    assert result.candidate_gesture == gestures[expected_index]
    assert result.route == str(expected["route"][0])
    assert np.isclose(result.acceptance_score, float(expected["acceptance_score"][0]))
    assert result.accepted is True
