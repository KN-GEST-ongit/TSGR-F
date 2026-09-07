from __future__ import annotations

import numpy as np

from tsgr.reference_models.masking import build_feature_mask
from tsgr.reference_models.outliers import detect_session_outliers


def test_outlier_detector_flags_large_multifeature_failure() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(0.0, 0.02, size=(40, 12))
    values[-1, :8] += 4.0
    result = detect_session_outliers(values)
    assert result.accepted_mask[:-1].sum() >= 38
    assert not result.accepted_mask[-1]
    assert result.robust_rms_score[-1] > 4.0


def test_feature_mask_removes_constant_and_near_constant_columns() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=(50, 4))
    values[:, 0] = 0.0
    values[:, 1] = 1.0 + rng.normal(scale=1.0e-10, size=50)
    result = build_feature_mask(values, ["a", "b", "c", "d"])
    assert result.active_mask.tolist() == [False, False, True, True]
    assert result.reasons[0] == "constant"
    assert result.reasons[1] == "near_constant"


def test_outlier_detector_catches_single_spike_in_otherwise_constant_feature() -> None:
    values = np.zeros((30, 6), dtype=np.float64)
    values[-1, 2] = 2.0
    result = detect_session_outliers(values, maximum_extreme_fraction=0.0)
    assert result.accepted_mask[:-1].all()
    assert not result.accepted_mask[-1]
    assert result.top_feature_indices[-1] == [2]
