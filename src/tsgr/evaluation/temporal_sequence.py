"""Temporal sequence metrics for frame-wise static-gesture predictions.

The module compares binary sequences without assuming statistical independence
between adjacent video frames.  MCC/phi measures whole-sequence agreement,
temporal IoU measures overlap of positive regions, and best-lag MCC quantifies
systematic annotation/prediction displacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalSequenceMetrics:
    mcc: float
    temporal_iou: float
    best_lag_mcc: float
    best_lag_frames: int
    onset_error_frames: float
    offset_error_frames: float
    predicted_segment_count: int
    longest_predicted_segment_frames: int


def binary_mcc(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have the same shape")
    tp = int(np.sum(ref & pred))
    tn = int(np.sum(~ref & ~pred))
    fp = int(np.sum(~ref & pred))
    fn = int(np.sum(ref & ~pred))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0.0:
        # Identical constant sequences are perfect agreement; otherwise the
        # correlation is undefined and reported conservatively as zero.
        return 1.0 if np.array_equal(ref, pred) else 0.0
    return float((tp * tn - fp * fn) / denominator)


def temporal_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have the same shape")
    union = int(np.sum(ref | pred))
    if union == 0:
        return 1.0
    return float(np.sum(ref & pred) / union)


def shift_prediction(prediction: np.ndarray, lag_frames: int) -> np.ndarray:
    """Shift prediction with zero fill.

    Positive lag shifts the prediction later in time. Therefore a positive
    best lag means that the unshifted system tends to fire too early relative
    to the reference; a negative lag means it tends to fire too late.
    """
    pred = np.asarray(prediction, dtype=bool)
    lag = int(lag_frames)
    shifted = np.zeros_like(pred, dtype=bool)
    if lag == 0:
        shifted[:] = pred
    elif lag > 0:
        if lag < pred.size:
            shifted[lag:] = pred[:-lag]
    else:
        amount = -lag
        if amount < pred.size:
            shifted[:-amount] = pred[amount:]
    return shifted


def best_lag_mcc(reference: np.ndarray, prediction: np.ndarray, max_abs_lag_frames: int) -> tuple[float, int]:
    max_lag = int(max_abs_lag_frames)
    if max_lag < 0:
        raise ValueError("max_abs_lag_frames must be non-negative")
    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        candidates.append((binary_mcc(reference, shift_prediction(prediction, lag)), lag))
    # Prefer better MCC, then smaller absolute correction, then negative lag to
    # make ties deterministic.
    score, lag = max(candidates, key=lambda item: (item[0], -abs(item[1]), -item[1]))
    return float(score), int(lag)


def _segment_lengths(sequence: np.ndarray) -> list[int]:
    seq = np.asarray(sequence, dtype=bool)
    lengths: list[int] = []
    start: int | None = None
    for index, value in enumerate(seq.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            lengths.append(index - start)
            start = None
    return lengths


def _first_last_positive(sequence: np.ndarray) -> tuple[int | None, int | None]:
    indices = np.flatnonzero(np.asarray(sequence, dtype=bool))
    if indices.size == 0:
        return None, None
    return int(indices[0]), int(indices[-1])


def sequence_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    max_abs_lag_frames: int = 10,
) -> TemporalSequenceMetrics:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if ref.ndim != 1 or pred.ndim != 1 or ref.shape != pred.shape:
        raise ValueError("reference and prediction must be equal-length 1D sequences")
    mcc = binary_mcc(ref, pred)
    iou = temporal_iou(ref, pred)
    best_mcc, best_lag = best_lag_mcc(ref, pred, max_abs_lag_frames)
    ref_on, ref_off = _first_last_positive(ref)
    pred_on, pred_off = _first_last_positive(pred)
    onset = float("nan") if ref_on is None or pred_on is None else float(pred_on - ref_on)
    offset = float("nan") if ref_off is None or pred_off is None else float(pred_off - ref_off)
    segments = _segment_lengths(pred)
    return TemporalSequenceMetrics(
        mcc=mcc,
        temporal_iou=iou,
        best_lag_mcc=best_mcc,
        best_lag_frames=best_lag,
        onset_error_frames=onset,
        offset_error_frames=offset,
        predicted_segment_count=len(segments),
        longest_predicted_segment_frames=max(segments, default=0),
    )
