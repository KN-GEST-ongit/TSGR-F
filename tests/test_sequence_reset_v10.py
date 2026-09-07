from __future__ import annotations

from tsgr.detection.mediapipe_hand_landmarker import MediaPipeHandLandmarker


def test_new_sequence_preserves_monotonic_timestamp_space() -> None:
    detector = MediaPipeHandLandmarker.__new__(MediaPipeHandLandmarker)
    detector._last_timestamp_ms = 2500
    detector._sequence_timestamp_offset_ms = 0
    detector.start_new_sequence(gap_ms=1000)
    assert detector._sequence_timestamp_offset_ms == 3500
