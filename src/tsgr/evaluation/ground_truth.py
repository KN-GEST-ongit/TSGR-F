"""Operational ground-truth rules for annotated TSGR-F test videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NO_GESTURE = "NO_GESTURE"
NO_HAND = "NO_HAND"
UNANNOTATED = "UNANNOTATED"
HAND_PRESENT = "HAND_PRESENT"
HAND_MISSING = "HAND_MISSING"


@dataclass(frozen=True, slots=True)
class GroundTruthDecision:
    """Resolved frame-level state and provenance."""

    evaluation_state: str
    hand_present: bool | None
    source: str


def is_gesture_label(label: str) -> bool:
    normalized = str(label).strip().upper()
    return bool(normalized and normalized not in {NO_GESTURE, NO_HAND, UNANNOTATED})


def evaluation_state(true_label: str, hand_presence_reference: bool | None) -> GroundTruthDecision:
    """Resolve the final frame state.

    An annotated gesture interval is authoritative: the hand is considered present
    without running the hand-presence detector on that frame. Outside the gesture
    interval the dedicated high-recall MediaPipe reference decides whether the state
    is NO_GESTURE (hand visible) or NO_HAND (hand absent).
    """
    label = str(true_label).strip().upper()
    if is_gesture_label(label):
        return GroundTruthDecision(
            evaluation_state=f"GESTURE_{label}",
            hand_present=True,
            source="gesture_annotation_override",
        )
    if label == UNANNOTATED or not label:
        return GroundTruthDecision(
            evaluation_state=UNANNOTATED,
            hand_present=hand_presence_reference,
            source="unannotated_excluded_from_evaluation",
        )
    if label != NO_GESTURE:
        raise ValueError(f"Unsupported imported frame label: {true_label!r}")
    if hand_presence_reference is None:
        return GroundTruthDecision(
            evaluation_state=UNANNOTATED,
            hand_present=None,
            source="missing_hand_presence_reference",
        )
    if hand_presence_reference:
        return GroundTruthDecision(
            evaluation_state=NO_GESTURE,
            hand_present=True,
            source="mediapipe_hand_presence_reference",
        )
    return GroundTruthDecision(
        evaluation_state=NO_HAND,
        hand_present=False,
        source="mediapipe_hand_presence_reference",
    )


def state_gesture_id(state: str) -> str:
    normalized = str(state).strip().upper()
    return normalized.removeprefix("GESTURE_") if normalized.startswith("GESTURE_") else ""


def hand_reference_bool(row: dict[str, Any] | None) -> bool | None:
    if row is None:
        return None
    raw = str(row.get("hand_present_reference", "")).strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return None
