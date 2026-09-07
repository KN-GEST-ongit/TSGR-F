"""Strict right-hand selection policy."""

from __future__ import annotations

from tsgr.types import HandCandidate


def effective_handedness(label: str, invert_labels: bool) -> str:
    """Return handedness after the configured mirror-label correction."""
    normalized = label.strip().lower()
    if normalized not in {"left", "right"}:
        return normalized
    if not invert_labels:
        return normalized
    return "left" if normalized == "right" else "right"


def effective_handedness_labels(
    candidates: list[HandCandidate],
    invert_labels: bool = False,
) -> list[str]:
    """Return effective labels for all detector candidates."""
    return [
        effective_handedness(candidate.handedness_label, invert_labels)
        for candidate in candidates
    ]


def contains_effective_left_hand(
    candidates: list[HandCandidate],
    invert_labels: bool = False,
) -> bool:
    """Return True when any candidate is effectively classified as a left hand."""
    return "left" in effective_handedness_labels(candidates, invert_labels)


def select_right_hand(
    candidates: list[HandCandidate],
    invert_labels: bool = False,
) -> HandCandidate | None:
    """Select the highest-confidence effective right-hand candidate."""
    right_candidates = [
        candidate
        for candidate in candidates
        if effective_handedness(candidate.handedness_label, invert_labels) == "right"
    ]
    if not right_candidates:
        return None
    return max(right_candidates, key=lambda item: item.handedness_score)


def select_hand_ignoring_handedness(
    candidates: list[HandCandidate],
) -> HandCandidate | None:
    """Select the strongest detected hand without using the Left/Right label.

    This policy is intended only for controlled right-hand-only datasets where
    handedness is known from acquisition protocol and the raw MediaPipe label is
    retained for diagnostics rather than used as a rejection gate.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.handedness_score)
