"""Detection-policy helpers shared by IMAGE and VIDEO MediaPipe adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DetectionPass:
    """One immutable MediaPipe detector pass configuration."""

    name: str
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float
    padding_ratio: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("min_hand_detection_confidence", self.min_hand_detection_confidence),
            ("min_hand_presence_confidence", self.min_hand_presence_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if float(self.padding_ratio) < 0.0:
            raise ValueError("padding_ratio cannot be negative.")


def _profile(config: dict[str, Any], section: str, profile_name: str) -> dict[str, Any]:
    profiles = config.get(section, {}).get("profiles", {})
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown MediaPipe {section} profile: {profile_name!r}.")
    return profile


def image_detection_passes(config: dict[str, Any], profile_name: str | None = None) -> tuple[DetectionPass, ...]:
    """Resolve the configured ordered IMAGE-mode retry chain."""
    section = config.get("image_detection", {})
    selected = str(profile_name or section.get("active_profile", "high_recall"))
    profile = _profile(config, "image_detection", selected)
    raw_passes = profile.get("passes", [])
    if not isinstance(raw_passes, list) or not raw_passes:
        raise ValueError(f"MediaPipe IMAGE profile {selected!r} must contain at least one pass.")
    passes: list[DetectionPass] = []
    default_tracking = float(config.get("min_tracking_confidence", 0.5))
    for index, raw in enumerate(raw_passes):
        if not isinstance(raw, dict):
            raise ValueError(f"IMAGE pass {index} in profile {selected!r} must be a mapping.")
        item = DetectionPass(
            name=str(raw.get("name", f"pass_{index + 1}")),
            min_hand_detection_confidence=float(raw.get("min_hand_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(raw.get("min_hand_presence_confidence", 0.5)),
            min_tracking_confidence=float(raw.get("min_tracking_confidence", default_tracking)),
            padding_ratio=float(raw.get("padding_ratio", 0.0)),
        )
        item.validate()
        passes.append(item)
    return tuple(passes)


def video_detection_pass(config: dict[str, Any], profile_name: str | None = None) -> DetectionPass:
    """Resolve the single stateful VIDEO-mode profile used by the tracker."""
    section = config.get("video_detection", {})
    selected = str(profile_name or section.get("active_profile", "balanced"))
    profile = _profile(config, "video_detection", selected)
    item = DetectionPass(
        name=selected,
        min_hand_detection_confidence=float(profile.get("min_hand_detection_confidence", 0.5)),
        min_hand_presence_confidence=float(profile.get("min_hand_presence_confidence", 0.5)),
        min_tracking_confidence=float(profile.get("min_tracking_confidence", 0.5)),
        padding_ratio=0.0,
    )
    item.validate()
    return item


def video_recovery_passes(config: dict[str, Any]) -> tuple[DetectionPass, ...]:
    """Resolve the optional stateless IMAGE fallback used after VIDEO tracking loss."""
    recovery = config.get("video_recovery", {})
    profile_name = str(recovery.get("image_profile", "high_recall"))
    return image_detection_passes(config, profile_name=profile_name)
