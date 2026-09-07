"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tsgr.config import load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--temporal-filter",
        choices=("none", "ema", "one_euro", "kalman", "half_pound"),
        default=None,
        help="Temporal landmark filter used by the filtered branch. Defaults to YAML.",
    )
    parser.add_argument(
        "--classification-landmarks",
        choices=("raw", "filtered"),
        default=None,
        help=(
            "Landmark branch prepared for the future classifier. Raw preserves a "
            "strict single-frame input; filtered uses the active temporal filter."
        ),
    )
    parser.add_argument(
        "--classification-scale",
        choices=("wrist_middle_mcp", "middle_finger"),
        default=None,
        help="Scale branch prepared for the future classifier.",
    )
    parser.add_argument(
        "--mediapipe-mode",
        choices=("image", "video"),
        default=None,
        help="MediaPipe execution mode. IMAGE is stateless and can use retries; VIDEO is stateful tracking.",
    )
    parser.add_argument(
        "--mediapipe-profile",
        choices=("fast", "balanced", "high_recall"),
        default=None,
        help="Detection profile for the selected MediaPipe mode.",
    )
    parser.add_argument(
        "--video-recovery",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable/disable IMAGE high-recall fallback after VIDEO tracking loss.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to hand_landmarker.task.",
    )
    return parser


def resolved_config(
    args: argparse.Namespace,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(overrides or {})

    mediapipe_mode = getattr(args, "mediapipe_mode", None)
    mediapipe_profile = getattr(args, "mediapipe_profile", None)
    video_recovery = getattr(args, "video_recovery", None)
    if mediapipe_mode is not None or mediapipe_profile is not None or video_recovery is not None:
        mediapipe = dict(merged.get("mediapipe", {}))
        if mediapipe_mode is not None:
            mediapipe["running_mode"] = mediapipe_mode
        effective_mode = mediapipe_mode or str(mediapipe.get("running_mode", "video"))
        if mediapipe_profile is not None:
            section_name = "image_detection" if effective_mode == "image" else "video_detection"
            section = dict(mediapipe.get(section_name, {}))
            section["active_profile"] = mediapipe_profile
            mediapipe[section_name] = section
        if video_recovery is not None:
            recovery = dict(mediapipe.get("video_recovery", {}))
            recovery["enabled"] = bool(video_recovery)
            mediapipe["video_recovery"] = recovery
        merged["mediapipe"] = mediapipe

    selected_filter = getattr(args, "temporal_filter", None)
    if selected_filter is not None:
        temporal = dict(merged.get("temporal_filter", {}))
        temporal["active"] = selected_filter
        merged["temporal_filter"] = temporal

    landmark_source = getattr(args, "classification_landmarks", None)
    scale = getattr(args, "classification_scale", None)
    if landmark_source is not None or scale is not None:
        classification_input = dict(merged.get("classification_input", {}))
        if landmark_source is not None:
            classification_input["landmark_source"] = landmark_source
        if scale is not None:
            classification_input["scale"] = scale
        merged["classification_input"] = classification_input

    return load_config(args.config, overrides=merged or None)


def resolved_model_path(config: dict[str, Any], explicit: Path | None) -> Path:
    return explicit or Path(config["paths"]["model_path"])
