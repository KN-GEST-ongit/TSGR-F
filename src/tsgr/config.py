"""Configuration loading, validation, and command-line override helpers."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any, TextIO

import yaml

from tsgr.detection.policy import image_detection_passes, video_detection_pass


class ConfigurationError(ValueError):
    """Raised when a configuration file or override is invalid."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _parse_yaml(handle: TextIO, source_name: str) -> dict[str, Any]:
    data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"The configuration root must be a mapping: {source_name}"
        )
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return _parse_yaml(handle, str(config_path))


def source_checkout_config_path() -> Path:
    """Return the editable-source configuration path used during development."""
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_default_config() -> dict[str, Any]:
    checkout_path = source_checkout_config_path()
    if checkout_path.is_file():
        return load_yaml(checkout_path)
    resource = resources.files("tsgr.resources").joinpath("default.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        return _parse_yaml(handle, "tsgr.resources/default.yaml")


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_yaml(path) if path is not None else load_default_config()
    if overrides:
        config = _deep_merge(config, overrides)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    try:
        fps = float(config["camera"]["target_fps"])
        num_hands = int(config["mediapipe"]["num_hands"])
        margin = float(config["roi"]["margin_ratio"])
        alpha = float(config["roi"]["smoothing_alpha"])
        full_frame_interval = int(config["roi"]["full_frame_interval"])
        minimum_size_px = int(config["roi"]["minimum_size_px"])
        writer_queue_size = int(config["recording"]["writer_queue_size"])
        processing_queue_size = int(config["processing"]["queue_size"])
        extension = str(config["recording"]["image_extension"]).lower()
        png_compression = int(config["recording"]["png_compression"])
        minimum_axis_norm = float(config["normalization"]["minimum_axis_norm"])
        minimum_scale = float(config["normalization"]["minimum_scale"])
        primary_scale = str(config["normalization"]["primary_scale"])
        comparison_scale = str(config["normalization"]["comparison_scale"])
        preferred_fourcc = str(config["camera"].get("preferred_fourcc", ""))
        buffer_size = int(config["camera"].get("buffer_size", 1))
        active_filter = str(config["temporal_filter"]["active"])
        maximum_gap_s = float(config["temporal_filter"]["maximum_gap_s"])
        classification_source = str(config["classification_input"]["landmark_source"])
        classification_scale = str(config["classification_input"]["scale"])
        feature_schema_version = str(config["features"]["schema_version"])
        reference_branches = list(config["reference_models"]["branches"])
        session_balance_mode = str(config["reference_models"]["session_balance"]["mode"])
        outlier_minimum_frames = int(config["reference_models"]["outlier_detection"]["minimum_accepted_frames_per_session"])
        covariance_shrinkage = float(config["reference_models"]["covariance"]["shrinkage"])
        covariance_floor = float(config["reference_models"]["covariance"]["diagonal_floor"])
        mediapipe_running_mode = str(config["mediapipe"].get("running_mode", "video")).lower()
        video_recovery_after = int(config["mediapipe"].get("video_recovery", {}).get("after_consecutive_failures", 1))
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Missing or invalid configuration value: {error}") from error

    if not 0 < fps <= 30:
        raise ConfigurationError("camera.target_fps must be in the interval (0, 30].")
    if num_hands < 1:
        raise ConfigurationError("mediapipe.num_hands must be at least one.")
    if mediapipe_running_mode not in {"image", "video"}:
        raise ConfigurationError("mediapipe.running_mode must be image or video.")
    if video_recovery_after < 1:
        raise ConfigurationError("mediapipe.video_recovery.after_consecutive_failures must be at least one.")
    try:
        image_detection_passes(config["mediapipe"])
        video_detection_pass(config["mediapipe"])
        recovery_profile = str(config["mediapipe"].get("video_recovery", {}).get("image_profile", "high_recall"))
        image_detection_passes(config["mediapipe"], profile_name=recovery_profile)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Invalid MediaPipe detection profile: {error}") from error
    if margin < 0:
        raise ConfigurationError("roi.margin_ratio cannot be negative.")
    if not 0 < alpha <= 1:
        raise ConfigurationError("roi.smoothing_alpha must be in the interval (0, 1].")
    if full_frame_interval < 1:
        raise ConfigurationError("roi.full_frame_interval must be at least one.")
    if minimum_size_px < 1:
        raise ConfigurationError("roi.minimum_size_px must be at least one.")
    if writer_queue_size < 1 or processing_queue_size < 1:
        raise ConfigurationError("All queue sizes must be at least one.")
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ConfigurationError("recording.image_extension must be PNG or JPEG.")
    if not 0 <= png_compression <= 9:
        raise ConfigurationError("recording.png_compression must be in the interval [0, 9].")
    if minimum_axis_norm <= 0 or minimum_scale <= 0:
        raise ConfigurationError("Normalization thresholds must be positive.")
    if primary_scale != "wrist_middle_mcp":
        raise ConfigurationError("normalization.primary_scale must be wrist_middle_mcp in release 0.6.0.")
    if comparison_scale != "middle_finger_polyline":
        raise ConfigurationError(
            "normalization.comparison_scale must be middle_finger_polyline in release 0.6.0."
        )
    if preferred_fourcc and len(preferred_fourcc) != 4:
        raise ConfigurationError("camera.preferred_fourcc must contain exactly four characters.")
    if buffer_size < 1:
        raise ConfigurationError("camera.buffer_size must be at least one.")
    if active_filter not in {"none", "ema", "one_euro", "kalman", "half_pound"}:
        raise ConfigurationError("temporal_filter.active is not supported.")
    if maximum_gap_s <= 0:
        raise ConfigurationError("temporal_filter.maximum_gap_s must be positive.")
    if classification_source not in {"raw", "filtered"}:
        raise ConfigurationError("classification_input.landmark_source must be raw or filtered.")
    if classification_scale not in {"wrist_middle_mcp", "middle_finger"}:
        raise ConfigurationError("classification_input.scale must be wrist_middle_mcp or middle_finger.")
    if feature_schema_version != "tsgr_full_features_v1":
        raise ConfigurationError("features.schema_version must be tsgr_full_features_v1 in release 0.6.0.")
    valid_branches = {
        "raw.wrist_middle_mcp",
        "raw.middle_finger",
        "filtered.wrist_middle_mcp",
        "filtered.middle_finger",
    }
    if not reference_branches or any(branch not in valid_branches for branch in reference_branches):
        raise ConfigurationError("reference_models.branches contains an unsupported feature branch.")
    if session_balance_mode not in {"minimum_available", "fixed_count", "all_equal_person_weight"}:
        raise ConfigurationError("reference_models.session_balance.mode is not supported.")
    if outlier_minimum_frames < 1:
        raise ConfigurationError("reference model sessions must retain at least one frame.")
    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ConfigurationError("reference_models.covariance.shrinkage must be in [0, 1].")
    if covariance_floor <= 0:
        raise ConfigurationError("reference_models.covariance.diagonal_floor must be positive.")


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
