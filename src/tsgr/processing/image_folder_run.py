"""Reusable offline processing of one image folder into a versioned TSGR-F run."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from tsgr.config import save_config_snapshot
from tsgr.io.image_folder_source import ImageFolderSource
from tsgr.pipeline.frame_pipeline import FramePipeline
from tsgr.processing.image_outputs import save_roi_crop
from tsgr.processing.result_writer import FrameResultWriter
from tsgr.utils.serialization import environment_report, write_json
from tsgr.visualization.overlay import draw_analysis_overlay


def process_image_folder_run(
    image_folder: str | Path,
    *,
    output_root: str | Path,
    config: dict[str, Any],
    model_path: str | Path,
    fps: float,
    preview: bool = False,
    save_overlays: bool = False,
    save_crops: bool = False,
    recursive: bool = False,
    run_name: str | None = None,
    pipeline: FramePipeline | None = None,
) -> Path:
    """Process one folder and return the created run directory."""
    if not 0 < fps <= 30:
        raise ValueError("fps must be in the interval (0, 30].")
    source_folder = Path(image_folder)
    source = ImageFolderSource(
        source_folder,
        fps=fps,
        extensions=list(config["image_folder"]["extensions"]),
        recursive=recursive or bool(config["image_folder"]["recursive"]),
    )
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / name
    overlay_dir = run_dir / "overlays"
    crop_dir = run_dir / "cropped_frames"
    run_dir.mkdir(parents=True, exist_ok=False)
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, run_dir / "config_snapshot.yaml")
    writer = FrameResultWriter(
        run_dir / "frame_results.jsonl",
        save_compact_npz=bool(
            config.get("normalization", {}).get("save_compact_npz", True)
        ),
    )

    status_counts: dict[str, int] = {}
    processed = 0
    saved_crops = 0
    owns_pipeline = pipeline is None
    pipeline_context = FramePipeline(config, model_path) if owns_pipeline else nullcontext(pipeline)
    try:
        with pipeline_context as active_pipeline:
            if active_pipeline is None:
                raise RuntimeError("Frame pipeline was not initialized.")
            active_pipeline.config = config
            active_pipeline.detector.update_runtime_config(config["mediapipe"])
            active_pipeline.reset_sequence()
            for packet in source:
                analysis = active_pipeline.process(packet)
                writer.write(analysis)
                status_counts[analysis.status] = status_counts.get(analysis.status, 0) + 1
                processed += 1
                if save_crops and analysis.roi_xyxy is not None:
                    path = save_roi_crop(
                        packet.image_bgr,
                        analysis.roi_xyxy,
                        crop_dir,
                        packet.frame_index,
                        int(config["recording"].get("png_compression", 3)),
                    )
                    saved_crops += int(path is not None)
                overlay = None
                if preview or save_overlays:
                    overlay = draw_analysis_overlay(packet.image_bgr, analysis)
                if save_overlays and overlay is not None:
                    cv2.imwrite(
                        str(overlay_dir / f"frame_{packet.frame_index:08d}.png"),
                        overlay,
                    )
                if preview and overlay is not None:
                    cv2.imshow("TSGR-F image folder - Q/Esc stops", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break
    finally:
        writer.close()
        if preview:
            cv2.destroyAllWindows()

    write_json(
        run_dir / "run_summary.json",
        {
            "source_folder": str(source_folder.resolve()),
            "synthetic_fps": fps,
            "processed_frames": processed,
            "saved_crops": saved_crops,
            "status_counts": status_counts,
            "feature_schema_version": config.get("features", {}).get("schema_version"),
            "classification_feature_branch": (
                f"{config['classification_input']['landmark_source']}."
                f"{config['classification_input']['scale']}"
            ),
            "input_is_mirrored": bool(config.get("camera", {}).get("input_is_mirrored", False)),
            "invert_handedness_labels": bool(
                config.get("mediapipe", {}).get("invert_handedness_labels", False)
            ),
            "mediapipe_running_mode": str(config.get("mediapipe", {}).get("running_mode", "video")),
            "mediapipe_image_profile": str(config.get("mediapipe", {}).get("image_detection", {}).get("active_profile", "high_recall")),
            "mediapipe_video_profile": str(config.get("mediapipe", {}).get("video_detection", {}).get("active_profile", "balanced")),
            "mediapipe_video_recovery_enabled": bool(config.get("mediapipe", {}).get("video_recovery", {}).get("enabled", False)),
            "handedness_policy": str(config.get("mediapipe", {}).get("handedness_policy", "reject_left")),
            "strict_right_hand_only": str(config.get("mediapipe", {}).get("handedness_policy", "reject_left")) == "reject_left",
        },
    )
    write_json(run_dir / "environment.json", environment_report(Path(model_path)))
    return run_dir
