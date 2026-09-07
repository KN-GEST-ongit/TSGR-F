"""Process canonical test takes in explicit IMAGE or VIDEO evaluation mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.cli.common import base_parser, resolved_config, resolved_model_path
from tsgr.processing.test_take_audit import process_experiment_test_takes

DEFAULT_TEST_FALLBACK_FPS = 30.0


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--test-mode",
        choices=("image", "video"),
        required=True,
        help=(
            "Explicit evaluation perspective. IMAGE treats every canonical frame as an independent "
            "stateless sample; VIDEO preserves MediaPipe tracking across the take."
        ),
    )
    parser.add_argument(
        "--handedness-policy",
        choices=("reject_left", "ignore_handedness"),
        required=True,
        help=(
            "reject_left rejects a frame whenever effective Left is observed; ignore_handedness "
            "retains the raw handedness diagnostics but accepts the detected landmarks."
        ),
    )
    fps_group = parser.add_mutually_exclusive_group()
    fps_group.add_argument(
        "--fallback-fps",
        type=float,
        default=DEFAULT_TEST_FALLBACK_FPS,
        help="Fallback FPS used only when source-video FPS metadata is unavailable or invalid. Default: 30.",
    )
    fps_group.add_argument(
        "--fps",
        dest="legacy_fps",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--report-name", type=str, default=None, help="Explicit leaf report directory name inside the selected test variant tree.")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Create a new run even when a compatible completed run exists.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed runs only from the exact same test-mode/profile/handedness policy.",
    )
    parser.add_argument("--all-data", action=argparse.BooleanOptionalAction, default=False, help="Explicitly process the complete dataset; cannot be combined with subject/background/gesture filters.")
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--background", action="append", default=None)
    parser.add_argument("--gesture", action="append", default=None)
    parser.add_argument(
        "--minimum-success-rate",
        "-minimum-success-rate",
        type=float,
        default=0.95,
    )
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--detection-profile",
        choices=("fast", "balanced", "high_recall"),
        default=None,
        help=(
            "Detector profile. Default is high_recall for IMAGE and balanced for VIDEO. "
            "IMAGE high_recall executes the ordered retry chain."
        ),
    )
    parser.add_argument(
        "--video-recovery-after",
        type=int,
        default=1,
        help="Consecutive VIDEO misses before optional stateless IMAGE high-recall recovery.",
    )
    parser.add_argument("--copy-failed-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of persistent MediaPipe worker processes. Use 0 for automatic selection.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress after completed takes.",
    )
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.legacy_fps is not None:
        args.fallback_fps = args.legacy_fps
        print("WARNING: --fps is deprecated for tsgr-process-experiment-testing; use --fallback-fps instead.")
    if args.fallback_fps <= 0:
        parser.error("--fallback-fps must be > 0.")
    if args.all_data and (args.subject or args.background or args.gesture):
        parser.error("--all-data cannot be combined with --subject, --background, or --gesture.")

    if args.video_recovery_after < 1:
        parser.error("--video-recovery-after must be at least 1.")
    if args.mediapipe_mode is not None and args.mediapipe_mode != args.test_mode:
        parser.error("--mediapipe-mode conflicts with --test-mode. Use only --test-mode for this command.")
    if args.mediapipe_profile is not None and args.detection_profile is not None and args.mediapipe_profile != args.detection_profile:
        parser.error("--mediapipe-profile conflicts with --detection-profile.")
    if args.test_mode == "image" and args.video_recovery is True:
        parser.error("--video-recovery is valid only with --test-mode video. IMAGE already uses its configured retry chain.")

    profile = args.detection_profile or args.mediapipe_profile or (
        "high_recall" if args.test_mode == "image" else "balanced"
    )
    video_recovery_enabled = bool(args.video_recovery) if args.test_mode == "video" else False
    mediapipe_overrides: dict[str, object] = {
        "running_mode": args.test_mode,
        "handedness_policy": args.handedness_policy,
    }
    if args.test_mode == "image":
        mediapipe_overrides["image_detection"] = {"active_profile": profile}
        mediapipe_overrides["video_recovery"] = {
            "enabled": False,
            "after_consecutive_failures": int(args.video_recovery_after),
            "image_profile": "high_recall",
        }
    else:
        mediapipe_overrides["video_detection"] = {"active_profile": profile}
        mediapipe_overrides["video_recovery"] = {
            "enabled": video_recovery_enabled,
            "after_consecutive_failures": int(args.video_recovery_after),
            "image_profile": "high_recall",
        }

    config = resolved_config(
        args,
        overrides={
            "mediapipe": mediapipe_overrides,
            # IMAGE evaluation must be truly frame-independent. VIDEO keeps all
            # sequence-local behaviour but the audit itself still exports raw features.
            "roi": {"inference_enabled": False} if args.test_mode == "image" else {},
        },
    )
    model_path = resolved_model_path(config, args.model)
    report = process_experiment_test_takes(
        args.dataset_root,
        config=config,
        model_path=model_path,
        fps=args.fallback_fps,
        force=args.force,
        resume=args.resume,
        subjects=set(args.subject or []),
        backgrounds={value.upper() for value in args.background or []},
        gestures={value.upper() for value in args.gesture or []},
        minimum_success_rate=args.minimum_success_rate,
        save_overlays=args.save_overlays,
        copy_failed_frames=args.copy_failed_frames,
        workers=args.workers,
        progress=args.progress,
        fail_fast=args.fail_fast,
        report_name=args.report_name,
    )
    print(f"Test processing report: {report.resolve()}")


if __name__ == "__main__":
    main()
