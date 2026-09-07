"""Process all independent training photos in a canonical experiment database."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.cli.common import base_parser, resolved_config, resolved_model_path
from tsgr.processing.training_photo_audit import process_experiment_training_photos


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--report-name", type=str, default=None, help="Explicit report directory name under reports/training_processing.")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--all-data", action=argparse.BooleanOptionalAction, default=False, help="Explicitly process the complete dataset; cannot be combined with subject/background/gesture filters.")
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--background", action="append", default=None)
    parser.add_argument("--gesture", action="append", default=None)
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--detection-profile",
        choices=("fast", "balanced", "high_recall"),
        default="high_recall",
        help="IMAGE-mode MediaPipe retry profile. high_recall is the research default.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes. 0 selects an automatic value (up to 8); 1 preserves serial processing.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-cell progress from workers.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop scheduling useful work after a worker reports a cell processing error.",
    )
    args = parser.parse_args()
    if args.all_data and (args.subject or args.background or args.gesture):
        parser.error("--all-data cannot be combined with --subject, --background, or --gesture.")
    if not args.dataset_root.is_dir():
        parser.error(f"Dataset root does not exist: {args.dataset_root}")
    if not 0 < args.fps <= 30:
        parser.error("--fps must be in the interval (0, 30].")
    if args.workers < 0:
        parser.error("--workers must be zero (automatic) or a positive integer.")

    subjects = set(args.subject or []) or None
    backgrounds = {value.upper() for value in (args.background or [])} or None
    gestures = {value.upper() for value in (args.gesture or [])} or None
    base_config = resolved_config(args)
    model_path = resolved_model_path(base_config, args.model)

    report_dir = process_experiment_training_photos(
        args.dataset_root,
        config=base_config,
        model_path=model_path,
        fps=args.fps,
        force=args.force,
        subjects=subjects,
        backgrounds=backgrounds,
        gestures=gestures,
        save_overlays=args.save_overlays,
        detection_profile=args.detection_profile,
        workers=args.workers,
        progress=args.progress,
        fail_fast=args.fail_fast,
        report_name=args.report_name,
    )

    import json

    report = json.loads((report_dir / "processing_report.json").read_text(encoding="utf-8"))
    print(f"Processing report: {report_dir.resolve()}")
    print(
        f"Processed: {len(report.get('processed', []))}, "
        f"skipped: {len(report.get('skipped', []))}, "
        f"failed: {len(report.get('failures', []))}"
    )
    print(
        f"Workers: {report.get('resolved_workers', 0)}, "
        f"images: {report.get('processed_image_count', 0)}, "
        f"wall time: {float(report.get('processing_wall_time_s', 0.0)):.2f}s, "
        f"throughput: {float(report.get('aggregate_images_per_second', 0.0)):.2f} images/s"
    )
    if report.get("failures"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
