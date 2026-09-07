"""Build the detector-derived hand-presence reference used by the evaluation protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.cli.common import resolved_model_path
from tsgr.config import load_config
from tsgr.evaluation.hand_presence_reference import build_hand_presence_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration. The reference builder uses IMAGE/high_recall with handedness ignored.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to hand_landmarker.task. Defaults to paths.model_path in the selected configuration.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0, help="Persistent MediaPipe worker processes; 0 selects automatically.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model_path = resolved_model_path(config, args.model)
    report = build_hand_presence_reference(
        args.dataset_root,
        config=config,
        model_path=model_path,
        output_dir=args.output_dir,
        workers=args.workers,
        progress=args.progress,
        update_annotations=False,
    )
    print(f"Hand-presence reference saved to: {report}")


if __name__ == "__main__":
    main()
