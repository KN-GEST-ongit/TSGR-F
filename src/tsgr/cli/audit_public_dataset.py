"""Audit a TSGR-F public dataset for structural completeness and privacy-safe annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsgr.dataset.contract import validate_public_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--require-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--release-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the immutable public-release layout. Use --no-release-layout when "
            "auditing a reproduction workspace that contains derived runs/reports."
        ),
    )
    args = parser.parse_args()
    audit = validate_public_dataset(
        args.dataset_root,
        require_videos=args.require_videos,
        require_annotations=True,
        release_layout=args.release_layout,
    )
    print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
    if args.strict and not audit.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
