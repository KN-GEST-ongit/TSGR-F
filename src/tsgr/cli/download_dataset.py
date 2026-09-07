"""Download or install the released TSGR-F dataset packages."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsgr.dataset.downloader import DEFAULT_DATASET_VERSION, DEFAULT_REPOSITORY, download_dataset_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "tsgr_dataset")
    parser.add_argument("--version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--subject", action="append", default=None, help="Install one subject package; repeat as needed.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="GitHub owner/repository used for release assets.")
    parser.add_argument("--tag", default=None, help="Optional GitHub Release tag override.")
    parser.add_argument("--archive-dir", type=Path, default=None, help="Install from an existing directory of release ZIP files instead of downloading.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = download_dataset_release(
        args.output_dir,
        version=args.version,
        subjects=args.subject,
        repository=args.repository,
        tag=args.tag,
        archive_dir=args.archive_dir,
        overwrite=args.overwrite,
        validate=args.validate,
    )
    print(f"TSGR-F dataset installed at: {output}")


if __name__ == "__main__":
    main()
