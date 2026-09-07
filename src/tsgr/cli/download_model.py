"""Download the official MediaPipe Hand Landmarker model."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from tsgr.config import load_config


DEFAULT_EXPECTED_SIZE = 7_819_105


def download(url: str, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        print(f"Model already exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    def report(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        if total_size > 0:
            percentage = min(100.0, downloaded * 100.0 / total_size)
            print(f"\rDownloading: {percentage:6.2f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, temporary_path, reporthook=report)
        print()
        size = temporary_path.stat().st_size
        if size < 1_000_000:
            raise OSError(f"Downloaded file is unexpectedly small: {size} bytes.")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Saved model: {output_path.resolve()}")
    print(f"Size: {output_path.stat().st_size} bytes")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"SHA-256: {digest}")
    if output_path.stat().st_size != DEFAULT_EXPECTED_SIZE:
        print(
            "Notice: model size differs from the reference float16 bundle. "
            "This can be valid if Google updated the asset.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or Path(config["paths"]["model_path"])
    url = args.url or str(config["mediapipe"]["model_url"])
    download(url, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
