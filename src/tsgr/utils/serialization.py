"""JSON serialization helpers for reproducible result files."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=json_default)


def append_jsonl(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=json_default))
        handle.write("\n")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def environment_report(model_path: str | Path | None = None) -> dict[str, Any]:
    packages = {}
    for package_name in (
        "numpy",
        "opencv-contrib-python",
        "mediapipe",
        "PyYAML",
    ):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None

    report: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }
    if model_path is not None:
        path = Path(model_path)
        report["model"] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
        }
        if path.is_file():
            stat = path.stat()
            report["model"].update(
                {
                    "size_bytes": stat.st_size,
                    "modified_time_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                }
            )
    return report
