"""Resolve run directories and authoritative result files without placeholder paths."""

from __future__ import annotations
from pathlib import Path


def latest_run_directory(path: str | Path) -> Path:
    base = Path(path)
    if base.is_dir() and base.name.startswith("run_") and (base / "frame_results.jsonl").is_file():
        return base
    runs = base / "runs" if (base / "runs").is_dir() else base
    candidates = [p for p in runs.glob("run_*") if p.is_dir() and (p / "frame_results.jsonl").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No run_* directory with frame_results.jsonl was found under {base}.")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def resolve_results_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file(): return candidate
    direct = candidate / "frame_results.jsonl"
    if direct.is_file(): return direct
    live = candidate / "processing" / "live" / "frame_results.jsonl"
    if live.is_file(): return live
    return latest_run_directory(candidate) / "frame_results.jsonl"
