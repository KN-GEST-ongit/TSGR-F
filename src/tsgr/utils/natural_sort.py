"""Natural sorting utilities for image sequences."""

from __future__ import annotations

import re
from pathlib import Path

_SPLIT_PATTERN = re.compile(r"(\d+)")


def natural_key(value: str | Path) -> list[int | str]:
    text = str(value).lower()
    return [int(part) if part.isdigit() else part for part in _SPLIT_PATTERN.split(text)]
