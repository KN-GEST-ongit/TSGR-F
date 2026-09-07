"""Common interfaces and numerical helpers for temporal landmark filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LandmarkFilter(ABC):
    """Stateful filter accepting arbitrary finite NumPy arrays at increasing timestamps."""

    name: str

    def __init__(self, *, maximum_gap_s: float = 0.25) -> None:
        self.maximum_gap_s = float(maximum_gap_s)
        self._last_timestamp_s: float | None = None
        self._shape: tuple[int, ...] | None = None

    def reset(self) -> None:
        self._last_timestamp_s = None
        self._shape = None
        self._reset_state()

    def update(self, values: np.ndarray, timestamp_s: float) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Temporal filter input contains NaN or infinite values.")
        if self._shape is not None and array.shape != self._shape:
            raise ValueError(f"Filter shape changed from {self._shape} to {array.shape}.")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite.")
        if self._last_timestamp_s is None:
            self._shape = array.shape
            self._last_timestamp_s = timestamp
            return self._initialize(array)
        dt = timestamp - self._last_timestamp_s
        if dt <= 0.0 or dt > self.maximum_gap_s:
            self.reset()
            self._shape = array.shape
            self._last_timestamp_s = timestamp
            return self._initialize(array)
        self._last_timestamp_s = timestamp
        return self._update(array, dt)

    @abstractmethod
    def _initialize(self, values: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _update(self, values: np.ndarray, dt: float) -> np.ndarray: ...

    @abstractmethod
    def _reset_state(self) -> None: ...

    def diagnostics(self) -> dict[str, Any]:
        return {"name": self.name, "last_timestamp_s": self._last_timestamp_s}


def low_pass_alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray:
    cutoff = np.asarray(cutoff_hz, dtype=np.float64)
    cutoff = np.maximum(cutoff, 1e-9)
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / float(dt))
