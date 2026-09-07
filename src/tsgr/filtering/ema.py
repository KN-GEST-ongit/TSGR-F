"""Exponential moving-average filter."""

from __future__ import annotations
import numpy as np
from tsgr.filtering.base import LandmarkFilter


class EMAFilter(LandmarkFilter):
    name = "ema"

    def __init__(self, *, alpha: float = 0.35, maximum_gap_s: float = 0.25) -> None:
        super().__init__(maximum_gap_s=maximum_gap_s)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self._filtered: np.ndarray | None = None

    def _initialize(self, values: np.ndarray) -> np.ndarray:
        self._filtered = values.copy()
        return self._filtered.copy()

    def _update(self, values: np.ndarray, dt: float) -> np.ndarray:
        assert self._filtered is not None
        self._filtered = (1.0 - self.alpha) * self._filtered + self.alpha * values
        return self._filtered.copy()

    def _reset_state(self) -> None:
        self._filtered = None

    def diagnostics(self) -> dict[str, float | str | None]:
        data = super().diagnostics(); data["alpha"] = self.alpha; return data
