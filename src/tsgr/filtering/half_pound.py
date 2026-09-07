"""Half Pound Filter based on Lasagno (2026), applied independently per coordinate."""

from __future__ import annotations
import numpy as np
from tsgr.filtering.base import LandmarkFilter, low_pass_alpha


class HalfPoundFilter(LandmarkFilter):
    name = "half_pound"

    def __init__(self, *, minimum_cutoff_hz: float = 1.0, maximum_cutoff_hz: float = 5.0,
                 maximum_absolute_derivative: float = 1.5,
                 maximum_gap_s: float = 0.25) -> None:
        super().__init__(maximum_gap_s=maximum_gap_s)
        if minimum_cutoff_hz <= 0 or maximum_cutoff_hz < minimum_cutoff_hz:
            raise ValueError("Half Pound cutoffs must satisfy 0 < min <= max.")
        if maximum_absolute_derivative <= 0:
            raise ValueError("maximum_absolute_derivative must be positive.")
        self.minimum_cutoff_hz = float(minimum_cutoff_hz)
        self.maximum_cutoff_hz = float(maximum_cutoff_hz)
        self.maximum_absolute_derivative = float(maximum_absolute_derivative)
        self._filtered: np.ndarray | None = None
        self._last_cutoff: np.ndarray | None = None
        self._last_normalized_derivative: np.ndarray | None = None

    def _initialize(self, values: np.ndarray) -> np.ndarray:
        self._filtered = values.copy()
        self._last_cutoff = np.full_like(values, self.minimum_cutoff_hz)
        self._last_normalized_derivative = np.zeros_like(values)
        return values.copy()

    def _update(self, values: np.ndarray, dt: float) -> np.ndarray:
        assert self._filtered is not None
        derivative = (values - self._filtered) / dt
        normalized = np.clip(np.abs(derivative) / self.maximum_absolute_derivative, 0.0, 1.0)
        cutoff = ((1.0 - normalized) * self.minimum_cutoff_hz +
                  normalized * self.maximum_cutoff_hz)
        alpha = low_pass_alpha(cutoff, dt)
        self._filtered = (1.0 - alpha) * self._filtered + alpha * values
        self._last_cutoff = cutoff.copy()
        self._last_normalized_derivative = normalized.copy()
        return self._filtered.copy()

    def _reset_state(self) -> None:
        self._filtered = None; self._last_cutoff = None; self._last_normalized_derivative = None

    def diagnostics(self) -> dict[str, float | str | None]:
        data = super().diagnostics()
        data.update({"minimum_cutoff_hz": self.minimum_cutoff_hz,
                     "maximum_cutoff_hz": self.maximum_cutoff_hz,
                     "maximum_absolute_derivative": self.maximum_absolute_derivative,
                     "mean_active_cutoff_hz": None if self._last_cutoff is None else float(np.mean(self._last_cutoff)),
                     "mean_normalized_derivative": None if self._last_normalized_derivative is None else float(np.mean(self._last_normalized_derivative))})
        return data
