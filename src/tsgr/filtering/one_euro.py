"""Vectorized One Euro Filter following Casiez, Roussel, and Vogel (2012)."""

from __future__ import annotations
import numpy as np
from tsgr.filtering.base import LandmarkFilter, low_pass_alpha


class OneEuroFilter(LandmarkFilter):
    name = "one_euro"

    def __init__(self, *, minimum_cutoff_hz: float = 1.0, beta: float = 0.02,
                 derivative_cutoff_hz: float = 1.0, maximum_gap_s: float = 0.25) -> None:
        super().__init__(maximum_gap_s=maximum_gap_s)
        if minimum_cutoff_hz <= 0 or derivative_cutoff_hz <= 0 or beta < 0:
            raise ValueError("One Euro parameters must satisfy min_cutoff>0, d_cutoff>0, beta>=0.")
        self.minimum_cutoff_hz = float(minimum_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self._raw_previous: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._filtered_derivative: np.ndarray | None = None
        self._last_cutoff: np.ndarray | None = None

    def _initialize(self, values: np.ndarray) -> np.ndarray:
        self._raw_previous = values.copy()
        self._filtered = values.copy()
        self._filtered_derivative = np.zeros_like(values)
        self._last_cutoff = np.full_like(values, self.minimum_cutoff_hz)
        return values.copy()

    def _update(self, values: np.ndarray, dt: float) -> np.ndarray:
        assert self._raw_previous is not None and self._filtered is not None
        assert self._filtered_derivative is not None
        derivative = (values - self._raw_previous) / dt
        alpha_d = low_pass_alpha(self.derivative_cutoff_hz, dt)
        self._filtered_derivative = (
            (1.0 - alpha_d) * self._filtered_derivative + alpha_d * derivative
        )
        cutoff = self.minimum_cutoff_hz + self.beta * np.abs(self._filtered_derivative)
        alpha = low_pass_alpha(cutoff, dt)
        self._filtered = (1.0 - alpha) * self._filtered + alpha * values
        self._raw_previous = values.copy()
        self._last_cutoff = cutoff.copy()
        return self._filtered.copy()

    def _reset_state(self) -> None:
        self._raw_previous = None; self._filtered = None
        self._filtered_derivative = None; self._last_cutoff = None

    def diagnostics(self) -> dict[str, float | str | None]:
        data = super().diagnostics()
        data.update({"minimum_cutoff_hz": self.minimum_cutoff_hz, "beta": self.beta,
                     "derivative_cutoff_hz": self.derivative_cutoff_hz,
                     "mean_active_cutoff_hz": None if self._last_cutoff is None else float(np.mean(self._last_cutoff))})
        return data
