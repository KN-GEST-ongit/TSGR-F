"""Independent constant-velocity Kalman filters for every landmark coordinate."""

from __future__ import annotations
import numpy as np
from tsgr.filtering.base import LandmarkFilter


class KalmanLandmarkFilter(LandmarkFilter):
    name = "kalman"

    def __init__(self, *, process_acceleration_variance: float = 0.5,
                 measurement_variance: float = 1.0e-4,
                 initial_position_variance: float = 1.0e-3,
                 initial_velocity_variance: float = 1.0,
                 maximum_gap_s: float = 0.25) -> None:
        super().__init__(maximum_gap_s=maximum_gap_s)
        if min(process_acceleration_variance, measurement_variance,
               initial_position_variance, initial_velocity_variance) <= 0:
            raise ValueError("Kalman variances must be positive.")
        self.process_acceleration_variance = float(process_acceleration_variance)
        self.measurement_variance = float(measurement_variance)
        self.initial_position_variance = float(initial_position_variance)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._original_shape: tuple[int, ...] | None = None
        self._last_gain_mean: float | None = None

    def _initialize(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(-1)
        self._original_shape = values.shape
        self._state = np.column_stack((flat, np.zeros_like(flat)))
        self._covariance = np.zeros((flat.size, 2, 2), dtype=np.float64)
        self._covariance[:, 0, 0] = self.initial_position_variance
        self._covariance[:, 1, 1] = self.initial_velocity_variance
        self._last_gain_mean = None
        return values.copy()

    def _update(self, values: np.ndarray, dt: float) -> np.ndarray:
        assert self._state is not None and self._covariance is not None
        flat = values.reshape(-1)
        f = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = self.process_acceleration_variance * np.asarray(
            [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]], dtype=np.float64
        )
        predicted_state = self._state @ f.T
        predicted_covariance = f[None, :, :] @ self._covariance @ f.T[None, :, :] + q[None, :, :]
        innovation = flat - predicted_state[:, 0]
        innovation_variance = predicted_covariance[:, 0, 0] + self.measurement_variance
        gain = predicted_covariance[:, :, 0] / innovation_variance[:, None]
        self._state = predicted_state + gain * innovation[:, None]
        # Joseph-like simplified scalar update; explicit symmetric cleanup controls drift.
        self._covariance = predicted_covariance.copy()
        self._covariance[:, 0, 0] -= gain[:, 0] * predicted_covariance[:, 0, 0]
        self._covariance[:, 0, 1] -= gain[:, 0] * predicted_covariance[:, 0, 1]
        self._covariance[:, 1, 0] -= gain[:, 1] * predicted_covariance[:, 0, 0]
        self._covariance[:, 1, 1] -= gain[:, 1] * predicted_covariance[:, 0, 1]
        self._covariance = 0.5 * (self._covariance + np.swapaxes(self._covariance, 1, 2))
        self._last_gain_mean = float(np.mean(gain[:, 0]))
        return self._state[:, 0].reshape(values.shape).copy()

    def _reset_state(self) -> None:
        self._state = None; self._covariance = None; self._original_shape = None; self._last_gain_mean = None

    def diagnostics(self) -> dict[str, float | str | None]:
        data = super().diagnostics()
        data.update({"process_acceleration_variance": self.process_acceleration_variance,
                     "measurement_variance": self.measurement_variance,
                     "mean_position_gain": self._last_gain_mean})
        return data
