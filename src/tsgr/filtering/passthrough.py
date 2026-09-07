"""Identity filter used as the raw-data reference."""

from __future__ import annotations
import numpy as np
from tsgr.filtering.base import LandmarkFilter


class PassthroughFilter(LandmarkFilter):
    name = "none"

    def _initialize(self, values: np.ndarray) -> np.ndarray:
        return values.copy()

    def _update(self, values: np.ndarray, dt: float) -> np.ndarray:
        return values.copy()

    def _reset_state(self) -> None:
        return None
