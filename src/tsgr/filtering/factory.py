"""Factory for configured landmark filters."""

from __future__ import annotations
from typing import Any
from tsgr.filtering.base import LandmarkFilter
from tsgr.filtering.ema import EMAFilter
from tsgr.filtering.half_pound import HalfPoundFilter
from tsgr.filtering.kalman import KalmanLandmarkFilter
from tsgr.filtering.one_euro import OneEuroFilter
from tsgr.filtering.passthrough import PassthroughFilter

_NAMES = ("none", "ema", "one_euro", "kalman", "half_pound")

def supported_filter_names() -> tuple[str, ...]: return _NAMES

def create_landmark_filter(name: str, config: dict[str, Any] | None = None) -> LandmarkFilter:
    normalized = str(name).strip().lower().replace("-", "_")
    cfg = config or {}
    common = {"maximum_gap_s": float(cfg.get("maximum_gap_s", 0.25))}
    if normalized == "none": return PassthroughFilter(**common)
    if normalized == "ema": return EMAFilter(alpha=float(cfg.get("alpha", 0.35)), **common)
    if normalized == "one_euro":
        return OneEuroFilter(minimum_cutoff_hz=float(cfg.get("minimum_cutoff_hz", 1.0)),
                             beta=float(cfg.get("beta", 0.02)),
                             derivative_cutoff_hz=float(cfg.get("derivative_cutoff_hz", 1.0)), **common)
    if normalized == "kalman":
        return KalmanLandmarkFilter(process_acceleration_variance=float(cfg.get("process_acceleration_variance", 0.5)),
                                    measurement_variance=float(cfg.get("measurement_variance", 1e-4)),
                                    initial_position_variance=float(cfg.get("initial_position_variance", 1e-3)),
                                    initial_velocity_variance=float(cfg.get("initial_velocity_variance", 1.0)), **common)
    if normalized == "half_pound":
        return HalfPoundFilter(minimum_cutoff_hz=float(cfg.get("minimum_cutoff_hz", 1.0)),
                               maximum_cutoff_hz=float(cfg.get("maximum_cutoff_hz", 5.0)),
                               maximum_absolute_derivative=float(cfg.get("maximum_absolute_derivative", 1.5)), **common)
    raise ValueError(f"Unsupported temporal filter: {name!r}. Supported: {', '.join(_NAMES)}")
