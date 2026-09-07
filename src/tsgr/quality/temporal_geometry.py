"""Sequence-only geometric consistency checks that never replace frame classification."""

from __future__ import annotations
from collections import deque
from typing import Any
import numpy as np
from tsgr.preprocessing.spatial_normalization import SpatialNormalizationResult

_BONES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
          (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20))

class TemporalGeometryMonitor:
    """Detect scale, ratio, bone-length, and angular jumps relative to rolling history."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.window_size = max(3, int(config.get("window_size", 15)))
        self.minimum_history = max(2, int(config.get("minimum_history", 5)))
        self.maximum_scale_relative_deviation = float(config.get("maximum_scale_relative_deviation", 0.35))
        self.maximum_ratio_relative_deviation = float(config.get("maximum_ratio_relative_deviation", 0.35))
        self.maximum_bone_relative_step = float(config.get("maximum_bone_relative_step", 0.30))
        self.maximum_angular_speed_deg_s = float(config.get("maximum_angular_speed_deg_s", 900.0))
        self._palm = deque(maxlen=self.window_size); self._middle = deque(maxlen=self.window_size)
        self._ratios = deque(maxlen=self.window_size); self._bone_lengths: np.ndarray | None = None
        self._axes: np.ndarray | None = None; self._timestamp_s: float | None = None

    def reset(self) -> None:
        self._palm.clear(); self._middle.clear(); self._ratios.clear()
        self._bone_lengths = None; self._axes = None; self._timestamp_s = None

    @staticmethod
    def _relative_deviation(value: float, history: deque[float]) -> float | None:
        if len(history) == 0: return None
        median = float(np.median(np.asarray(history, dtype=np.float64)))
        return abs(value - median) / max(abs(median), 1e-12)

    @staticmethod
    def _rotation_angle_deg(previous_axes: np.ndarray, axes: np.ndarray) -> float:
        delta = previous_axes.T @ axes
        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def update(self, points: np.ndarray, normalization: SpatialNormalizationResult,
               timestamp_s: float) -> dict[str, Any]:
        points = np.asarray(points, dtype=np.float64)
        palm = float(normalization.scale_wrist_middle_mcp)
        middle = float(normalization.scale_middle_finger)
        ratio = float(normalization.scale_ratio_middle_to_palm)
        bone_lengths = np.asarray([np.linalg.norm(points[b]-points[a]) for a,b in _BONES])
        palm_dev = self._relative_deviation(palm, self._palm) if len(self._palm)>=self.minimum_history else None
        middle_dev = self._relative_deviation(middle, self._middle) if len(self._middle)>=self.minimum_history else None
        ratio_dev = self._relative_deviation(ratio, self._ratios) if len(self._ratios)>=self.minimum_history else None
        bone_step = None
        if self._bone_lengths is not None:
            bone_step = float(np.max(np.abs(bone_lengths-self._bone_lengths)/np.maximum(self._bone_lengths,1e-12)))
        angle_deg = None; angular_speed = None
        if self._axes is not None and self._timestamp_s is not None:
            dt = float(timestamp_s)-self._timestamp_s
            if dt>0:
                angle_deg = self._rotation_angle_deg(self._axes, normalization.axes_world)
                angular_speed = angle_deg/dt
        reasons=[]
        if palm_dev is not None and palm_dev>self.maximum_scale_relative_deviation: reasons.append('palm_scale_outlier')
        if middle_dev is not None and middle_dev>self.maximum_scale_relative_deviation: reasons.append('middle_scale_outlier')
        if ratio_dev is not None and ratio_dev>self.maximum_ratio_relative_deviation: reasons.append('scale_ratio_outlier')
        if bone_step is not None and bone_step>self.maximum_bone_relative_step: reasons.append('bone_length_jump')
        if angular_speed is not None and angular_speed>self.maximum_angular_speed_deg_s: reasons.append('angular_speed_outlier')
        self._palm.append(palm); self._middle.append(middle); self._ratios.append(ratio)
        self._bone_lengths=bone_lengths; self._axes=normalization.axes_world.copy(); self._timestamp_s=float(timestamp_s)
        return {"temporal_geometry_outlier": bool(reasons), "warning_reasons": reasons,
                "palm_scale_relative_deviation": palm_dev, "middle_scale_relative_deviation": middle_dev,
                "scale_ratio_relative_deviation": ratio_dev, "maximum_bone_relative_step": bone_step,
                "orientation_step_deg": angle_deg, "angular_speed_deg_s": angular_speed,
                "history_size": len(self._palm)}
