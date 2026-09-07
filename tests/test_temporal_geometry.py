from __future__ import annotations
import numpy as np
from tsgr.preprocessing.spatial_normalization import normalize_world_landmarks
from tsgr.quality.temporal_geometry import TemporalGeometryMonitor
from test_spatial_normalization import synthetic_right_hand

def test_stable_sequence_is_not_flagged():
    m=TemporalGeometryMonitor({"window_size":10,"minimum_history":3,"maximum_scale_relative_deviation":.3,"maximum_ratio_relative_deviation":.3,"maximum_bone_relative_step":.3,"maximum_angular_speed_deg_s":900})
    p=synthetic_right_hand(); last=None
    for i in range(8): last=m.update(p,normalize_world_landmarks(p),i/30)
    assert last is not None and not last['temporal_geometry_outlier']

def test_large_scale_jump_is_flagged_after_history():
    m=TemporalGeometryMonitor({"window_size":10,"minimum_history":3,"maximum_scale_relative_deviation":.2,"maximum_ratio_relative_deviation":.3,"maximum_bone_relative_step":2,"maximum_angular_speed_deg_s":99999})
    p=synthetic_right_hand()
    for i in range(5): m.update(p,normalize_world_landmarks(p),i/30)
    q=p.copy(); q[9:13]*=.25
    result=m.update(q,normalize_world_landmarks(q),5/30)
    assert result['temporal_geometry_outlier']; assert 'palm_scale_outlier' in result['warning_reasons']
