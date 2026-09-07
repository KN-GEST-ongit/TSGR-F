from __future__ import annotations
import numpy as np
import pytest
from tsgr.filtering.factory import create_landmark_filter,supported_filter_names

def sequence(filter_name):
    f=create_landmark_filter(filter_name,{"alpha":0.25,"minimum_cutoff_hz":1.0,"maximum_cutoff_hz":5.0,"maximum_absolute_derivative":2.0,"beta":0.02,"derivative_cutoff_hz":1.0,"process_acceleration_variance":0.5,"measurement_variance":1e-4,"initial_position_variance":1e-3,"initial_velocity_variance":1.0,"maximum_gap_s":.25})
    return np.asarray([f.update(np.full((21,3),0.0 if i<5 else 1.0),i/30) for i in range(12)])

def test_all_filter_names_construct_and_return_finite_arrays():
    for name in supported_filter_names():
        out=sequence(name); assert out.shape==(12,21,3); assert np.isfinite(out).all()

def test_passthrough_is_exact():
    out=sequence('none'); assert np.all(out[:5]==0); assert np.all(out[5:]==1)

def test_smoothing_filters_do_not_overshoot_unit_step():
    for name in ('ema','one_euro','half_pound'):
        out=sequence(name); assert float(out.min())>=-1e-12; assert float(out.max())<=1.000001; assert float(out[5,0,0])<1.0

def test_non_increasing_timestamp_resets_filter():
    f=create_landmark_filter('ema',{'alpha':.2,'maximum_gap_s':.25}); f.update(np.zeros((21,3)),1.0); result=f.update(np.ones((21,3)),.5); np.testing.assert_allclose(result,1.0)

def test_unknown_filter_is_rejected():
    with pytest.raises(ValueError): create_landmark_filter('mystery',{})
