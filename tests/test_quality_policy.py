from __future__ import annotations
import numpy as np
from tsgr.config import load_config
from tsgr.quality.frame_quality import evaluate_frame_quality
from tsgr.types import HandCandidate

def hand():
    l=np.zeros((21,3)); l[:,0]=np.linspace(.3,.7,21); l[:,1]=np.linspace(.2,.8,21); return HandCandidate('Right',.95,l,np.zeros((21,3)),0)

def test_low_sharpness_is_warning_not_hard_rejection():
    q=evaluate_frame_quality(np.full((240,320,3),127,np.uint8),hand(),load_config()['quality'])
    assert 'low_sharpness' in q['warning_reasons']; assert q['quality_pass'] is True; assert q['quality_status']=='warning'

def test_roi_sharpness_metrics_are_recorded():
    im=np.indices((240,320)).sum(0).astype(np.uint8); im=np.repeat(im[:,:,None],3,2); q=evaluate_frame_quality(im,hand(),load_config()['quality'])
    assert 'sharpness_laplacian_variance_roi' in q and 'sharpness_tenengrad_roi' in q
