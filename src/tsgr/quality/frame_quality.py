"""Image and landmark quality diagnostics with separate warnings and hard rejection."""

from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from tsgr.types import HandCandidate


def _sharpness_metrics(gray: np.ndarray) -> tuple[float, float]:
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    return laplacian_variance, tenengrad


def _hand_roi(gray: np.ndarray, landmarks: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    h,w=gray.shape[:2]
    xs=landmarks[:,0]*w; ys=landmarks[:,1]*h
    x1,x2=float(np.min(xs)),float(np.max(xs)); y1,y2=float(np.min(ys)),float(np.max(ys))
    margin=max(x2-x1,y2-y1)*max(0.0,margin_ratio)
    ix1=max(0,int(np.floor(x1-margin))); iy1=max(0,int(np.floor(y1-margin)))
    ix2=min(w,int(np.ceil(x2+margin))); iy2=min(h,int(np.ceil(y2+margin)))
    if ix2<=ix1 or iy2<=iy1: return gray,(0,0,w,h)
    return gray[iy1:iy2,ix1:ix2],(ix1,iy1,ix2,iy2)


def evaluate_frame_quality(image_bgr: np.ndarray, selected_hand: HandCandidate | None,
                           config: dict[str, Any]) -> dict[str, float | bool | str | list[str] | list[int]]:
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    mean_brightness=float(np.mean(gray)); under=float(np.mean(gray<=10)); over=float(np.mean(gray>=245))
    full_lap,full_ten=_sharpness_metrics(gray)
    warnings=[]; hard=[]
    metrics: dict[str, Any]={"mean_brightness":mean_brightness,"underexposed_fraction":under,
        "overexposed_fraction":over,"sharpness_laplacian_variance":full_lap,
        "sharpness_laplacian_variance_full":full_lap,"sharpness_tenengrad_full":full_ten}
    if selected_hand is None:
        metrics.update({"hand_present":False,"warning_reasons":warnings,"hard_rejection_reasons":["no_hand"],
                        "rejection_reasons":["no_hand"],"quality_status":"reject","quality_pass":False,"quality_score":0.0})
        return metrics
    landmarks=selected_hand.image_landmarks
    roi_gray,roi_xyxy=_hand_roi(gray,landmarks,float(config.get("sharpness_roi_margin_ratio",0.15)))
    roi_lap,roi_ten=_sharpness_metrics(roi_gray)
    metrics.update({"sharpness_laplacian_variance_roi":roi_lap,"sharpness_tenengrad_roi":roi_ten,
                    "quality_roi_xyxy":list(roi_xyxy)})
    sharp_metric=str(config.get("sharpness_metric","roi_laplacian_variance"))
    selected_sharpness = roi_ten if sharp_metric=="roi_tenengrad" else roi_lap
    threshold=float(config.get("minimum_sharpness",25.0) if sharp_metric!="roi_tenengrad" else config.get("minimum_tenengrad",100.0))
    if selected_sharpness<threshold: warnings.append("low_sharpness")
    if mean_brightness<float(config["minimum_mean_brightness"]): hard.append("too_dark")
    if mean_brightness>float(config["maximum_mean_brightness"]): hard.append("too_bright")
    if under>float(config["maximum_underexposed_fraction"]): hard.append("large_underexposed_area")
    if over>float(config["maximum_overexposed_fraction"]): hard.append("large_overexposed_area")
    finite=bool(np.all(np.isfinite(landmarks)))
    x_min,y_min=np.min(landmarks[:,:2],axis=0); x_max,y_max=np.max(landmarks[:,:2],axis=0)
    area=max(0.0,float(x_max-x_min))*max(0.0,float(y_max-y_min))
    border=min(float(x_min),float(y_min),float(1-x_max),float(1-y_max))
    if not finite: hard.append("non_finite_landmarks")
    if area<float(config["minimum_hand_bbox_area_ratio"]): hard.append("hand_too_small")
    if area>float(config["maximum_hand_bbox_area_ratio"]): hard.append("hand_too_large")
    if border<float(config["minimum_border_margin_ratio"]): warnings.append("hand_near_or_outside_border")
    sharp_score=min(1.0,selected_sharpness/max(threshold,1e-9))
    components=[sharp_score,max(0,1-under),max(0,1-over),min(1,area/max(float(config["minimum_hand_bbox_area_ratio"]),1e-9)),
                float(selected_hand.handedness_score),1.0 if finite else 0.0]
    status="reject" if hard else ("warning" if warnings else "pass")
    all_reasons=hard+warnings
    metrics.update({"hand_present":True,"finite_geometry":finite,"hand_bbox_area_ratio":area,
                    "hand_border_margin_ratio":border,"handedness_score":float(selected_hand.handedness_score),
                    "selected_sharpness_metric":sharp_metric,"selected_sharpness_value":selected_sharpness,
                    "sharpness_warning_threshold":threshold,"warning_reasons":warnings,"hard_rejection_reasons":hard,
                    "rejection_reasons":all_reasons,"quality_status":status,"quality_pass":not hard,
                    "quality_warning":bool(warnings),"quality_score":float(np.mean(components))})
    return metrics
