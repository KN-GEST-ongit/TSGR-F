"""OpenCV overlay generation for capture and replay previews."""

from __future__ import annotations

import cv2
import numpy as np

from tsgr.constants import HAND_CONNECTIONS
from tsgr.types import FrameAnalysis


def draw_analysis_overlay(image_bgr: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
    output = image_bgr.copy()
    height, width = output.shape[:2]
    if analysis.roi_xyxy is not None:
        x1, y1, x2, y2 = analysis.roi_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2)
    hand = analysis.selected_right_hand
    if hand is not None:
        points = np.column_stack(
            (
                np.clip(hand.image_landmarks[:, 0] * width, 0, width - 1),
                np.clip(hand.image_landmarks[:, 1] * height, 0, height - 1),
            )
        ).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, tuple(points[start]), tuple(points[end]), (70, 220, 70), 2)
        for index, point in enumerate(points):
            cv2.circle(output, tuple(point), 4, (40, 40, 240), -1)
            cv2.putText(
                output,
                str(index),
                (int(point[0] + 4), int(point[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    quality_score = float(analysis.quality_metrics.get("quality_score", 0.0))
    lines = [
        f"frame={analysis.frame_index} status={analysis.status} quality={quality_score:.3f}"
    ]
    diagnostics = analysis.detection_diagnostics or {}
    if diagnostics:
        lines.append(
            f"detector={diagnostics.get('running_mode', '?')}/{diagnostics.get('profile', '?')} "
            f"attempt={diagnostics.get('selected_attempt') or '-'} recovered={int(bool(diagnostics.get('recovered', False)))}"
        )
    if analysis.classification_feature_branch is not None:
        vector = analysis.classification_feature_vector
        finite = vector.finite_count if vector is not None else 0
        lines.append(
            f"feature_branch={analysis.classification_feature_branch} "
            f"finite={finite}/159 filter={analysis.temporal_filter_name}"
        )
    if analysis.normalization is not None:
        norm = analysis.normalization
        lines.append(
            "scale palm={:.5f} m middle={:.5f} m ratio={:.3f}".format(
                norm.scale_wrist_middle_mcp,
                norm.scale_middle_finger,
                norm.scale_ratio_middle_to_palm,
            )
        )
    elif analysis.notes:
        lines.append(analysis.notes[0][:100])
    box_width = min(width, 900)
    box_height = 12 + 24 * len(lines)
    cv2.rectangle(output, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (10, 23 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output
