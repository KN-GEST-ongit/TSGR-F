from __future__ import annotations

import csv
import json

import numpy as np

from tsgr.features import extract_feature_vector
from tsgr.preprocessing.spatial_normalization import normalize_world_landmarks
from tsgr.processing.result_writer import FrameResultWriter
from tsgr.types import FrameAnalysis


def synthetic_right_hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    points[1:5] = [[-0.02, 0.02, 0.01], [-0.04, 0.04, 0.01], [-0.06, 0.06, 0.015], [-0.08, 0.08, 0.02]]
    for base, x in ((5, -0.03), (9, 0.0), (13, 0.025), (17, 0.05)):
        points[base] = [x, 0.05, 0.0]
        points[base + 1] = [x, 0.09, 0.002]
        points[base + 2] = [x, 0.125, 0.004]
        points[base + 3] = [x, 0.155, 0.006]
    return points


def test_writer_exports_schema_selected_csv_all_branches_and_npz(tmp_path) -> None:
    normalization = normalize_world_landmarks(synthetic_right_hand())
    vectors = {
        f"{source}.{scale}": extract_feature_vector(
            normalization,
            landmark_source=source,
            scale_name=scale,
        )
        for source in ("raw", "filtered")
        for scale in ("wrist_middle_mcp", "middle_finger")
    }
    analysis = FrameAnalysis(
        frame_index=7,
        capture_timestamp_ns=1,
        relative_time_s=0.2,
        status="valid",
        normalization=normalization,
        raw_normalization=normalization,
        filtered_normalization=normalization,
        feature_vectors=vectors,
        classification_feature_branch="filtered.wrist_middle_mcp",
    )
    with FrameResultWriter(tmp_path / "frame_results.jsonl") as writer:
        writer.write(analysis)

    schema = json.loads((tmp_path / "features" / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["feature_count"] == 159
    with (tmp_path / "features" / "classification_features.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["classification_feature_branch"] == "filtered.wrist_middle_mcp"
    assert rows[0]["finite_feature_count"] == "159"
    arrays = np.load(tmp_path / "features" / "feature_arrays.npz", allow_pickle=False)
    assert arrays["values"].shape == (1, 4, 159)
