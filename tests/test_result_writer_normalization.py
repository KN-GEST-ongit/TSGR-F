from __future__ import annotations

import csv
import json

import numpy as np

from tsgr.preprocessing.spatial_normalization import normalize_world_landmarks
from tsgr.processing.result_writer import FrameResultWriter
from tsgr.types import FrameAnalysis
from test_spatial_normalization import synthetic_right_hand


def test_writer_creates_jsonl_scale_csv_and_npz(tmp_path) -> None:
    normalization = normalize_world_landmarks(synthetic_right_hand())
    analysis = FrameAnalysis(
        frame_index=7,
        capture_timestamp_ns=123,
        relative_time_s=0.25,
        status="valid",
        normalization=normalization,
    )
    output = tmp_path / "frame_results.jsonl"
    writer = FrameResultWriter(output)
    writer.write(analysis)
    writer.close()

    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["normalization"]["scales"]["wrist_middle_mcp"] > 0

    with (tmp_path / "normalization_scales.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["normalization_valid"] == "1"

    arrays = np.load(tmp_path / "normalization_arrays.npz")
    assert arrays["frame_index"].tolist() == [7]
    assert arrays["valid"].tolist() == [True]
    assert arrays["normalized_wrist_middle_mcp"].shape == (1, 21, 3)
