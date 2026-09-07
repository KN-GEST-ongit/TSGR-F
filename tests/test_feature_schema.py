from __future__ import annotations

from collections import Counter

from tsgr.features.schema import FEATURE_COUNT, FEATURE_SCHEMA_VERSION, feature_definitions


def test_feature_schema_has_stable_unique_159_positions() -> None:
    definitions = feature_definitions()
    assert FEATURE_SCHEMA_VERSION == "tsgr_full_features_v1"
    assert FEATURE_COUNT == 159
    assert [definition.index for definition in definitions] == list(range(159))
    assert len({definition.feature_id for definition in definitions}) == 159


def test_feature_schema_contains_thumb_and_orientation_groups() -> None:
    definitions = feature_definitions()
    counts = Counter(definition.group for definition in definitions)
    assert counts["normalized_landmarks"] == 63
    assert counts["joint_bending_angles"] == 15
    assert counts["thumb_geometry"] == 16
    assert counts["global_orientation"] == 6
    ids = {definition.feature_id for definition in definitions}
    assert "bend.thumb.cmc" in ids
    assert "thumb_index.grip_angle" in ids
    assert "thumb.opposition_angle" in ids
    assert "thumb.plane_signed_distance.tip" in ids
    assert "thumb.axis_component.z" in ids
