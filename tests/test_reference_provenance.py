from __future__ import annotations

from tsgr.reference_models.provenance import BranchSignature


def signature(filter_name: str | None) -> BranchSignature:
    return BranchSignature(
        schema_version="v1",
        feature_count=3,
        feature_ids_sha256="ids",
        feature_schema_sha256="schema",
        branch_name="filtered.wrist_middle_mcp" if filter_name else "raw.wrist_middle_mcp",
        landmark_source="filtered" if filter_name else "raw",
        scale_name="wrist_middle_mcp",
        temporal_filter_name=filter_name,
        temporal_filter_parameters_sha256="params" if filter_name else None,
        input_is_mirrored=False,
        mediapipe_model_sha256="model",
    )


def test_branch_signature_reports_filter_mismatch() -> None:
    expected = signature("half_pound")
    actual = BranchSignature.from_dict({**expected.to_dict(), "temporal_filter_name": "one_euro"})
    issues = expected.compatibility_issues(actual)
    assert any("temporal_filter_name" in issue for issue in issues)


def test_raw_signature_has_no_temporal_filter_dependency() -> None:
    raw = signature(None)
    assert raw.temporal_filter_name is None
    assert raw.temporal_filter_parameters_sha256 is None
