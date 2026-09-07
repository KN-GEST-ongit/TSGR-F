"""Strict provenance signatures for reference models and processed feature runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tsgr.utils.serialization import sha256_file


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_ids_hash(feature_ids: list[str] | tuple[str, ...]) -> str:
    return _canonical_hash(list(feature_ids))


def feature_schema_hash(schema_payload: dict[str, Any]) -> str:
    """Return a formatting-independent SHA-256 for the complete schema payload."""
    return _canonical_hash(schema_payload)


def feature_mask_hash(feature_ids: list[str] | tuple[str, ...], mask: list[bool]) -> str:
    return _canonical_hash(
        {
            "feature_ids": list(feature_ids),
            "active_mask": [bool(value) for value in mask],
        }
    )


def _load_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    return value if isinstance(value, dict) else {}


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class BranchSignature:
    """Compatibility-critical properties of one model or feature branch."""

    schema_version: str
    feature_count: int
    feature_ids_sha256: str
    feature_schema_sha256: str
    branch_name: str
    landmark_source: str
    scale_name: str
    temporal_filter_name: str | None
    temporal_filter_parameters_sha256: str | None
    input_is_mirrored: bool
    mediapipe_model_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_count": self.feature_count,
            "feature_ids_sha256": self.feature_ids_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "branch_name": self.branch_name,
            "landmark_source": self.landmark_source,
            "scale_name": self.scale_name,
            "temporal_filter_name": self.temporal_filter_name,
            "temporal_filter_parameters_sha256": self.temporal_filter_parameters_sha256,
            "input_is_mirrored": self.input_is_mirrored,
            "mediapipe_model_sha256": self.mediapipe_model_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BranchSignature":
        return cls(
            schema_version=str(payload["schema_version"]),
            feature_count=int(payload["feature_count"]),
            feature_ids_sha256=str(payload["feature_ids_sha256"]),
            feature_schema_sha256=str(payload["feature_schema_sha256"]),
            branch_name=str(payload["branch_name"]),
            landmark_source=str(payload["landmark_source"]),
            scale_name=str(payload["scale_name"]),
            temporal_filter_name=(
                None
                if payload.get("temporal_filter_name") in (None, "")
                else str(payload["temporal_filter_name"])
            ),
            temporal_filter_parameters_sha256=(
                None
                if payload.get("temporal_filter_parameters_sha256") in (None, "")
                else str(payload["temporal_filter_parameters_sha256"])
            ),
            input_is_mirrored=bool(payload["input_is_mirrored"]),
            mediapipe_model_sha256=(
                None
                if payload.get("mediapipe_model_sha256") in (None, "")
                else str(payload["mediapipe_model_sha256"])
            ),
        )

    def compatibility_issues(self, other: "BranchSignature") -> list[str]:
        issues: list[str] = []
        for field_name in (
            "schema_version",
            "feature_count",
            "feature_ids_sha256",
            "feature_schema_sha256",
            "branch_name",
            "landmark_source",
            "scale_name",
            "temporal_filter_name",
            "temporal_filter_parameters_sha256",
            "input_is_mirrored",
            "mediapipe_model_sha256",
        ):
            expected = getattr(self, field_name)
            actual = getattr(other, field_name)
            if expected != actual:
                issues.append(f"{field_name}: expected={expected!r}, actual={actual!r}")
        return issues


def signature_from_run(
    run_dir: str | Path,
    *,
    branch_name: str,
    schema_version: str,
    feature_ids: list[str],
    schema_payload: dict[str, Any],
) -> BranchSignature:
    """Build a branch signature from saved run configuration and environment metadata."""
    run_path = Path(run_dir)
    config = _load_yaml_if_exists(run_path / "config_snapshot.yaml")
    environment = _load_json_if_exists(run_path / "environment.json")
    landmark_source, scale_token = branch_name.split(".", maxsplit=1)
    scale_name = (
        "middle_finger" if scale_token in {"middle_finger", "middle_finger_polyline"} else scale_token
    )
    active_filter = str(config.get("temporal_filter", {}).get("active", "none"))
    if landmark_source == "raw":
        filter_name: str | None = None
        filter_hash: str | None = None
    else:
        filter_name = active_filter
        parameters = (
            config.get("temporal_filter", {})
            .get("filters", {})
            .get(active_filter, {})
        )
        filter_hash = _canonical_hash(parameters)
    model_hash = (
        environment.get("model", {}).get("sha256")
        if isinstance(environment.get("model"), dict)
        else None
    )
    if model_hash is None:
        model_path = Path(config.get("paths", {}).get("model_path", ""))
        if model_path.is_file():
            model_hash = sha256_file(model_path)
    return BranchSignature(
        schema_version=schema_version,
        feature_count=len(feature_ids),
        feature_ids_sha256=feature_ids_hash(feature_ids),
        feature_schema_sha256=feature_schema_hash(schema_payload),
        branch_name=branch_name,
        landmark_source=landmark_source,
        scale_name=scale_name,
        temporal_filter_name=filter_name,
        temporal_filter_parameters_sha256=filter_hash,
        input_is_mirrored=bool(config.get("camera", {}).get("input_is_mirrored", False)),
        mediapipe_model_sha256=(str(model_hash) if model_hash else None),
    )
