"""Versioned metadata for the complete TSGR-F single-frame feature vector."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from tsgr.constants import LANDMARK_NAMES

FEATURE_SCHEMA_VERSION = "tsgr_full_features_v1"
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_BASES = (2, 5, 9, 13, 17)
FINGER_TIPS = (4, 8, 12, 16, 20)
PAIR_NAMES = tuple(
    (FINGER_NAMES[i], FINGER_NAMES[j])
    for i in range(len(FINGER_NAMES))
    for j in range(i + 1, len(FINGER_NAMES))
)


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """One stable feature position in the versioned classifier input vector."""

    index: int
    feature_id: str
    group: str
    unit: str
    description: str
    formula: str
    orientation_sensitive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "feature_id": self.feature_id,
            "group": self.group,
            "unit": self.unit,
            "description": self.description,
            "formula": self.formula,
            "orientation_sensitive": self.orientation_sensitive,
        }


@dataclass(slots=True)
class FeatureVector:
    """Feature values and provenance for one landmark/scale processing branch."""

    values: np.ndarray
    landmark_source: str
    scale_name: str
    warnings: list[str]
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        if array.shape != (FEATURE_COUNT,):
            raise ValueError(
                f"Expected {FEATURE_COUNT} features, got array with shape {array.shape}."
            )
        self.values = array

    @property
    def finite_count(self) -> int:
        return int(np.isfinite(self.values).sum())

    @property
    def complete(self) -> bool:
        return self.finite_count == FEATURE_COUNT

    def named_values(self) -> dict[str, float | None]:
        return {
            definition.feature_id: (
                float(value) if np.isfinite(value) else None
            )
            for definition, value in zip(feature_definitions(), self.values)
        }

    def to_dict(self, *, include_named_values: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "landmark_source": self.landmark_source,
            "scale_name": self.scale_name,
            "feature_count": FEATURE_COUNT,
            "finite_count": self.finite_count,
            "complete": self.complete,
            "warnings": list(self.warnings),
            "values": [float(value) if np.isfinite(value) else None for value in self.values],
        }
        if include_named_values:
            payload["named_values"] = self.named_values()
        return payload


def _append(
    items: list[FeatureDefinition],
    feature_id: str,
    group: str,
    unit: str,
    description: str,
    formula: str,
    *,
    orientation_sensitive: bool = False,
) -> None:
    items.append(
        FeatureDefinition(
            index=len(items),
            feature_id=feature_id,
            group=group,
            unit=unit,
            description=description,
            formula=formula,
            orientation_sensitive=orientation_sensitive,
        )
    )


@lru_cache(maxsize=1)
def feature_definitions() -> tuple[FeatureDefinition, ...]:
    """Return the immutable 159-feature schema in classifier vector order."""
    items: list[FeatureDefinition] = []

    for landmark_name in LANDMARK_NAMES:
        for axis in ("x", "y", "z"):
            _append(
                items,
                f"landmark.{landmark_name}.{axis}",
                "normalized_landmarks",
                "normalized_unit",
                f"Canonical {axis.upper()} coordinate of {landmark_name}.",
                f"N[{landmark_name},{axis}]",
            )

    joint_features = (
        ("thumb.cmc", 0, 1, 2),
        ("thumb.mcp", 1, 2, 3),
        ("thumb.ip", 2, 3, 4),
        ("index.mcp", 0, 5, 6),
        ("index.pip", 5, 6, 7),
        ("index.dip", 6, 7, 8),
        ("middle.mcp", 0, 9, 10),
        ("middle.pip", 9, 10, 11),
        ("middle.dip", 10, 11, 12),
        ("ring.mcp", 0, 13, 14),
        ("ring.pip", 13, 14, 15),
        ("ring.dip", 14, 15, 16),
        ("pinky.mcp", 0, 17, 18),
        ("pinky.pip", 17, 18, 19),
        ("pinky.dip", 18, 19, 20),
    )
    for name, a, b, c in joint_features:
        _append(
            items,
            f"bend.{name}",
            "joint_bending_angles",
            "radian",
            f"Internal bending angle at landmark P{b} for {name}.",
            f"angle(P{a}, P{b}, P{c})",
        )

    for first, second in PAIR_NAMES:
        first_index = FINGER_NAMES.index(first)
        second_index = FINGER_NAMES.index(second)
        first_base = FINGER_BASES[first_index]
        second_base = FINGER_BASES[second_index]
        _append(
            items,
            f"spread.mcp.{first}_{second}",
            "mcp_spread_angles",
            "radian",
            f"Angular spread between the {first} and {second} bases around the wrist.",
            f"angle(P{first_base}, P0, P{second_base})",
        )

    finger_axis_points = {
        "thumb": (2, 4),
        "index": (5, 8),
        "middle": (9, 12),
        "ring": (13, 16),
        "pinky": (17, 20),
    }
    for first, second in PAIR_NAMES:
        a0, a1 = finger_axis_points[first]
        b0, b1 = finger_axis_points[second]
        _append(
            items,
            f"spread.axis.{first}_{second}",
            "finger_axis_spread_angles",
            "radian",
            f"Angle between complete {first} and {second} finger axes.",
            f"angle(P{a1}-P{a0}, P{b1}-P{b0})",
        )

    adjacent_pairs = (
        ("thumb", "index"),
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
    )
    for first, second in adjacent_pairs:
        _append(
            items,
            f"spread.segment_mean.{first}_{second}",
            "segment_mean_spread_angles",
            "radian",
            f"Mean angle between corresponding phalanx segments of adjacent {first} and {second} fingers.",
            "mean_k angle(segment_first_k, segment_second_k)",
        )

    for first, second in PAIR_NAMES:
        first_tip = FINGER_TIPS[FINGER_NAMES.index(first)]
        second_tip = FINGER_TIPS[FINGER_NAMES.index(second)]
        _append(
            items,
            f"distance.tip.{first}_{second}",
            "fingertip_pair_distances",
            "normalized_unit",
            f"Normalized Euclidean distance between the {first} and {second} fingertips.",
            f"||N[P{first_tip}]-N[P{second_tip}]||",
        )

    for finger, tip in zip(FINGER_NAMES, FINGER_TIPS):
        _append(
            items,
            f"distance.tip_to_wrist.{finger}",
            "fingertip_wrist_distances",
            "normalized_unit",
            f"Normalized distance from the {finger} fingertip to the wrist.",
            f"||N[P{tip}]-N[P0]||",
        )

    for finger, tip in zip(FINGER_NAMES, FINGER_TIPS):
        _append(
            items,
            f"distance.tip_to_palm_center.{finger}",
            "fingertip_palm_center_distances",
            "normalized_unit",
            f"Normalized distance from the {finger} fingertip to the palm reference center.",
            f"||N[P{tip}]-C_palm||",
        )

    for finger in FINGER_NAMES:
        _append(
            items,
            f"straightness.{finger}",
            "finger_straightness_ratios",
            "ratio",
            f"Chord-to-polyline straightness ratio of the {finger} finger.",
            "||tip-base|| / sum(segment_lengths)",
        )

    for finger in FINGER_NAMES:
        _append(
            items,
            f"out_of_plane.{finger}",
            "finger_out_of_plane_angles",
            "radian",
            f"Signed elevation of the {finger} axis relative to the canonical palm plane.",
            "asin(v_finger,z / ||v_finger||)",
        )

    thumb_features = (
        (
            "thumb_index.grip_angle",
            "radian",
            "Angle between the thumb and index finger axes.",
            "angle(P4-P2, P8-P5)",
        ),
        (
            "thumb_index.distal_joint_distance",
            "normalized_unit",
            "Distance between thumb IP and index DIP.",
            "||N[P3]-N[P7]||",
        ),
        (
            "thumb_index.proximal_joint_distance",
            "normalized_unit",
            "Distance between thumb MCP and index PIP.",
            "||N[P2]-N[P6]||",
        ),
        (
            "thumb.opposition_angle",
            "radian",
            "Angle between the palm-plane and thumb-plane normals.",
            "angle(n_palm, n_thumb)",
        ),
        (
            "thumb.in_plane_angle",
            "radian",
            "Signed angle of the projected thumb axis within the palm plane.",
            "atan2(v_thumb,y, v_thumb,x)",
        ),
    )
    for feature_id, unit, description, formula in thumb_features:
        _append(items, feature_id, "thumb_geometry", unit, description, formula)

    for landmark_name, landmark_index in (
        ("cmc", 1),
        ("mcp", 2),
        ("ip", 3),
        ("tip", 4),
    ):
        _append(
            items,
            f"thumb.plane_signed_distance.{landmark_name}",
            "thumb_geometry",
            "normalized_unit",
            f"Signed canonical distance of thumb {landmark_name.upper()} from the palm plane.",
            f"N[P{landmark_index},z]",
        )

    for finger, base in (("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
        _append(
            items,
            f"thumb.tip_to_mcp_distance.{finger}",
            "thumb_geometry",
            "normalized_unit",
            f"Distance from thumb tip to the {finger} MCP landmark.",
            f"||N[P4]-N[P{base}]||",
        )

    for axis in ("x", "y", "z"):
        _append(
            items,
            f"thumb.axis_component.{axis}",
            "thumb_geometry",
            "unit_component",
            f"{axis.upper()} component of the unit thumb axis in canonical coordinates.",
            f"unit(P4-P2).{axis}",
        )

    palm_features = (
        ("palm.width", "normalized_unit", "Normalized distance P5-P17.", "||N[P5]-N[P17]||"),
        ("palm.mcp_distance.index_middle", "normalized_unit", "Normalized distance P5-P9.", "||N[P5]-N[P9]||"),
        ("palm.mcp_distance.middle_ring", "normalized_unit", "Normalized distance P9-P13.", "||N[P9]-N[P13]||"),
        ("palm.mcp_distance.ring_pinky", "normalized_unit", "Normalized distance P13-P17.", "||N[P13]-N[P17]||"),
        ("palm.aspect_ratio", "ratio", "Palm width divided by wrist-middle MCP length.", "||P5-P17|| / ||P9-P0||"),
    )
    for feature_id, unit, description, formula in palm_features:
        _append(items, feature_id, "palm_geometry", unit, description, formula)

    for axis in ("x", "y", "z"):
        _append(
            items,
            f"orientation.palm_normal_camera.{axis}",
            "global_orientation",
            "unit_component",
            f"{axis.upper()} component of the canonical palm normal in MediaPipe world coordinates.",
            f"B[{axis},z]",
            orientation_sensitive=True,
        )
    for axis in ("x", "y", "z"):
        _append(
            items,
            f"orientation.hand_axis_camera.{axis}",
            "global_orientation",
            "unit_component",
            f"{axis.upper()} component of the wrist-to-middle-MCP axis in MediaPipe world coordinates.",
            f"B[{axis},y]",
            orientation_sensitive=True,
        )

    if len(items) != 159:
        raise RuntimeError(f"Feature schema construction produced {len(items)} entries instead of 159.")
    return tuple(items)


FEATURE_COUNT = len(feature_definitions())
