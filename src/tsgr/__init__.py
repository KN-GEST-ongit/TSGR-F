"""Trustworthy Static Gesture Recognition Framework."""

from .features import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    extract_feature_vector,
    feature_definitions,
)
from .preprocessing.spatial_normalization import (
    SpatialNormalizationError,
    SpatialNormalizationResult,
    normalize_world_landmarks,
)
from .types import FrameAnalysis, FramePacket, HandCandidate

__all__ = [
    "FEATURE_COUNT",
    "FEATURE_SCHEMA_VERSION",
    "FeatureVector",
    "FrameAnalysis",
    "FramePacket",
    "HandCandidate",
    "SpatialNormalizationError",
    "SpatialNormalizationResult",
    "extract_feature_vector",
    "feature_definitions",
    "normalize_world_landmarks",
]
__version__ = "0.42.15"
