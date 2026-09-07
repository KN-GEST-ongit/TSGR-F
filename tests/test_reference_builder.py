from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from tsgr.features.schema import feature_definitions
from tsgr.processing.result_writer import FEATURE_BRANCH_ORDER
from tsgr.reference_models.builder import build_reference_models
from tsgr.reference_models.manifest import ReferenceManifestEntry
from tsgr.reference_models.repository import load_reference_gesture_model


def make_run(
    root: Path,
    *,
    gesture_value: float,
    seed: int,
    filter_name: str = "half_pound",
) -> Path:
    run = root
    features = run / "features"
    features.mkdir(parents=True)
    ids = [definition.feature_id for definition in feature_definitions()]
    rng = np.random.default_rng(seed)
    frame_count = 24
    values = rng.normal(0.0, 0.015, size=(frame_count, 4, len(ids)))
    values[:, :, 0:3] = 0.0
    values[:, :, 27] = 1.0
    values[:, :, 10:16] += gesture_value
    np.savez_compressed(
        features / "feature_arrays.npz",
        frame_index=np.arange(frame_count),
        relative_time_s=np.arange(frame_count) / 30.0,
        status=np.asarray(["valid"] * frame_count),
        classification_feature_branch=np.asarray(["raw.wrist_middle_mcp"] * frame_count),
        values=values,
        branch_names=np.asarray(FEATURE_BRANCH_ORDER),
        feature_ids=np.asarray(ids),
    )
    (features / "feature_schema.json").write_text(
        json.dumps(
            {
                "schema_version": "tsgr_full_features_v1",
                "feature_count": len(ids),
                "branch_order": list(FEATURE_BRANCH_ORDER),
                "features": [definition.to_dict() for definition in feature_definitions()],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "camera": {"input_is_mirrored": False},
        "temporal_filter": {
            "active": filter_name,
            "filters": {
                "half_pound": {"minimum_cutoff_hz": 1.0},
                "one_euro": {"minimum_cutoff_hz": 1.0},
            },
        },
        "paths": {"model_path": "models/hand_landmarker.task"},
    }
    (run / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (run / "environment.json").write_text(
        json.dumps({"model": {"sha256": "same_model_hash"}}), encoding="utf-8"
    )
    return run


def build_config() -> dict:
    return {
        "branches": ["raw.wrist_middle_mcp", "filtered.wrist_middle_mcp"],
        "session_balance": {"mode": "minimum_available", "fixed_count": None, "seed": 2026},
        "outlier_detection": {
            "feature_z_threshold": 6.0,
            "robust_rms_threshold": 4.0,
            "maximum_extreme_fraction": 0.05,
            "scale_floor": 1.0e-6,
            "maximum_reported_features": 8,
            "minimum_accepted_frames_per_session": 10,
        },
        "feature_mask": {
            "exact_tolerance": 1.0e-12,
            "absolute_std_threshold": 1.0e-8,
            "relative_std_threshold": 1.0e-6,
        },
        "compact_feature_mask": {"correlation_threshold": 0.995},
        "covariance": {"shrinkage": 0.1, "diagonal_floor": 1.0e-8},
    }


def test_builder_writes_balanced_models_masks_statistics_and_compatibility(tmp_path: Path) -> None:
    entries = []
    seed = 1
    for gesture, gesture_value in (("A", 0.0), ("B", 0.6)):
        for person in ("p1", "p2"):
            for session in ("s1", "s2"):
                run = make_run(
                    tmp_path / "runs" / gesture / person / session / "run_001",
                    gesture_value=gesture_value,
                    seed=seed,
                )
                seed += 1
                entries.append(ReferenceManifestEntry(gesture, person, session, run))
    result = build_reference_models(
        entries,
        branches=["raw.wrist_middle_mcp", "filtered.wrist_middle_mcp"],
        output_root=tmp_path / "models",
        config=build_config(),
        model_set_name="test_set",
    )
    assert result.model_count == 4
    assert (result.output_dir / "model_set.json").is_file()
    assert (result.output_dir / "branches" / "raw__wrist_middle_mcp" / "feature_schema_snapshot.json").is_file()
    assert (result.output_dir / "branches" / "raw__wrist_middle_mcp" / "session_outliers.csv").is_file()
    model = load_reference_gesture_model(
        result.output_dir,
        gesture_id="A",
        branch_name="raw.wrist_middle_mcp",
    )
    assert model.prototype_mean.shape == (159,)
    assert model.active_mask.shape == (159,)
    assert int((~model.active_mask).sum()) >= 4
    assert model.covariance_regularized_active.shape[0] == int(model.active_mask.sum())
    assert np.all(np.linalg.eigvalsh(model.covariance_regularized_active) > 0)


def test_builder_rejects_mixed_filtered_provenance(tmp_path: Path) -> None:
    first = make_run(tmp_path / "a", gesture_value=0.0, seed=1, filter_name="half_pound")
    second = make_run(tmp_path / "b", gesture_value=0.1, seed=2, filter_name="one_euro")
    entries = [
        ReferenceManifestEntry("A", "p1", "s1", first),
        ReferenceManifestEntry("A", "p2", "s1", second),
    ]
    try:
        build_reference_models(
            entries,
            branches=["filtered.wrist_middle_mcp"],
            output_root=tmp_path / "models",
            config=build_config(),
            model_set_name="bad",
        )
    except ValueError as error:
        assert "Incompatible runs" in str(error)
    else:
        raise AssertionError("Expected strict filtered-branch compatibility failure.")


def test_model_loader_detects_tampered_active_mask(tmp_path: Path) -> None:
    entries = []
    for seed, (gesture, value) in enumerate((("A", 0.0), ("B", 0.6)), start=1):
        run = make_run(
            tmp_path / gesture / "p1" / "s1" / "run_001",
            gesture_value=value,
            seed=seed,
        )
        entries.append(ReferenceManifestEntry(gesture, "p1", "s1", run))
    result = build_reference_models(
        entries,
        branches=["raw.wrist_middle_mcp"],
        output_root=tmp_path / "models",
        config=build_config(),
        model_set_name="tamper",
    )
    arrays_path = (
        result.output_dir
        / "branches"
        / "raw__wrist_middle_mcp"
        / "gestures"
        / "A"
        / "model_arrays.npz"
    )
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["active_mask"] = arrays["active_mask"].copy()
    arrays["active_mask"][10] = ~arrays["active_mask"][10]
    np.savez_compressed(arrays_path, **arrays)
    try:
        load_reference_gesture_model(
            result.output_dir,
            gesture_id="A",
            branch_name="raw.wrist_middle_mcp",
        )
    except ValueError as error:
        assert "indices" in str(error) or "hash" in str(error)
    else:
        raise AssertionError("Expected tampered model-array validation failure.")
