#!/usr/bin/env python
"""Standalone SGRF worker executed by the external Python 3.11 environment.

This module intentionally has no imports from :mod:`tsgr`.  The main framework
runs on its pinned Python 3.12.10 environment, while the upstream SGRF toolbox
is validated in a separate Python 3.11 environment.  Communication uses JSON
job/result files and CSV manifests.
"""

from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


GESTURE_ORDER = ("A", "B", "C", "E", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")

_COORD_CLASSES: dict[str, tuple[str, str, str, str]] = {
    "ADITHYA_RAJESH": (
        "sgrf.algorithms.adithya_rajesh.adithya_rajesh_payload", "AdithyaRajeshPayload",
        "sgrf.algorithms.adithya_rajesh.adithya_rajesh_learning_data", "AdithyaRajeshLearningData",
    ),
    "CHANG_CHEN": (
        "sgrf.algorithms.chang_chen.chang_chen_payload", "ChangChenPayload",
        "sgrf.algorithms.chang_chen.chang_chen_learning_data", "ChangChenLearningData",
    ),
    "GUPTA_JAAFAR": (
        "sgrf.algorithms.gupta_jaafar.gupta_jaafar_payload", "GuptaJaafarPayload",
        "sgrf.algorithms.gupta_jaafar.gupta_jaafar_learning_data", "GuptaJaafarLearningData",
    ),
    "JOSHI_KUMAR": (
        "sgrf.algorithms.joshi_kumar.joshi_kumar_payload", "JoshiKumarPayload",
        "sgrf.algorithms.joshi_kumar.joshi_kumar_learning_data", "JoshiKumarLearningData",
    ),
    "MAUNG": (
        "sgrf.algorithms.maung.maung_payload", "MaungPayload",
        "sgrf.algorithms.maung.maung_learning_data", "MaungLearningData",
    ),
    "MOHANTY_RAMBHATLA": (
        "sgrf.algorithms.mohanty_rambhatla.mohanty_rambhatla_payload", "MohantyRambhatlaPayload",
        "sgrf.algorithms.mohanty_rambhatla.mohanty_rambhatla_learning_data", "MohantyRambhatlaLearningData",
    ),
    "NGUYEN_HUYNH": (
        "sgrf.algorithms.nguyen_huynh.nguyen_huynh_payload", "NguyenHuynhPayload",
        "sgrf.algorithms.nguyen_huynh.nguyen_huynh_learning_data", "NguyenHuynhLearningData",
    ),
    "OYEDOTUN_KHASHMAN": (
        "sgrf.algorithms.oyedotun_khashman.oyedotun_khashman_payload", "OyedotunKhashmanPayload",
        "sgrf.algorithms.oyedotun_khashman.oyedotun_khashman_learning_data", "OyedotunKhashmanLearningData",
    ),
    "PINTO_BORGES": (
        "sgrf.algorithms.pinto_borges.pinto_borges_payload", "PintoBorgesPayload",
        "sgrf.algorithms.pinto_borges.pinto_borges_learning_data", "PintoBorgesLearningData",
    ),
    "ZHUANG_YANG": (
        "sgrf.algorithms.zhuang_yang.zhuang_yang_payload", "ZhuangYangPayload",
        "sgrf.algorithms.zhuang_yang.zhuang_yang_learning_data", "ZhuangYangLearningData",
    ),
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _environment_payload() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sgrf_version": _package_version("sgrf"),
        "numpy_version": _package_version("numpy"),
        "opencv_python_version": _package_version("opencv-python"),
        "tensorflow_cpu_version": _package_version("tensorflow-cpu"),
        "tensorflow_version": _package_version("tensorflow"),
        "keras_version": _package_version("keras"),
        "scikit_learn_version": _package_version("scikit-learn"),
        "scikit_image_version": _package_version("scikit-image"),
    }


def _audit() -> dict[str, Any]:
    payload = _environment_payload()
    error = ""
    algorithms: list[str] = []
    adapter_imports: dict[str, str] = {}
    try:
        from sgrf.data.algorithm import ALGORITHM
        algorithms = [item.value for item in ALGORITHM]
        # Importing the public functions forces the complete algorithm registry to
        # load and therefore detects missing runtime dependencies early.
        from sgrf import classify, learn  # noqa: F401
        from sgrf.models.image_payload import ImagePayload  # noqa: F401
        from sgrf.models.learning_data import LearningData  # noqa: F401
        for method_id, (payload_module, payload_class, learning_module, learning_class) in _COORD_CLASSES.items():
            _import_class(payload_module, payload_class)
            _import_class(learning_module, learning_class)
            adapter_imports[method_id] = "ok"
    except BaseException as exc:  # pragma: no cover - executed in external env
        error = f"{type(exc).__name__}: {exc}"
    payload.update(
        {
            "sgrf_import_ok": not bool(error),
            "sgrf_import_error": error,
            "available_algorithms": algorithms,
            "adapter_imports": adapter_imports,
            "reference_python_series": "3.11",
            "reference_sgrf_version": "3.2.0",
        }
    )
    return payload


def _gesture_enum(gestures: list[str] | tuple[str, ...]):
    normalized = [str(value).strip().upper() for value in gestures]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Gesture list contains duplicates.")
    if not normalized:
        raise ValueError("Gesture list is empty.")
    return enum.Enum("TSGRGesture", {name: index + 1 for index, name in enumerate(normalized)})




def _normalize_certainty(value: Any, scale: str) -> float | None:
    """Normalize heterogeneous upstream certainty conventions to [0, 1]."""
    if value is None:
        return None
    numeric = float(value)
    if not (numeric == numeric):
        return None
    if scale == "unit":
        normalized = numeric
    elif scale == "percent":
        normalized = numeric / 100.0
    elif scale == "none":
        return None
    else:
        raise ValueError(f"Unsupported certainty scale: {scale!r}")
    return min(1.0, max(0.0, normalized))

def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def _coordinates(policy: str, width: int, height: int):
    if policy == "none":
        return None
    if policy == "full_frame":
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for full-frame coordinates: {width}x{height}")
        return [(0, 0), (int(width), int(height))]
    raise ValueError(f"Unsupported coordinate policy: {policy!r}")


def _import_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _payload_for(method_id: str, image, *, coordinate_policy: str):
    from sgrf.models.image_payload import ImagePayload

    if method_id not in _COORD_CLASSES:
        return ImagePayload(image=image)
    payload_module, payload_class, _, _ = _COORD_CLASSES[method_id]
    cls = _import_class(payload_module, payload_class)
    height, width = image.shape[:2]
    return cls(image=image, coords=_coordinates(coordinate_policy, width, height))


def _learning_item(method_id: str, row: dict[str, str], label, *, coordinate_policy: str):
    from sgrf.models.learning_data import LearningData

    image_path = str(Path(row["image_path"]).resolve())
    if method_id not in _COORD_CLASSES:
        return LearningData(image_path=image_path, label=label)
    _, _, learning_module, learning_class = _COORD_CLASSES[method_id]
    cls = _import_class(learning_module, learning_class)
    width = int(float(row.get("width", "0") or 0))
    height = int(float(row.get("height", "0") or 0))
    if coordinate_policy == "full_frame" and (width <= 0 or height <= 0):
        import cv2
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read training image for dimensions: {image_path}")
        height, width = image.shape[:2]
    return cls(image_path=image_path, coords=_coordinates(coordinate_policy, width, height), label=label)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(model_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in model_dir.rglob("*") if candidate.is_file()):
        output.append(
            {
                "relative_path": path.relative_to(model_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    return output


def _train(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(int(job.get("seed", 2026)))
    from sgrf import learn
    from sgrf.data.algorithm import ALGORITHM

    method_id = str(job["method_id"]).upper()
    algorithm = ALGORITHM[method_id]
    gestures = list(job.get("gestures") or GESTURE_ORDER)
    gesture_enum = _gesture_enum(gestures)
    manifest_rows = _read_csv(job["training_manifest"])
    if not manifest_rows:
        raise ValueError("Training manifest is empty.")
    labels = set(gestures)
    unknown = sorted({str(row.get("gesture_id", "")).upper() for row in manifest_rows} - labels)
    if unknown:
        raise ValueError(f"Training manifest contains gestures outside the configured enum: {unknown}")
    coordinate_policy = str(job.get("coordinate_policy", "none"))
    learning_data = [
        _learning_item(
            method_id,
            row,
            gesture_enum[str(row["gesture_id"]).upper()],
            coordinate_policy=coordinate_policy,
        )
        for row in manifest_rows
    ]
    model_dir = Path(job["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    custom_options = dict(job.get("custom_options") or {})
    custom_options["gesture_enum"] = gesture_enum
    train_started = time.perf_counter()
    accuracy, loss = learn(
        algorithm=algorithm,
        learning_data=learning_data,
        target_model_path=str(model_dir),
        custom_options=custom_options,
    )
    training_elapsed = time.perf_counter() - train_started
    return {
        "status": "ok",
        "method_id": method_id,
        "training_sample_count": len(learning_data),
        "accuracy": None if accuracy is None else float(accuracy),
        "loss": None if loss is None else float(loss),
        "training_elapsed_s": training_elapsed,
        "total_elapsed_s": time.perf_counter() - started,
        "model_dir": str(model_dir.resolve()),
        "model_files": _model_files(model_dir),
        "environment": _environment_payload(),
    }


def _install_prediction_model_cache() -> dict[str, int]:
    """Cache model deserialization without changing algorithm preprocessing/classification."""
    stats = {"keras_model_load_calls": 0, "keras_model_cache_hits": 0, "pickle_load_calls": 0, "pickle_cache_hits": 0}
    try:
        import keras
        original_load_model = keras.models.load_model
        keras_cache: dict[str, Any] = {}

        def cached_load_model(path, *args, **kwargs):
            stats["keras_model_load_calls"] += 1
            key = os.path.abspath(os.fspath(path))
            if key in keras_cache:
                stats["keras_model_cache_hits"] += 1
                return keras_cache[key]
            model = original_load_model(path, *args, **kwargs)
            keras_cache[key] = model
            return model

        keras.models.load_model = cached_load_model
    except Exception:
        pass
    try:
        import pickle
        original_pickle_load = pickle.load
        pickle_cache: dict[str, Any] = {}

        def cached_pickle_load(file_obj, *args, **kwargs):
            stats["pickle_load_calls"] += 1
            name = getattr(file_obj, "name", "")
            key = os.path.abspath(os.fspath(name)) if name else ""
            if key and key in pickle_cache:
                stats["pickle_cache_hits"] += 1
                return pickle_cache[key]
            value = original_pickle_load(file_obj, *args, **kwargs)
            if key:
                pickle_cache[key] = value
            return value

        pickle.load = cached_pickle_load
    except Exception:
        pass
    return stats


def _predict(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(int(job.get("seed", 2026)))
    import cv2
    from sgrf import classify
    from sgrf.data.algorithm import ALGORITHM

    method_id = str(job["method_id"]).upper()
    algorithm = ALGORITHM[method_id]
    gestures = list(job.get("gestures") or GESTURE_ORDER)
    gesture_enum = _gesture_enum(gestures)
    rows = _read_csv(job["prediction_manifest"])
    model_dir = Path(job["model_dir"])
    coordinate_policy = str(job.get("coordinate_policy", "none"))
    loader_resize = job.get("loader_resize")
    rejection_policy = str(job.get("rejection_policy", "closed_set"))
    certainty_threshold_normalized = job.get("certainty_threshold_normalized")
    certainty_scale = str(job.get("certainty_scale", "none"))
    if rejection_policy == "certainty_reject" and certainty_threshold_normalized is None:
        raise ValueError("certainty_threshold_normalized is required for certainty_reject.")
    if rejection_policy == "certainty_reject" and certainty_scale == "none":
        raise ValueError(f"{method_id} does not expose an upstream certainty value and cannot use certainty_reject.")
    if rejection_policy not in {"closed_set", "certainty_reject"}:
        raise ValueError(f"Unsupported rejection_policy: {rejection_policy!r}")
    model_load_policy = str(job.get("model_load_policy", "process_cache"))
    cache_stats: dict[str, int] = {}
    if model_load_policy == "process_cache":
        cache_stats = _install_prediction_model_cache()
    elif model_load_policy != "upstream_each_call":
        raise ValueError(f"Unsupported model_load_policy: {model_load_policy!r}")
    custom_options = dict(job.get("custom_options") or {})
    custom_options["gesture_enum"] = gesture_enum
    output_rows: list[dict[str, Any]] = []
    failures = 0
    classification_elapsed_s = 0.0
    decode_elapsed_s = 0.0
    for index, row in enumerate(rows, start=1):
        frame_path = Path(row["frame_path"])
        record: dict[str, Any] = {
            "row_id": row.get("row_id", ""),
            "take_id": row.get("take_id", ""),
            "frame_index": row.get("frame_index", ""),
            "frame_path": str(frame_path),
            "raw_predicted_label": "",
            "raw_predicted_state": "",
            "predicted_label": "",
            "predicted_state": "",
            "rejected_by_policy": 0,
            "raw_certainty": "",
            "normalized_certainty": "",
            "image_decode_ms": "",
            "classification_ms": "",
            "error": "",
        }
        try:
            decode_started = time.perf_counter()
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            decode_elapsed = time.perf_counter() - decode_started
            decode_elapsed_s += decode_elapsed
            record["image_decode_ms"] = decode_elapsed * 1000.0
            if image is None:
                raise ValueError(f"Cannot read image: {frame_path}")
            if loader_resize:
                width, height = int(loader_resize[0]), int(loader_resize[1])
                image = cv2.resize(image, (width, height))
            payload = _payload_for(method_id, image, coordinate_policy=coordinate_policy)
            classify_started = time.perf_counter()
            prediction, certainty = classify(
                algorithm=algorithm,
                payload=payload,
                custom_model_dir=str(model_dir),
                custom_options=custom_options,
            )
            classification_elapsed = time.perf_counter() - classify_started
            classification_elapsed_s += classification_elapsed
            record["classification_ms"] = classification_elapsed * 1000.0
            label = str(getattr(prediction, "name", prediction)).upper()
            score = None if certainty is None else float(certainty)
            normalized_score = _normalize_certainty(score, certainty_scale)
            if label not in gestures:
                raise ValueError(f"SGRF returned label outside the configured gesture enum: {label!r}")
            rejected = rejection_policy == "certainty_reject" and (normalized_score is None or normalized_score < float(certainty_threshold_normalized))
            record["raw_predicted_label"] = label
            record["raw_predicted_state"] = f"GESTURE_{label}"
            record["predicted_label"] = "NO_GESTURE" if rejected else label
            record["predicted_state"] = "NO_GESTURE" if rejected else f"GESTURE_{label}"
            record["rejected_by_policy"] = int(rejected)
            record["raw_certainty"] = "" if score is None else score
            record["normalized_certainty"] = "" if normalized_score is None else normalized_score
        except BaseException as exc:
            failures += 1
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["predicted_state"] = "PREDICTION_ERROR"
            if bool(job.get("fail_fast", False)):
                output_rows.append(record)
                raise
        output_rows.append(record)
        progress_every = int(job.get("progress_every", 0) or 0)
        if progress_every and (index % progress_every == 0 or index == len(rows)):
            print(f"[{method_id} {index}/{len(rows)}] frames", flush=True)
    fields = [
        "row_id", "take_id", "frame_index", "frame_path", "raw_predicted_label", "raw_predicted_state",
        "predicted_label", "predicted_state", "rejected_by_policy", "raw_certainty", "normalized_certainty",
        "image_decode_ms", "classification_ms", "error",
    ]
    _write_csv(job["output_csv"], output_rows, fields)
    return {
        "status": "ok" if failures == 0 else "completed_with_errors",
        "method_id": method_id,
        "frame_count": len(rows),
        "prediction_failure_count": failures,
        "image_decode_elapsed_s": decode_elapsed_s,
        "classification_elapsed_s": classification_elapsed_s,
        "total_elapsed_s": time.perf_counter() - started,
        "model_load_policy": model_load_policy,
        "certainty_scale": certainty_scale,
        "certainty_threshold_normalized": certainty_threshold_normalized,
        "model_cache_stats": cache_stats,
        "output_csv": str(Path(job["output_csv"]).resolve()),
        "environment": _environment_payload(),
    }


def _run_command(command: str, job_path: Path | None) -> dict[str, Any]:
    if command == "audit":
        return _audit()
    if job_path is None:
        raise ValueError(f"--job-json is required for command {command}.")
    job = _read_json(job_path)
    if command == "train":
        return _train(job)
    if command == "predict":
        return _predict(job)
    raise ValueError(f"Unsupported command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "train", "predict"))
    parser.add_argument("--job-json", type=Path, default=None)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = _run_command(args.command, args.job_json)
        result.setdefault("worker_status", "ok")
        result.setdefault("worker_elapsed_s", time.perf_counter() - started)
        _write_json(args.result_json, result)
    except BaseException as exc:
        payload = {
            "worker_status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "worker_elapsed_s": time.perf_counter() - started,
            "environment": _environment_payload(),
        }
        _write_json(args.result_json, payload)
        raise


if __name__ == "__main__":
    main()
