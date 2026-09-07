"""Runtime and model-footprint benchmark for the frozen TSGR-F routing classifier.

The benchmark uses cached feature arrays and persisted IMAGE landmarks. It deliberately
separates classifier timing from MediaPipe inference so the result is reproducible
without reprocessing videos. A second value reports persisted IMAGE-feature extraction
cost from stored landmarks/archive access.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tsgr.dataset.contract import resolve_dataset_root_from_plan
from tsgr.evaluation.experiment_evaluator import _load_run_features
from tsgr.evaluation.frozen_routing_v40 import (
    _load_model, _run_image_features, _query_class_scores, _query_fused_score, _query_pair_scores, _top2_gate
)
from tsgr.evaluation.io_utils import read_csv_rows, write_workbook
from tsgr.utils.serialization import write_json

SCHEMA = "tsgrf_frozen_routing_runtime_v42"


def _timed_components(model: dict[str, Any], arrays: dict[str, np.ndarray], qfull: np.ndarray, q2: np.ndarray) -> dict[str, float]:
    gestures = tuple(str(g) for g in model["gesture_ids"]); gi = {g:i for i,g in enumerate(gestures)}
    fidx = np.asarray(arrays["feature_indices"], dtype=np.int64)
    q3 = np.asarray(qfull[:, fidx], dtype=np.float64); q2 = np.asarray(q2, dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64); train3=np.asarray(arrays["train3"],dtype=np.float64); train2=np.asarray(arrays["train2"],dtype=np.float64)
    w3=np.asarray(arrays["w3"],dtype=np.float64); w2=np.asarray(arrays["w2"],dtype=np.float64)
    total_start=time.perf_counter()
    t=time.perf_counter(); scores=_query_class_scores(q3,train3,labels,w3,len(gestures)); order=np.argsort(scores,axis=1); base=order[:,0].copy(); global_s=time.perf_counter()-t
    t=time.perf_counter(); alpha=float(model["os_specialist"]["alpha"]); so=_query_fused_score(q3,q2,train3,train2,labels,gi["O"],w3,w2,alpha); ss=_query_fused_score(q3,q2,train3,train2,labels,gi["S"],w3,w2,alpha); gate_os=_top2_gate(scores,gi["O"],gi["S"]); pred_os=base.copy(); pred_os[gate_os]=np.where((so-ss)[gate_os]<=float(model["os_specialist"]["threshold"]),gi["O"],gi["S"]); os_s=time.perf_counter()-t
    t=time.perf_counter(); comb=np.column_stack((q3,q2)); train_comb=np.column_stack((train3,train2)); iy_idx=np.asarray(arrays["iy_indices"],dtype=np.int64); iy=_query_pair_scores(comb[:,iy_idx],train_comb[:,iy_idx],labels,gi["I"],gi["Y"],np.asarray(arrays["iy_weights"],dtype=np.float64)); gate_iy=_top2_gate(scores,gi["I"],gi["Y"]); pred_iy=pred_os.copy(); pred_iy[gate_iy]=np.where(iy[gate_iy,0]<=iy[gate_iy,1],gi["I"],gi["Y"]); iy_s=time.perf_counter()-t
    t=time.perf_counter(); cidx=np.asarray(arrays["c_indices"],dtype=np.int64); cx=comb[:,cidx]; za=np.sqrt(np.mean(((cx-arrays["c_a_center"])/arrays["c_a_scale"])**2,axis=1)); zc=np.sqrt(np.mean(((cx-arrays["c_c_center"])/arrays["c_c_scale"])**2,axis=1)); final=pred_iy.copy(); rescue=(base==gi["A"])&(zc<za); final[rescue]=gi["C"]; c_s=time.perf_counter()-t
    total_s=time.perf_counter()-total_start
    n=max(1,len(qfull))
    return {
        "global_ranking_ms_per_frame": global_s*1000.0/n,
        "os_specialist_ms_per_frame": os_s*1000.0/n,
        "iy_specialist_ms_per_frame": iy_s*1000.0/n,
        "c_rescue_ms_per_frame": c_s*1000.0/n,
        "frozen_routing_ms_per_frame": total_s*1000.0/n,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "median", "p95", "min", "max")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _sample_fold_queries(
    fold_dir: Path,
    *,
    model: dict[str, Any],
    arrays: dict[str, np.ndarray],
    manifest: dict[str, dict[str, str]],
    dataset_root: Path,
    branch_name: str,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    qfull_rows: list[np.ndarray] = []
    q2_rows: list[np.ndarray] = []
    image_feature_seconds = 0.0
    image_feature_frames = 0
    expected = tuple(str(v) for v in arrays["image_feature_ids"].tolist())
    takes = [str(r["take_id"]) for r in read_csv_rows(fold_dir / "test_takes.csv")]
    for take in sorted(takes):
        if len(qfull_rows) >= max_frames:
            break
        row = manifest.get(take)
        if row is None:
            continue
        run = Path(str(row.get("run_path", "")))
        run = run if run.is_absolute() else dataset_root / run
        if not run.is_dir():
            continue
        frame_indices, statuses, values = _load_run_features(run, branch_name)
        started = time.perf_counter()
        img, ok, ids = _run_image_features(run, frame_indices)
        image_feature_seconds += time.perf_counter() - started
        image_feature_frames += len(frame_indices)
        if ids and tuple(ids) != expected:
            raise ValueError(f"IMAGE feature schema mismatch in runtime benchmark: {take}")
        complete = np.isfinite(values).all(axis=1) & ok
        for idx in np.flatnonzero(complete).tolist():
            if len(qfull_rows) >= max_frames:
                break
            qfull_rows.append(np.asarray(values[idx], dtype=np.float64))
            q2_rows.append(np.asarray(img[idx], dtype=np.float64))
    if not qfull_rows:
        raise ValueError(f"No complete cached queries available for runtime benchmark in {fold_dir}")
    return np.vstack(qfull_rows), np.vstack(q2_rows), image_feature_seconds, image_feature_frames


def benchmark_frozen_routing(
    experiment_plan_dir: str | Path,
    *,
    frozen_model_dir: str | Path,
    test_processing_report: str | Path,
    output_dir: str | Path,
    branch_name: str,
    scenarios: set[str] | None = None,
    folds: set[str] | None = None,
    max_frames_per_fold: int = 1000,
    batch_size: int = 64,
    repeats: int = 5,
    warmup: int = 2,
    dataset_root_override: str | Path | None = None,
    progress: bool = True,
) -> Path:
    if max_frames_per_fold < 1 or batch_size < 1 or repeats < 1 or warmup < 0:
        raise ValueError("Invalid runtime benchmark parameters.")
    plan = Path(experiment_plan_dir)
    dataset = resolve_dataset_root_from_plan(plan, override=dataset_root_override)
    models = Path(frozen_model_dir)
    proc = Path(test_processing_report)
    manifest_rows = read_csv_rows(proc / "test_take_mediapipe_status.csv")
    if not manifest_rows:
        raise ValueError(f"Missing test_take_mediapipe_status.csv in {proc}")
    manifest = {str(r["take_id"]): r for r in manifest_rows}
    processing_report_path = proc / "processing_report.json"
    processing_report = json.loads(processing_report_path.read_text(encoding="utf-8")) if processing_report_path.is_file() else {}
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)

    fold_rows: list[dict[str, Any]] = []
    for fold_json in sorted(plan.glob("*/*/fold.json")):
        meta = json.loads(fold_json.read_text(encoding="utf-8"))
        scenario = str(meta["scenario"]); fold_id = str(meta["fold_id"])
        if scenarios and scenario not in scenarios:
            continue
        if folds and fold_id not in folds:
            continue
        if str(meta.get("fold_status", "active")) != "active":
            continue
        try:
            model, arrays = _load_model(models, scenario, fold_id)
        except FileNotFoundError:
            continue
        qfull, q2, img_seconds, img_frames = _sample_fold_queries(
            fold_json.parent,
            model=model,
            arrays=arrays,
            manifest=manifest,
            dataset_root=dataset,
            branch_name=branch_name,
            max_frames=int(max_frames_per_fold),
        )
        n = len(qfull)
        batches = [(qfull[i:i+batch_size], q2[i:i+batch_size]) for i in range(0, n, batch_size)]
        for _ in range(int(warmup)):
            for qb, qi in batches:
                _timed_components(model, arrays, qb, qi)
        component_values: dict[str, list[float]] = defaultdict(list)
        for _ in range(int(repeats)):
            weighted: dict[str, float] = defaultdict(float); processed = 0
            for qb, qi in batches:
                timing = _timed_components(model, arrays, qb, qi)
                for key, value in timing.items(): weighted[key] += float(value) * len(qb)
                processed += len(qb)
            for key, total in weighted.items(): component_values[key].append(total / max(1, processed))
        elapsed_ms_per_frame = component_values["frozen_routing_ms_per_frame"]
        fold_model_dir = models / scenario / fold_id
        model_bytes = sum(p.stat().st_size for p in fold_model_dir.rglob("*") if p.is_file())
        arrays_bytes = sum(int(np.asarray(v).nbytes) for v in arrays.values())
        row = {
            "scenario": scenario,
            "fold_id": fold_id,
            "training_sample_count": int(model.get("training_sample_count", len(arrays["labels"]))),
            "global_feature_count": int(np.asarray(arrays["train3"]).shape[1]),
            "image2d_feature_count": int(np.asarray(arrays["train2"]).shape[1]),
            "iy_feature_count": int(np.asarray(arrays["iy_indices"]).size),
            "c_rescue_feature_count": int(np.asarray(arrays["c_indices"]).size),
            "benchmark_query_frames": n,
            "batch_size": int(batch_size),
            "repeats": int(repeats),
            "warmup_runs": int(warmup),
            "model_files_bytes": int(model_bytes),
            "model_arrays_uncompressed_bytes": int(arrays_bytes),
            "persisted_image_feature_build_ms_per_frame": (img_seconds * 1000.0 / img_frames) if img_frames else float("nan"),
            **_stats(component_values["global_ranking_ms_per_frame"], "global_ranking_ms_per_frame"),
            **_stats(component_values["os_specialist_ms_per_frame"], "os_specialist_ms_per_frame"),
            **_stats(component_values["iy_specialist_ms_per_frame"], "iy_specialist_ms_per_frame"),
            **_stats(component_values["c_rescue_ms_per_frame"], "c_rescue_ms_per_frame"),
            **_stats(elapsed_ms_per_frame, "frozen_routing_ms_per_frame"),
        }
        row["frozen_routing_fps_from_mean"] = 1000.0 / row["frozen_routing_ms_per_frame_mean"] if row["frozen_routing_ms_per_frame_mean"] > 0 else float("nan")
        fold_rows.append(row)
        if progress:
            print(f"[{scenario}/{fold_id}] routing={row['frozen_routing_ms_per_frame_mean']:.4f} ms/frame | model={model_bytes/1024/1024:.2f} MiB", flush=True)
    if not fold_rows:
        raise ValueError("No frozen models selected for runtime benchmark.")

    scenario_rows: list[dict[str, Any]] = []
    for scenario in sorted({str(r["scenario"]) for r in fold_rows}):
        rows = [r for r in fold_rows if str(r["scenario"]) == scenario]
        scenario_rows.append({
            "scenario": scenario,
            "fold_count": len(rows),
            "training_sample_count_mean": float(np.mean([float(r["training_sample_count"]) for r in rows])),
            "global_feature_count_mean": float(np.mean([float(r["global_feature_count"]) for r in rows])),
            "model_files_mib_mean": float(np.mean([float(r["model_files_bytes"]) for r in rows])) / 1024.0 / 1024.0,
            "frozen_routing_ms_per_frame_mean_of_folds": float(np.mean([float(r["frozen_routing_ms_per_frame_mean"]) for r in rows])),
            "frozen_routing_ms_per_frame_median_of_folds": float(np.median([float(r["frozen_routing_ms_per_frame_mean"]) for r in rows])),
            "persisted_image_feature_build_ms_per_frame_mean_of_folds": float(np.mean([float(r["persisted_image_feature_build_ms_per_frame"]) for r in rows])),
        })

    source_processing_rows = [{
        "test_processing_report": str(proc),
        "test_mode": processing_report.get("test_mode", manifest_rows[0].get("test_mode", "")),
        "detection_profile": processing_report.get("detection_profile", manifest_rows[0].get("detection_profile", "")),
        "handedness_policy": processing_report.get("handedness_policy", manifest_rows[0].get("handedness_policy", "")),
        "take_count": int(processing_report.get("take_count", len(manifest_rows)) or 0),
        "processed_takes": int(processing_report.get("processed_takes", 0) or 0),
        "reused_takes": int(processing_report.get("reused_takes", 0) or 0),
        "processed_frame_count": int(processing_report.get("processed_frame_count", sum(int(r.get("processed_frames") or 0) for r in manifest_rows)) or 0),
        "processing_elapsed_wall_time_s": float(processing_report.get("processing_elapsed_wall_time_s", 0.0) or 0.0),
        "aggregate_processing_frames_per_second": float(processing_report.get("aggregate_processing_frames_per_second", 0.0) or 0.0),
        "serial_equivalent_sum_take_time_s": float(processing_report.get("serial_equivalent_sum_take_time_s", sum(float(r.get("elapsed_wall_time_s") or 0.0) for r in manifest_rows)) or 0.0),
        "estimated_parallel_speedup": float(processing_report.get("estimated_parallel_speedup", 0.0) or 0.0),
        "resolved_workers": int(processing_report.get("resolved_workers", 0) or 0),
        "recovered_frame_count": int(processing_report.get("recovered_frame_count", sum(int(r.get("recovered_frames") or 0) for r in manifest_rows)) or 0),
        "source_is_historical_processing_telemetry": 1,
    }]
    _write_csv(out / "runtime_by_fold.csv", fold_rows)
    _write_csv(out / "runtime_by_scenario.csv", scenario_rows)
    _write_csv(out / "source_processing_runtime.csv", source_processing_rows)
    write_workbook(
        out / "frozen_routing_runtime.xlsx",
        {
            "by_fold": (list(fold_rows[0]), fold_rows),
            "by_scenario": (list(scenario_rows[0]), scenario_rows),
            "source_processing": (list(source_processing_rows[0]), source_processing_rows),
        },
    )
    write_json(
        out / "frozen_routing_runtime_report.json",
        {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": str(plan),
            "frozen_model_dir": str(models),
            "test_processing_report": str(proc),
            "branch_name": branch_name,
            "max_frames_per_fold": int(max_frames_per_fold),
            "batch_size": int(batch_size),
            "repeats": int(repeats),
            "warmup": int(warmup),
            "notes": [
                "Classifier timing excludes MediaPipe and disk I/O.",
                "persisted_image_feature_build_ms_per_frame includes archive access plus 2D feature construction from stored landmarks.",
                "source_processing_runtime.csv preserves the original IMAGE/VIDEO processing telemetry (including MediaPipe) from processing_report.json; it is not a new controlled timing run.",
                "Use the same hardware/power mode when comparing controlled classifier implementations.",
            ],
        },
    )
    return out / "frozen_routing_runtime_report.json"
