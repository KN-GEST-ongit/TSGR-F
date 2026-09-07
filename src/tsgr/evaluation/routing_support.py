"""Selective-routing helpers used by the final TSGR-F model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tsgr.evaluation.ranking_support import fisher_scores, normalize_positive_weights


DEFAULT_IY_FEATURE_IDS: tuple[str, ...] = (
    "img2d.bend.index.pip",
    "img2d.straightness.index",
    "spread.segment_mean.thumb_index",
    "img2d.distance.wrist_tip.index",
    "bend.index.pip",
)


@dataclass(slots=True)
class GateMetrics:
    pair: str
    pair_errors: int
    pair_accuracy: float
    gated_count: int
    gated_changed: int
    gated_improved: int
    gated_worsened: int
    global_correct: int
    global_errors: int
    global_accuracy: float


@dataclass(slots=True)
class RoutingMetrics:
    stage: str
    policy: str
    gated_count: int
    gated_changed: int
    gated_improved: int
    gated_worsened: int
    global_correct: int
    global_errors: int
    global_accuracy: float


@dataclass(slots=True)
class ThresholdInterval:
    state_index: int
    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool
    threshold_representative: float
    pair_errors: int
    pair_accuracy: float
    gated_count: int
    gated_changed: int
    gated_improved: int
    gated_worsened: int
    global_errors_after_os: int
    global_accuracy_after_os: float


def feature_indices(all_ids: Sequence[str], selected_ids: Sequence[str]) -> np.ndarray:
    """Resolve a fixed feature-ID subset against the current combined schema."""
    lookup = {str(feature_id): index for index, feature_id in enumerate(all_ids)}
    missing = [str(feature_id) for feature_id in selected_ids if str(feature_id) not in lookup]
    if missing:
        raise ValueError(f"Required fixed feature IDs are missing: {missing}")
    return np.asarray([lookup[str(feature_id)] for feature_id in selected_ids], dtype=np.int64)


def _pair_gate(baseline_scores: np.ndarray, class_a: int, class_b: int) -> np.ndarray:
    order = np.argsort(np.asarray(baseline_scores, dtype=np.float64), axis=1)
    top2 = np.sort(order[:, :2], axis=1)
    low, high = sorted((int(class_a), int(class_b)))
    return (top2[:, 0] == low) & (top2[:, 1] == high)


def _binary_pair_predictions(
    values: np.ndarray,
    labels: np.ndarray,
    class_a: int,
    class_b: int,
    selected_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    mask = (labels == class_a) | (labels == class_b)
    global_indices = np.flatnonzero(mask)
    pair_values = np.asarray(values[global_indices][:, selected_features], dtype=np.float64)
    pair_labels = labels[global_indices]
    variances = []
    for class_index in (class_a, class_b):
        rows = pair_values[pair_labels == class_index]
        variances.append(np.var(rows, axis=0, ddof=1))
    within = np.mean(np.stack(variances, axis=0), axis=0)
    positive = within[within > 0]
    floor = max(float(np.median(positive)) * 1e-3 if positive.size else 1e-6, 1e-9)
    weights = normalize_positive_weights(1.0 / np.maximum(within, floor))
    scores = np.full((len(global_indices), 2), np.inf, dtype=np.float64)
    for column, class_index in enumerate((class_a, class_b)):
        candidates = np.flatnonzero(pair_labels == class_index)
        delta = pair_values[:, None, :] - pair_values[candidates][None, :, :]
        distance = np.sqrt(np.sum(delta * delta * weights[None, None, :], axis=2))
        own = {int(global_i): int(local) for local, global_i in enumerate(candidates.tolist())}
        for sample_index in range(len(global_indices)):
            row = distance[sample_index].copy()
            local = own.get(sample_index)
            if local is not None:
                row[local] = np.inf
            scores[sample_index, column] = float(np.min(row))
    pair_pred = np.where(scores[:, 0] <= scores[:, 1], class_a, class_b)
    full_pred = np.full(labels.shape[0], -1, dtype=np.int64)
    full_pred[global_indices] = pair_pred
    errors = int(np.sum(pair_pred != pair_labels))
    accuracy = float(np.mean(pair_pred == pair_labels))
    return full_pred, scores, errors, accuracy


def apply_fixed_iy_gate(
    baseline_scores: np.ndarray,
    current_pred: np.ndarray,
    baseline_pred: np.ndarray,
    combined_values: np.ndarray,
    labels: np.ndarray,
    *,
    class_i: int,
    class_y: int,
    selected_features: np.ndarray,
) -> tuple[np.ndarray, GateMetrics, np.ndarray]:
    """Apply the fixed I/Y specialist on samples whose baseline Top-2 is {I,Y}."""
    full_pair_pred, pair_scores, pair_errors, pair_accuracy = _binary_pair_predictions(
        combined_values,
        labels,
        class_i,
        class_y,
        selected_features,
    )
    gate = _pair_gate(baseline_scores, class_i, class_y)
    pred = current_pred.copy()
    pred[gate] = full_pair_pred[gate]
    before = current_pred[gate] == labels[gate]
    after = pred[gate] == labels[gate]
    global_correct = pred == labels
    metrics = GateMetrics(
        pair="I/Y",
        pair_errors=pair_errors,
        pair_accuracy=pair_accuracy,
        gated_count=int(np.sum(gate)),
        gated_changed=int(np.sum(pred[gate] != current_pred[gate])),
        gated_improved=int(np.sum((~before) & after)),
        gated_worsened=int(np.sum(before & (~after))),
        global_correct=int(np.sum(global_correct)),
        global_errors=int(np.sum(~global_correct)),
        global_accuracy=float(np.mean(global_correct)),
    )
    return pred, metrics, pair_scores


def safe_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return robust median/MAD location and scale with a deterministic floor."""
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None, :]), axis=0) * 1.4826
    positive = mad[mad > 0]
    floor = max(float(np.median(positive)) * 1e-3 if positive.size else 1e-6, 1e-9)
    return center, np.maximum(mad, floor)


def _one_class_conformity_score(query: np.ndarray, reference: np.ndarray, *, method: str) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    x = np.asarray(query, dtype=np.float64)
    if ref.shape[0] < 2:
        return float("inf")
    center, scale = safe_scale(ref)
    z = np.abs((x - center) / scale)
    if method == "robust_z_rms":
        return float(np.sqrt(np.mean(z * z)))
    if method == "robust_z_mean":
        return float(np.mean(z))
    if method == "quantile_interval":
        q05 = np.quantile(ref, 0.05, axis=0)
        q95 = np.quantile(ref, 0.95, axis=0)
        width = np.maximum(q95 - q05, scale)
        below = np.maximum(q05 - x, 0.0)
        above = np.maximum(x - q95, 0.0)
        exceed = (below + above) / width
        return float(np.mean(exceed * exceed) + 0.01 * np.mean(z))
    if method == "empirical_tail":
        ref_dev = np.abs(ref - center[None, :])
        query_dev = np.abs(x - center)
        counts = np.sum(ref_dev >= query_dev[None, :], axis=0)
        p = (counts + 1.0) / (ref.shape[0] + 1.0)
        return float(np.mean(-np.log(np.maximum(p, 1e-12))))
    if method == "robust_diagonal_nll":
        return float(np.mean(np.log(scale) + 0.5 * ((x - center) / scale) ** 2))
    raise ValueError(f"Unknown conformity method: {method}")


def binary_conformity_predictions(
    values: np.ndarray,
    labels: np.ndarray,
    class_a: int,
    class_b: int,
    selected_features: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Compute LOO A/C conformity predictions in a fixed feature subspace."""
    pair_mask = (labels == class_a) | (labels == class_b)
    global_indices = np.flatnonzero(pair_mask)
    chosen = np.asarray(selected_features, dtype=np.int64)
    if chosen.size == 0:
        raise ValueError("Conformity feature subset cannot be empty.")
    score_matrix = np.full((len(global_indices), 2), np.inf, dtype=np.float64)
    pair_labels = labels[global_indices]
    for local_index, global_index in enumerate(global_indices.tolist()):
        query = values[global_index, chosen]
        for column, class_index in enumerate((class_a, class_b)):
            references = np.flatnonzero(labels == class_index)
            if global_index in references:
                references = references[references != global_index]
            score_matrix[local_index, column] = _one_class_conformity_score(
                query,
                values[references][:, chosen],
                method=method,
            )
    pair_pred = np.where(score_matrix[:, 0] <= score_matrix[:, 1], class_a, class_b)
    full_pred = np.full(labels.shape[0], -1, dtype=np.int64)
    full_pred[global_indices] = pair_pred
    errors = int(np.sum(pair_pred != pair_labels))
    accuracy = float(np.mean(pair_pred == pair_labels))
    return full_pred, score_matrix, errors, accuracy


def _top2_exact_gate(scores: np.ndarray, class_a: int, class_b: int) -> np.ndarray:
    order = np.argsort(np.asarray(scores, dtype=np.float64), axis=1)
    top2 = np.sort(order[:, :2], axis=1)
    low, high = sorted((int(class_a), int(class_b)))
    return (top2[:, 0] == low) & (top2[:, 1] == high)


def _routing_metrics(
    before_pred: np.ndarray,
    after_pred: np.ndarray,
    labels: np.ndarray,
    gate: np.ndarray,
    *,
    stage: str,
    policy: str,
) -> RoutingMetrics:
    before = before_pred[gate] == labels[gate]
    after = after_pred[gate] == labels[gate]
    global_correct = after_pred == labels
    return RoutingMetrics(
        stage=stage,
        policy=policy,
        gated_count=int(np.sum(gate)),
        gated_changed=int(np.sum(after_pred[gate] != before_pred[gate])),
        gated_improved=int(np.sum((~before) & after)),
        gated_worsened=int(np.sum(before & (~after))),
        global_correct=int(np.sum(global_correct)),
        global_errors=int(np.sum(~global_correct)),
        global_accuracy=float(np.mean(global_correct)),
    )


def _threshold_state_intervals(margins: np.ndarray) -> list[tuple[int, float, float, float]]:
    unique = np.unique(np.asarray(margins, dtype=np.float64))
    if unique.size == 0:
        return []
    scale = max(1.0, float(np.max(np.abs(unique))))
    epsilon = np.finfo(np.float64).eps * 64.0 * scale
    rows: list[tuple[int, float, float, float]] = []
    rows.append((0, float("-inf"), float(unique[0]), float(unique[0] - epsilon)))
    for index, lower in enumerate(unique.tolist()):
        upper = float(unique[index + 1]) if index + 1 < len(unique) else float("inf")
        representative = float(lower + (upper - lower) / 2.0) if math.isfinite(upper) else float(lower + epsilon)
        rows.append((index + 1, float(lower), upper, representative))
    return rows


def apply_os_threshold(
    baseline_scores: np.ndarray,
    baseline_pred: np.ndarray,
    labels: np.ndarray,
    score_o: np.ndarray,
    score_s: np.ndarray,
    *,
    class_o: int,
    class_s: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, RoutingMetrics, int, float]:
    """Apply the TRAIN-derived O/S margin threshold inside the exact Top-2 gate."""
    margin = np.asarray(score_o, dtype=np.float64) - np.asarray(score_s, dtype=np.float64)
    pair_pred = np.where(margin <= float(threshold), int(class_o), int(class_s))
    gate = _top2_exact_gate(baseline_scores, class_o, class_s)
    pred = baseline_pred.copy()
    pred[gate] = pair_pred[gate]
    metrics = _routing_metrics(baseline_pred, pred, labels, gate, stage="O/S", policy=f"tau={threshold:.12g}")
    pair_mask = (labels == class_o) | (labels == class_s)
    pair_errors = int(np.sum(pair_pred[pair_mask] != labels[pair_mask]))
    pair_accuracy = float(np.mean(pair_pred[pair_mask] == labels[pair_mask]))
    return pred, pair_pred, metrics, pair_errors, pair_accuracy


def os_threshold_intervals(
    baseline_scores: np.ndarray,
    baseline_pred: np.ndarray,
    labels: np.ndarray,
    score_o: np.ndarray,
    score_s: np.ndarray,
    *,
    class_o: int,
    class_s: int,
) -> list[ThresholdInterval]:
    """Enumerate constant-decision intervals for the O/S threshold."""
    pair_mask = (labels == class_o) | (labels == class_s)
    pair_margin = (np.asarray(score_o) - np.asarray(score_s))[pair_mask]
    rows: list[ThresholdInterval] = []
    for state_index, lower, upper, representative in _threshold_state_intervals(pair_margin):
        _, _, metrics, pair_errors, pair_accuracy = apply_os_threshold(
            baseline_scores,
            baseline_pred,
            labels,
            score_o,
            score_s,
            class_o=class_o,
            class_s=class_s,
            threshold=representative,
        )
        rows.append(
            ThresholdInterval(
                state_index=state_index,
                lower=lower,
                upper=upper,
                lower_inclusive=bool(math.isfinite(lower)),
                upper_inclusive=False,
                threshold_representative=representative,
                pair_errors=pair_errors,
                pair_accuracy=pair_accuracy,
                gated_count=metrics.gated_count,
                gated_changed=metrics.gated_changed,
                gated_improved=metrics.gated_improved,
                gated_worsened=metrics.gated_worsened,
                global_errors_after_os=metrics.global_errors,
                global_accuracy_after_os=metrics.global_accuracy,
            )
        )
    return rows


def merge_best_threshold_plateaus(rows: Sequence[ThresholdInterval]) -> list[dict[str, Any]]:
    """Merge adjacent O/S threshold intervals sharing the best TRAIN objective."""
    if not rows:
        return []
    best_pair = min(int(row.pair_errors) for row in rows)
    best_global = min(int(row.global_errors_after_os) for row in rows if int(row.pair_errors) == best_pair)
    selected = [
        row
        for row in rows
        if int(row.pair_errors) == best_pair and int(row.global_errors_after_os) == best_global
    ]
    selected.sort(key=lambda row: row.state_index)
    groups: list[list[ThresholdInterval]] = []
    for row in selected:
        if not groups or row.state_index != groups[-1][-1].state_index + 1:
            groups.append([row])
        else:
            groups[-1].append(row)

    result: list[dict[str, Any]] = []
    for plateau_index, group in enumerate(groups, start=1):
        lower = group[0].lower
        upper = group[-1].upper
        width = float("inf") if not (math.isfinite(lower) and math.isfinite(upper)) else float(upper - lower)
        if math.isfinite(lower) and math.isfinite(upper):
            midpoint = float(lower + (upper - lower) / 2.0)
        elif math.isfinite(lower):
            midpoint = float(group[0].threshold_representative)
        else:
            midpoint = float(group[-1].threshold_representative)
        result.append(
            {
                "plateau_id": plateau_index,
                "state_index_start": group[0].state_index,
                "state_index_end": group[-1].state_index,
                "lower": lower,
                "upper": upper,
                "width": width,
                "midpoint": midpoint,
                "pair_errors": best_pair,
                "pair_accuracy": float(group[0].pair_accuracy),
                "global_errors_after_os": best_global,
                "global_accuracy_after_os": float(group[0].global_accuracy_after_os),
                "interval_count": len(group),
            }
        )
    return result


def select_threshold_plateau(plateaus: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the widest best plateau with a deterministic near-zero tie break."""
    if not plateaus:
        return None
    finite = [plateau for plateau in plateaus if math.isfinite(float(plateau["width"]))]
    pool = finite or list(plateaus)
    return sorted(
        pool,
        key=lambda plateau: (
            -float(plateau["width"]),
            abs(float(plateau["midpoint"])),
            int(plateau["plateau_id"]),
        ),
    )[0]


def c_one_vs_rest_top_indices(values: np.ndarray, labels: np.ndarray, class_c: int, top_n: int) -> np.ndarray:
    """Select the TRAIN-only C-vs-rest Fisher-ranked feature subset."""
    scores = fisher_scores(values[labels == class_c], values[labels != class_c])
    order = np.argsort(-scores, kind="stable")
    return order[: min(max(1, int(top_n)), len(order))]


def apply_c_route(
    current_pred: np.ndarray,
    baseline_pred: np.ndarray,
    baseline_scores: np.ndarray,
    pair_pred_full: np.ndarray,
    labels: np.ndarray,
    *,
    class_a: int,
    class_c: int,
    policy: str,
    confounder_indices: Sequence[int] = (),
    c_signed_margin: np.ndarray | None = None,
    margin_threshold: float = 0.0,
) -> tuple[np.ndarray, RoutingMetrics, np.ndarray]:
    """Apply the configured conservative C rescue gate."""
    top1 = np.asarray(baseline_pred, dtype=np.int64)
    if policy == "top2_exact_ac":
        gate = _top2_exact_gate(baseline_scores, class_a, class_c)
    elif policy == "top1_a":
        gate = top1 == class_a
    elif policy == "top1_a_rescue_c_only":
        gate = (top1 == class_a) & (pair_pred_full == class_c)
    elif policy == "top1_a_or_c":
        gate = (top1 == class_a) | (top1 == class_c)
    elif policy == "top1_confounder_rescue_c_only":
        allowed = np.asarray(tuple(int(value) for value in confounder_indices), dtype=np.int64)
        gate = np.isin(top1, allowed) & (pair_pred_full == class_c)
    elif policy == "top1_a_margin_rescue":
        if c_signed_margin is None:
            raise ValueError("top1_a_margin_rescue requires c_signed_margin")
        gate = (top1 == class_a) & (pair_pred_full == class_c) & (np.asarray(c_signed_margin) >= float(margin_threshold))
    else:
        raise ValueError(f"Unknown C rescue policy: {policy}")
    pred = current_pred.copy()
    pred[gate] = pair_pred_full[gate]
    return pred, _routing_metrics(current_pred, pred, labels, gate, stage="C rescue", policy=policy), gate
