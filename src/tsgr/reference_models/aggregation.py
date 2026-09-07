"""Hierarchical frame-to-session-to-person-to-gesture aggregation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from tsgr.reference_models.covariance import shrink_covariance, weighted_empirical_covariance


@dataclass(slots=True)
class SessionStatistics:
    gesture_id: str
    person_id: str
    session_id: str
    accepted_values: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    standard_deviation: np.ndarray
    q1: np.ndarray
    q3: np.ndarray
    total_frame_count: int
    accepted_frame_count: int
    outlier_frame_count: int


@dataclass(slots=True)
class PersonStatistics:
    gesture_id: str
    person_id: str
    selected_session_ids: tuple[str, ...]
    mean: np.ndarray
    median: np.ndarray
    standard_deviation: np.ndarray
    q1: np.ndarray
    q3: np.ndarray


@dataclass(slots=True)
class GestureStatistics:
    gesture_id: str
    session_statistics: list[SessionStatistics]
    person_statistics: list[PersonStatistics]
    prototype_mean: np.ndarray
    prototype_median: np.ndarray
    standard_deviation: np.ndarray
    q1: np.ndarray
    q3: np.ndarray
    covariance_empirical: np.ndarray
    covariance_regularized: np.ndarray
    diagonal_variance: np.ndarray
    shrinkage: float


def summarize_session(
    *,
    gesture_id: str,
    person_id: str,
    session_id: str,
    accepted_values: np.ndarray,
    total_frame_count: int,
) -> SessionStatistics:
    matrix = np.asarray(accepted_values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"Session {gesture_id}/{person_id}/{session_id} has no accepted frames.")
    return SessionStatistics(
        gesture_id=gesture_id,
        person_id=person_id,
        session_id=session_id,
        accepted_values=matrix,
        mean=np.mean(matrix, axis=0),
        median=np.median(matrix, axis=0),
        standard_deviation=np.std(matrix, axis=0),
        q1=np.quantile(matrix, 0.25, axis=0),
        q3=np.quantile(matrix, 0.75, axis=0),
        total_frame_count=int(total_frame_count),
        accepted_frame_count=int(matrix.shape[0]),
        outlier_frame_count=int(total_frame_count - matrix.shape[0]),
    )


def _deterministic_session_order(
    sessions: list[SessionStatistics],
    *,
    seed: int,
) -> list[SessionStatistics]:
    def key(session: SessionStatistics) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{seed}:{session.gesture_id}:{session.person_id}:{session.session_id}".encode("utf-8")
        ).hexdigest()
        return digest, session.session_id

    return sorted(sessions, key=key)


def balance_sessions(
    sessions: list[SessionStatistics],
    *,
    mode: str = "minimum_available",
    fixed_count: int | None = None,
    seed: int = 2026,
) -> dict[str, list[SessionStatistics]]:
    """Select an equal number of sessions per person for one gesture."""
    by_person: dict[str, list[SessionStatistics]] = {}
    for session in sessions:
        by_person.setdefault(session.person_id, []).append(session)
    if not by_person:
        raise ValueError("Cannot balance an empty session list.")
    counts = [len(value) for value in by_person.values()]
    if mode == "minimum_available":
        selected_count = min(counts)
    elif mode == "fixed_count":
        if fixed_count is None or fixed_count < 1:
            raise ValueError("fixed_count mode requires a positive fixed_count.")
        if min(counts) < fixed_count:
            raise ValueError(
                f"At least one person has fewer than {fixed_count} sessions: {counts}."
            )
        selected_count = fixed_count
    elif mode == "all_equal_person_weight":
        selected_count = -1
    else:
        raise ValueError(f"Unsupported session balance mode: {mode}")
    selected: dict[str, list[SessionStatistics]] = {}
    for person_id, person_sessions in sorted(by_person.items()):
        ordered = _deterministic_session_order(person_sessions, seed=seed)
        selected[person_id] = ordered if selected_count < 0 else ordered[:selected_count]
    return selected


def aggregate_gesture(
    gesture_id: str,
    sessions: list[SessionStatistics],
    *,
    session_balance_mode: str,
    fixed_session_count: int | None,
    balance_seed: int,
    covariance_shrinkage: float,
    covariance_diagonal_floor: float,
) -> GestureStatistics:
    """Build an equal-person gesture prototype and covariance statistics."""
    selected_by_person = balance_sessions(
        sessions,
        mode=session_balance_mode,
        fixed_count=fixed_session_count,
        seed=balance_seed,
    )
    people: list[PersonStatistics] = []
    balanced_session_means: list[np.ndarray] = []
    balanced_session_weights: list[float] = []
    for person_id, selected_sessions in selected_by_person.items():
        session_means = np.vstack([session.mean for session in selected_sessions])
        session_medians = np.vstack([session.median for session in selected_sessions])
        people.append(
            PersonStatistics(
                gesture_id=gesture_id,
                person_id=person_id,
                selected_session_ids=tuple(session.session_id for session in selected_sessions),
                mean=np.mean(session_means, axis=0),
                median=np.median(session_medians, axis=0),
                standard_deviation=np.std(session_means, axis=0),
                q1=np.quantile(session_means, 0.25, axis=0),
                q3=np.quantile(session_means, 0.75, axis=0),
            )
        )
        sample_weight = 1.0 / (len(selected_by_person) * len(selected_sessions))
        for session in selected_sessions:
            balanced_session_means.append(session.mean)
            balanced_session_weights.append(sample_weight)
    person_means = np.vstack([person.mean for person in people])
    prototype_mean = np.mean(person_means, axis=0)
    prototype_median = np.median(person_means, axis=0)
    covariance_samples = np.vstack(balanced_session_means)
    covariance = weighted_empirical_covariance(
        covariance_samples, np.asarray(balanced_session_weights, dtype=np.float64)
    )
    regularized = shrink_covariance(
        covariance,
        shrinkage=covariance_shrinkage,
        diagonal_floor=covariance_diagonal_floor,
    )
    return GestureStatistics(
        gesture_id=gesture_id,
        session_statistics=sessions,
        person_statistics=people,
        prototype_mean=prototype_mean,
        prototype_median=prototype_median,
        standard_deviation=np.std(person_means, axis=0),
        q1=np.quantile(person_means, 0.25, axis=0),
        q3=np.quantile(person_means, 0.75, axis=0),
        covariance_empirical=covariance,
        covariance_regularized=regularized,
        diagonal_variance=np.diag(regularized).copy(),
        shrinkage=float(covariance_shrinkage),
    )
