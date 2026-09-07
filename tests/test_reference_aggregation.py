from __future__ import annotations

import numpy as np

from tsgr.reference_models.aggregation import aggregate_gesture, summarize_session


def _session(person: str, session: str, value: float):
    values = np.full((8, 3), value, dtype=np.float64)
    return summarize_session(
        gesture_id="A",
        person_id=person,
        session_id=session,
        accepted_values=values,
        total_frame_count=8,
    )


def test_minimum_available_balances_sessions_and_people() -> None:
    sessions = [
        _session("p1", "s1", 0.0),
        _session("p1", "s2", 100.0),
        _session("p2", "s1", 10.0),
    ]
    result = aggregate_gesture(
        "A",
        sessions,
        session_balance_mode="minimum_available",
        fixed_session_count=None,
        balance_seed=2026,
        covariance_shrinkage=0.1,
        covariance_diagonal_floor=1.0e-8,
    )
    assert len(result.person_statistics) == 2
    assert all(len(person.selected_session_ids) == 1 for person in result.person_statistics)
    expected = np.mean(np.vstack([person.mean for person in result.person_statistics]), axis=0)
    np.testing.assert_allclose(result.prototype_mean, expected)
    assert np.all(np.linalg.eigvalsh(result.covariance_regularized) > 0)
