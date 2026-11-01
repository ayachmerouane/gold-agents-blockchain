"""Simulation en direct : état du registre et jours à ajouter."""

from datetime import date, timedelta

import pandas as pd

from gold_agents.live.session import LiveState, new_days

D = date(2026, 11, 2)


def test_live_state_round_trip_and_new_days() -> None:
    state = LiveState(cash=10.0, history=[0.1], last_date="2026-11-02")
    assert LiveState.from_event(state.event()) == state
    frame = pd.DataFrame(
        {"date": pd.to_datetime([D - timedelta(days=1), D, D + timedelta(days=1)], utc=True)}
    )
    assert new_days(frame, None, D.isoformat()) == [1, 2]
    assert new_days(frame, D.isoformat(), D.isoformat()) == [2]
