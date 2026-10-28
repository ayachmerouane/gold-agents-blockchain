"""Propriétés du moteur pour des prix et des ordres quelconques (§10.5)."""

from datetime import date

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.strategies import Action, Decision


class Random:
    name = "random"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions

    def decide(self, history: pd.DataFrame, qty: float) -> Decision:
        return Decision(self.actions[len(history) - 1], "hasard")


@settings(max_examples=60, deadline=None)
@given(
    prices=st.lists(st.floats(1, 10_000), min_size=2, max_size=40),
    actions=st.lists(st.sampled_from(list(Action)), min_size=40, max_size=40),
    fee=st.floats(0, 0.05),
    slip=st.floats(0, 0.05),
)
def test_cash_qty_value_and_fees_never_negative(
    prices: list[float], actions: list[Action], fee: float, slip: float
) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(prices), tz="UTC")
    frame = pd.DataFrame({"date": dates, "open": prices, "close": prices[::-1]})
    res = run_backtest(frame, Random(actions), CostModel(fee, slip), 1000, date(2024, 1, 1))
    eq = res.equity
    assert (eq["cash"] >= 0).all() and (eq["qty"] >= 0).all() and (eq["value"] >= 0).all()
    assert all(t.fees >= 0 and t.slippage_cost >= 0 for t in res.trades)
