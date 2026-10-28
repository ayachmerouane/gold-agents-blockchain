"""Moteur de backtest (§7, §10.2) : portefeuille, coûts, exécution au lendemain."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.portfolio import Portfolio, PortfolioError
from gold_agents.simulation.strategies import (
    HOLD,
    Action,
    BuyAndHold,
    Decision,
    SmaCrossover,
)

NO_COSTS = CostModel()


def _prices(opens: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    closes = closes or opens
    dates = pd.bdate_range("2024-01-01", periods=len(opens), tz="UTC")
    return pd.DataFrame({"date": dates, "open": opens, "close": closes})


class Scripted:
    """Joue une suite d'actions fixée et note ce qu'elle a vu."""

    name = "scripted"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.seen: list[pd.Timestamp] = []

    def decide(self, history: pd.DataFrame, qty: float) -> Decision:
        self.seen.append(history["date"].iloc[-1])
        return Decision(self.actions[len(self.seen) - 1], "script")


def test_portfolio_refuses_overspending_and_short() -> None:
    p = Portfolio(cash=100.0)
    with pytest.raises(PortfolioError):
        p.buy(1.0, 101.0, 0.0)
    with pytest.raises(PortfolioError):
        p.sell(0.5, 100.0, 0.0)
    p.buy(1.0, 99.0, 1.0)
    assert p.cash == 0.0 and p.value(120.0) == 120.0


def test_buy_and_hold_without_costs_tracks_price_exactly() -> None:
    rng = np.random.default_rng(0)
    opens = list(1800 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    closes = [o * 1.002 for o in opens]
    res = run_backtest(_prices(opens, closes), BuyAndHold(), NO_COSTS, 10_000, date(2024, 1, 1))
    # Décidé le 1er jour, exécuté à l'ouverture du 2e : valeur = capital × clôture / ouverture.
    expected = 10_000 * closes[-1] / opens[1]
    assert res.equity["value"].iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert res.equity["value"].iloc[0] == 10_000
    assert len(res.trades) == 1


def test_value_is_cash_plus_qty_times_close() -> None:
    frame = _prices([100, 102, 101, 105, 103, 108])
    strategy = Scripted([Action.BUY, Action.HOLD, Action.SELL, Action.BUY, Action.HOLD])
    res = run_backtest(frame, strategy, CostModel(0.001, 0.0002), 1000, date(2024, 1, 1))
    eq = res.equity
    np.testing.assert_allclose(eq["value"], eq["cash"] + eq["qty"] * eq["close"])
    assert (eq["cash"] >= 0).all() and (eq["qty"] >= 0).all()


def test_no_trade_keeps_value_constant() -> None:
    frame = _prices([100, 150, 80, 120])
    res = run_backtest(frame, Scripted([Action.HOLD] * 3), NO_COSTS, 500, date(2024, 1, 1))
    assert (res.equity["value"] == 500).all() and not res.trades


def test_costs_are_deducted() -> None:
    frame = _prices([100.0] * 4)
    costs = CostModel(fee_rate=0.01, slippage=0.005)
    res = run_backtest(
        frame, Scripted([Action.BUY, Action.SELL, Action.HOLD]), costs, 1000, date(2024, 1, 1)
    )
    buy, sell = res.trades
    assert buy.price == pytest.approx(100.5) and sell.price == pytest.approx(99.5)
    assert buy.qty * buy.price + buy.fees == pytest.approx(1000)
    assert buy.fees == pytest.approx(0.01 * buy.qty * 100.5)
    final = res.equity["value"].iloc[-1]
    assert final == pytest.approx(buy.qty * 99.5 * 0.99)
    assert final < 1000


def test_execution_happens_at_next_open() -> None:
    frame = _prices([100, 110, 120, 130], closes=[105, 115, 125, 135])
    res = run_backtest(
        frame, Scripted([Action.BUY, Action.SELL, Action.HOLD]), NO_COSTS, 1000, date(2024, 1, 1)
    )
    for trade, open_ in zip(res.trades, [110, 120], strict=True):
        assert trade.exec_date > trade.decision_date
        assert trade.market_open == open_


def test_strategy_never_sees_the_future() -> None:
    frame = _prices(list(range(100, 120)))
    strategy = Scripted([Action.HOLD] * 19)
    res = run_backtest(frame, strategy, NO_COSTS, 1000, date(2024, 1, 3))
    # Une décision par jour sauf le dernier, chacune avec l'historique jusqu'à ce jour-là.
    assert strategy.seen == list(frame["date"].iloc[2:-1])
    assert res.equity["action"].iloc[-1] == "END"


def test_history_before_start_is_visible_but_not_traded() -> None:
    frame = _prices(list(np.linspace(100, 200, 80)))
    res = run_backtest(frame, SmaCrossover(5, 20), NO_COSTS, 1000, date(2024, 3, 1))
    assert res.equity["date"].iloc[0] == date(2024, 3, 1)
    # La tendance monte depuis longtemps : achat dès le premier jour de la période.
    assert res.trades[0].decision_date == date(2024, 3, 1)


def test_sma_crossover_signals() -> None:
    up = pd.DataFrame({"close": np.linspace(100, 200, 60)})
    down = pd.DataFrame({"close": np.linspace(200, 100, 60)})
    s = SmaCrossover(20, 50)
    assert s.decide(up, 0).action is Action.BUY
    assert s.decide(up, 1).action is Action.HOLD
    assert s.decide(down, 1).action is Action.SELL
    assert s.decide(up.iloc[:10], 0) == HOLD


def test_cost_scaling() -> None:
    c = CostModel(0.001, 0.0002).scaled(2)
    assert c.fee_rate == pytest.approx(0.002) and c.slippage == pytest.approx(0.0004)
