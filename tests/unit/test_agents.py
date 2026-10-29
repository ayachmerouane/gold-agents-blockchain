"""Agents (§6, §10.2) : règles de la stratégie et du risque, orchestrateur, registre."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gold_agents.agents.execution import ExecutionAgent
from gold_agents.agents.forecast import ForecastAgent, ForecastTable
from gold_agents.agents.messages import Forecast, MarketState, Order, Signal
from gold_agents.agents.orchestrator import AgentSystem, run_agents
from gold_agents.agents.risk import RiskAgent
from gold_agents.agents.strategy import StrategyAgent, prob_up
from gold_agents.blockchain.chain import Chain
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.strategies import Action, BuyAndHold

DAY = date(2024, 1, 2)
COSTS = CostModel(0.001, 0.0002)


def _prices(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes), tz="UTC")
    opens = [closes[0], *closes[:-1]]
    return pd.DataFrame({"date": dates, "open": opens, "close": closes})


def _table(frame: pd.DataFrame, r: list[float], sigma: list[float] | None = None) -> ForecastTable:
    days = [d.date() for d in frame["date"]]
    return ForecastTable("test", days, np.array(r), None if sigma is None else np.array(sigma))


def _system(
    table: ForecastTable,
    k: float | None = None,
    stop: float = 0.99,
    mdd: float = 0.99,
    costs: CostModel = COSTS,
) -> AgentSystem:
    return AgentSystem(
        "agents_test",
        ForecastAgent(table),
        StrategyAgent(0.0, 0.0, k),
        RiskAgent(1.0, stop, mdd),
        ExecutionAgent(costs),
    )


def _state(
    value: float = 1000.0, qty: float = 0.0, close: float = 100.0, entry: float | None = None
) -> MarketState:
    return MarketState(DAY, close, value - qty * close, qty, value, entry)


# --- Stratégie -----------------------------------------------------------------------


def test_strategy_thresholds() -> None:
    agent = StrategyAgent(theta_buy=0.001, theta_sell=0.002)
    assert agent.act(Forecast(DAY, 0.0015, None, "m"), False).action is Action.BUY
    assert agent.act(Forecast(DAY, 0.0015, None, "m"), True).action is Action.HOLD
    assert agent.act(Forecast(DAY, -0.0015, None, "m"), True).action is Action.HOLD
    assert agent.act(Forecast(DAY, -0.0025, None, "m"), True).action is Action.SELL
    assert agent.act(Forecast(DAY, -0.0025, None, "m"), False).action is Action.HOLD
    assert agent.act(Forecast(DAY, None, None, "m"), False).reason == "pas de prévision"


def test_strategy_buys_only_when_confident() -> None:
    agent = StrategyAgent(0.0, 0.0, min_confidence=0.5)
    sure = agent.act(Forecast(DAY, 0.002, 0.002, "m"), False)  # z = 1
    assert sure.action is Action.BUY and sure.strength == pytest.approx(prob_up(1.0))
    unsure = agent.act(Forecast(DAY, 0.002, 0.008, "m"), False)  # z = 0,25
    assert unsure.action is Action.HOLD and "incertain" in unsure.reason
    no_sigma = agent.act(Forecast(DAY, 0.002, None, "m"), False)
    assert no_sigma.action is Action.HOLD
    # Pour sortir, pas besoin d'être sûr.
    assert agent.act(Forecast(DAY, -0.001, 0.01, "m"), True).action is Action.SELL


def test_prob_up() -> None:
    assert prob_up(0.0) == pytest.approx(0.5)
    assert prob_up(0.5) == pytest.approx(0.6915, abs=1e-4)
    assert prob_up(-1.0) == pytest.approx(1 - prob_up(1.0))


# --- Risque --------------------------------------------------------------------------


def _signal(action: Action) -> Signal:
    return Signal(DAY, action, 1.0, "test")


def test_risk_passes_valid_signals_and_rejects_impossible_ones() -> None:
    risk = RiskAgent(max_position=0.5)
    buy = risk.act(_signal(Action.BUY), _state())
    assert buy.executable and buy.fraction == 0.5 and buy.rule == "signal"
    assert not risk.act(_signal(Action.SELL), _state()).approved  # rien à vendre
    assert not risk.act(_signal(Action.BUY), _state(qty=5.0, entry=100.0)).approved


def test_risk_stop_loss_overrides_strategy() -> None:
    risk = RiskAgent(stop_loss=0.05)
    order = risk.act(_signal(Action.HOLD), _state(qty=10.0, close=94.0, entry=100.0))
    assert order.action is Action.SELL and order.rule == "stop_loss"
    ok = risk.act(_signal(Action.HOLD), _state(qty=10.0, close=96.0, entry=100.0))
    assert ok.action is Action.HOLD


def test_risk_max_drawdown_halts_for_good() -> None:
    risk = RiskAgent(max_drawdown=0.2)
    risk.act(_signal(Action.HOLD), _state(value=1000.0, qty=10.0, close=100.0, entry=90.0))
    order = risk.act(_signal(Action.BUY), _state(value=790.0, qty=10.0, close=79.0, entry=90.0))
    assert order.action is Action.SELL and order.rule == "max_drawdown" and risk.halted
    later = risk.act(_signal(Action.BUY), _state(value=790.0))
    assert not later.executable and later.rule == "rejet"


def test_execution_skips_hold_and_rejected_orders() -> None:
    from gold_agents.simulation.portfolio import Portfolio

    row = pd.Series({"date": pd.Timestamp("2024-01-03"), "open": 100.0, "close": 101.0})
    agent = ExecutionAgent(CostModel())
    p = Portfolio(cash=1000.0)
    assert agent.act(Order(DAY, Action.HOLD, 0.0, True, "signal", ""), row, p) is None
    assert agent.act(Order(DAY, Action.BUY, 1.0, False, "rejet", ""), row, p) is None
    trade = agent.act(Order(DAY, Action.BUY, 1.0, True, "signal", "go"), row, p)
    assert trade is not None and trade.qty == pytest.approx(10.0) and p.cash == 0


# --- Orchestrateur -------------------------------------------------------------------


def test_always_positive_forecast_is_buy_and_hold() -> None:
    """Même moteur, mêmes coûts : un agent toujours haussier refait exactement Buy & Hold."""
    rng = np.random.default_rng(1)
    frame = _prices(list(1800 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))))
    run = run_agents(frame, _system(_table(frame, [0.001] * 200)), 10_000, DAY)
    bh = run_backtest(frame, BuyAndHold(), COSTS, 10_000, DAY)
    assert np.allclose(run.result.equity["value"], bh.equity["value"], rtol=1e-12)
    assert len(run.result.trades) == len(bh.trades) == 1


def test_orders_execute_next_open_and_stop_loss_fires() -> None:
    closes = [100.0, 100.0, 100.0, 90.0, 90.0, 95.0, 95.0, 95.0]
    frame = _prices(closes)
    run = run_agents(
        frame,
        _system(_table(frame, [0.01] * 8), stop=0.05, costs=CostModel()),
        1000,
        date(2024, 1, 1),
    )
    sides = [(t.side, t.exec_date) for t in run.result.trades]
    days = [d.date() for d in frame["date"]]
    # Achat décidé le jour 0, exécuté à l'ouverture du jour 1 ; stop-loss vu à la clôture du
    # jour 3 (-10 %), vente à l'ouverture du jour 4, puis rachat car la prévision reste > 0.
    assert sides[:3] == [("BUY", days[1]), ("SELL", days[4]), ("BUY", days[5])]
    assert run.decisions.loc[3, "rule"] == "stop_loss"


def test_confidence_filter_keeps_agent_in_cash() -> None:
    frame = _prices([100.0 + i for i in range(30)])
    table = _table(frame, [0.001] * 30, [0.01] * 30)  # z = 0,1 : jamais assez sûr
    run = run_agents(frame, _system(table, k=0.5), 1000, date(2024, 1, 1))
    assert run.result.trades == []
    assert run.result.equity["value"].iloc[-1] == 1000


def test_every_decision_is_in_the_ledger() -> None:
    frame = _prices([100.0, 101.0, 99.0, 102.0, 103.0])
    ledger = Chain.start([{"type": "GENESIS"}])
    table = _table(frame, [0.01, -0.01, 0.01, 0.01, 0.0], [0.01] * 5)
    run_agents(frame, _system(table), 1000, date(2024, 1, 1), ledger=ledger)
    assert ledger.verify().valid and len(ledger) == 6
    for block in ledger.blocks[1:-1]:
        kinds = [e["type"] for e in block.events]
        assert {"FORECAST", "SIGNAL", "ORDER", "PORTFOLIO"} <= set(kinds)
    assert ledger.blocks[2].events[0]["type"] == "TRADE"  # achat de la veille
    forecast = ledger.blocks[1].events[0]
    assert forecast["r_hat"] == 0.01 and forecast["z"] == pytest.approx(1.0)


def test_day_without_forecast_holds() -> None:
    frame = _prices([100.0, 101.0, 102.0])
    table = ForecastTable("test", [frame["date"].iloc[1].date()], np.array([0.01]))
    run = run_agents(frame, _system(table), 1000, date(2024, 1, 1))
    assert list(run.decisions["signal"]) == ["HOLD", "BUY"]
