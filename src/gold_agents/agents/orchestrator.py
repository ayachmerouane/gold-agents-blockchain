"""Orchestrateur (§6.1) : fait agir les quatre agents chaque jour, dans l'ordre.

La boucle suit exactement les règles du moteur de la phase 3 (`simulation.engine`) :

1. l'ordre décidé la veille est exécuté à l'OUVERTURE de t (agent d'exécution) ;
2. le portefeuille est valorisé à la CLÔTURE de t ;
3. prévision -> stratégie -> risque décident pour le lendemain, avec ce qui est connu à
   la clôture de t et rien d'autre. La décision du dernier jour n'est pas exécutée.

Chaque jour ajoute un bloc au registre : trade éventuel, prévision, signal, ordre (avec
la règle qui l'a décidé) et état du portefeuille. Toute décision est donc traçable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from gold_agents.agents.execution import ExecutionAgent
from gold_agents.agents.forecast import ForecastAgent
from gold_agents.agents.messages import MarketState, Order
from gold_agents.agents.risk import RiskAgent
from gold_agents.agents.strategy import StrategyAgent
from gold_agents.blockchain.chain import Chain
from gold_agents.simulation.engine import BacktestResult, Trade, jsonable
from gold_agents.simulation.portfolio import Portfolio


@dataclass
class AgentSystem:
    name: str
    forecaster: ForecastAgent
    strategy: StrategyAgent
    risk: RiskAgent
    execution: ExecutionAgent

    def describe(self) -> dict[str, Any]:
        """Paramètres des agents, inscrits dans le bloc genèse du registre."""
        return {
            "name": self.name,
            "model_id": self.forecaster.table.model_id,
            "strategy": asdict(self.strategy),
            "risk": {
                "max_position": self.risk.max_position,
                "stop_loss": self.risk.stop_loss,
                "max_drawdown": self.risk.max_drawdown,
                "allow_short": self.risk.allow_short,
                "cooldown_days": self.risk.cooldown_days,
            },
            "costs": asdict(self.execution.costs),
        }


@dataclass(frozen=True)
class AgentRun:
    result: BacktestResult
    decisions: pd.DataFrame  # une ligne par jour : prévision, signal, ordre, règle


def _day(ts: Any) -> date:
    return pd.Timestamp(ts).date()


def run_agents(
    frame: pd.DataFrame,
    system: AgentSystem,
    initial_capital: float,
    start: date,
    end: date | None = None,
    ledger: Chain | None = None,
) -> AgentRun:
    """Simule `system` de `start` à `end` inclus sur `frame` (date, open, close, triés)."""
    frame = frame.reset_index(drop=True)
    days = frame["date"].map(_day)
    in_period = (days >= start) & ((days <= end) if end is not None else True)
    positions = frame.index[in_period]
    if len(positions) < 2:
        raise ValueError("période de simulation trop courte (au moins 2 jours)")

    portfolio = Portfolio(cash=initial_capital)
    entry_price: float | None = None
    trades: list[Trade] = []
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    pending: Order | None = None
    last = positions[-1]

    for i in positions:
        row = frame.loc[i]
        today = _day(row["date"])
        trade = None
        if pending is not None:
            trade = system.execution.act(pending, row, portfolio)
            if trade is not None:
                trades.append(trade)
                # Prix de revient frais compris, pour le stop-loss de l'agent de risque.
                entry_price = (trade.price * trade.qty + trade.fees) / trade.qty
                if trade.side == "SELL":
                    entry_price = None
        close = float(row["close"])
        value = portfolio.value(close)

        events: list[dict[str, Any]] = []
        if trade is not None:
            events.append({"type": "TRADE", **jsonable(asdict(trade))})
        action = "END"
        if i != last:
            state = MarketState(today, close, portfolio.cash, portfolio.qty, value, entry_price)
            forecast = system.forecaster.act(state)
            signal = system.strategy.act(forecast, state.invested)
            order = system.risk.act(signal, state)
            pending = order
            action = order.action.value if order.approved else "HOLD"
            events += [forecast.event(), signal.event(), order.event()]
            decisions.append(
                {
                    "date": today,
                    "r_hat": forecast.r_hat,
                    "sigma_hat": forecast.sigma_hat,
                    "z": forecast.z,
                    "signal": signal.action.value,
                    "strength": signal.strength,
                    "order": order.action.value,
                    "approved": order.approved,
                    "rule": order.rule,
                    "reason": order.reason,
                }
            )
        else:
            events.append({"type": "DECISION", "action": "END", "reason": "fin de période"})
        events.append(
            {"type": "PORTFOLIO", "cash": portfolio.cash, "qty": portfolio.qty, "value": value}
        )
        rows.append(
            {
                "date": today,
                "close": close,
                "cash": portfolio.cash,
                "qty": portfolio.qty,
                "value": value,
                "action": action,
            }
        )
        if ledger is not None:
            ledger.append(today.isoformat(), events)

    result = BacktestResult(system.name, initial_capital, pd.DataFrame(rows), trades)
    return AgentRun(result, pd.DataFrame(decisions))
