"""Moteur de backtest jour par jour (§7).

Pour chaque jour t de la période :

1. l'ordre décidé la veille est exécuté à l'OUVERTURE de t (règle d'or §2.2 : une
   décision prise avec la clôture de t-1 n'est jamais exécutée à ce prix-là) ;
2. le portefeuille est valorisé à la CLÔTURE de t ;
3. la stratégie reçoit l'historique jusqu'à t inclus, et seulement lui, et décide pour
   le lendemain. La décision du dernier jour n'est pas exécutée (pas de lendemain).

Si un registre (`Chain`) est fourni, chaque jour y ajoute un bloc : décision, trade
éventuel et état du portefeuille.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from gold_agents.blockchain.chain import Chain
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.portfolio import Portfolio
from gold_agents.simulation.strategies import Action, Decision, Strategy


@dataclass(frozen=True)
class Trade:
    decision_date: date
    exec_date: date
    side: str  # "BUY" ou "SELL"
    market_open: float
    price: float  # prix obtenu, slippage compris
    qty: float
    fees: float
    slippage_cost: float
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    initial_capital: float
    equity: pd.DataFrame  # date, close, cash, qty, value, action
    trades: list[Trade]

    def trades_frame(self) -> pd.DataFrame:
        columns = list(Trade.__dataclass_fields__)
        return pd.DataFrame([asdict(t) for t in self.trades], columns=columns)


def _day(ts: Any) -> date:
    return pd.Timestamp(ts).date()


def _execute(
    decision: Decision,
    decided_on: date,
    row: pd.Series,
    portfolio: Portfolio,
    costs: CostModel,
    max_position: float,
) -> Trade | None:
    market_open = float(row["open"])
    if decision.action is Action.BUY and portfolio.qty == 0:
        price = costs.buy_price(market_open)
        budget = portfolio.cash * max_position
        qty = budget / (price * (1 + costs.fee_rate))
        if qty <= 0:
            return None
        fee = costs.fee(qty * price)
        portfolio.buy(qty, price, fee)
    elif decision.action is Action.SELL and portfolio.qty > 0:
        price = costs.sell_price(market_open)
        qty = portfolio.qty
        fee = costs.fee(qty * price)
        portfolio.sell(qty, price, fee)
    else:
        return None
    return Trade(
        decision_date=decided_on,
        exec_date=_day(row["date"]),
        side=decision.action.value,
        market_open=market_open,
        price=price,
        qty=qty,
        fees=fee,
        slippage_cost=abs(price - market_open) * qty,
        reason=decision.reason,
    )


def run_backtest(
    frame: pd.DataFrame,
    strategy: Strategy,
    costs: CostModel,
    initial_capital: float,
    start: date,
    end: date | None = None,
    max_position: float = 1.0,
    ledger: Chain | None = None,
) -> BacktestResult:
    """Simule `strategy` de `start` à `end` inclus sur `frame` (date, open, close, triés).

    Les lignes avant `start` servent seulement d'historique à la stratégie.
    """
    frame = frame.reset_index(drop=True)
    days = frame["date"].map(_day)
    in_period = (days >= start) & ((days <= end) if end is not None else True)
    positions = frame.index[in_period]
    if len(positions) < 2:
        raise ValueError("période de backtest trop courte (au moins 2 jours)")

    portfolio = Portfolio(cash=initial_capital)
    trades: list[Trade] = []
    rows: list[dict[str, Any]] = []
    pending: tuple[Decision, date] | None = None
    last = positions[-1]

    for i in positions:
        row = frame.loc[i]
        today = _day(row["date"])
        trade = None
        if pending is not None:
            trade = _execute(*pending, row, portfolio, costs, max_position)
            if trade is not None:
                trades.append(trade)
        close = float(row["close"])
        value = portfolio.value(close)

        decision = None
        if i != last:
            decision = strategy.decide(frame.iloc[: i + 1], portfolio.qty)
            pending = (decision, today)
        action = decision.action.value if decision else "END"
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
            events: list[dict[str, Any]] = []
            if trade is not None:
                events.append({"type": "TRADE", **_jsonable(asdict(trade))})
            events.append(
                {
                    "type": "DECISION",
                    "strategy": strategy.name,
                    "action": action,
                    "reason": decision.reason if decision else "fin de période",
                    "close": close,
                }
            )
            events.append(
                {"type": "PORTFOLIO", "cash": portfolio.cash, "qty": portfolio.qty, "value": value}
            )
            ledger.append(today.isoformat(), events)

    return BacktestResult(strategy.name, initial_capital, pd.DataFrame(rows), trades)


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v.isoformat() if isinstance(v, date) else v for k, v in data.items()}
