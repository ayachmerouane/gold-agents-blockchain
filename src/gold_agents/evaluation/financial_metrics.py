"""Métriques financières d'une courbe de valeur de portefeuille (§7.4)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.simulation.engine import BacktestResult

TRADING_DAYS = 252


def daily_returns(values: pd.Series) -> pd.Series:
    return values.pct_change().dropna()


def drawdown(values: pd.Series) -> pd.Series:
    """Baisse depuis le plus haut atteint jusque-là (0 au plus haut, -0.2 = -20 %)."""
    return values / values.cummax() - 1


def max_drawdown(values: pd.Series) -> float:
    return float(drawdown(values).min())


def cagr(values: pd.Series, dates: pd.Series) -> float:
    """Rendement annualisé composé, sur la durée calendaire de la période."""
    years = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days / 365.25
    if years <= 0:
        return 0.0
    return float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)


def annual_volatility(values: pd.Series) -> float:
    r = daily_returns(values)
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def sharpe(values: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Gain moyen par unité de risque, annualisé. 0 si la courbe ne bouge pas."""
    r = daily_returns(values) - risk_free_rate / TRADING_DAYS
    std = r.std(ddof=1)
    if len(r) < 2 or not std > 1e-12:
        return 0.0
    return float(r.mean() / std * math.sqrt(TRADING_DAYS))


def sortino(values: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Comme le Sharpe, mais seules les baisses comptent comme risque."""
    r = daily_returns(values) - risk_free_rate / TRADING_DAYS
    downside = math.sqrt(float((np.minimum(r, 0) ** 2).mean())) if len(r) else 0.0
    if downside <= 1e-12:
        return 0.0
    return float(r.mean() / downside * math.sqrt(TRADING_DAYS))


def round_trip_pnl(result: BacktestResult) -> list[float]:
    """Gain net de chaque aller-retour fermé (achat puis vente), frais compris."""
    pnl: list[float] = []
    entry: float | None = None
    for t in result.trades:
        if t.side == "BUY":
            entry = t.qty * t.price + t.fees
        elif entry is not None:
            pnl.append(t.qty * t.price - t.fees - entry)
            entry = None
    return pnl


def summarize(result: BacktestResult, risk_free_rate: float = 0.0) -> dict[str, Any]:
    # Le premier jour, rien n'est encore exécuté : la courbe part du capital initial.
    eq = result.equity
    values = eq["value"]
    final = float(values.iloc[-1])
    mdd = max_drawdown(values)
    growth = cagr(values, eq["date"])
    pnl = round_trip_pnl(result)
    fees = sum(t.fees for t in result.trades)
    slip = sum(t.slippage_cost for t in result.trades)
    return {
        "strategy": result.strategy,
        "start": str(eq["date"].iloc[0]),
        "end": str(eq["date"].iloc[-1]),
        "days": len(eq),
        "initial_capital": result.initial_capital,
        "final_value": final,
        "net_profit": final - result.initial_capital,
        "total_return": final / result.initial_capital - 1,
        "cagr": growth,
        "annual_volatility": annual_volatility(values),
        "sharpe": sharpe(values, risk_free_rate),
        "sortino": sortino(values, risk_free_rate),
        "max_drawdown": mdd,
        "calmar": growth / abs(mdd) if mdd < 0 else 0.0,
        "n_trades": len(result.trades),
        "n_round_trips": len(pnl),
        "win_rate": sum(p > 0 for p in pnl) / len(pnl) if pnl else None,
        "total_fees": fees,
        "total_slippage": slip,
        "total_costs": fees + slip,
        "exposure": float((eq["qty"] > 0).mean()),
    }
