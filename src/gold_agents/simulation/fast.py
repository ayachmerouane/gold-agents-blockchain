"""Simulation rapide pour le réglage (phase 6) : mêmes règles, sans objets ni registre.

La grille de la phase 6 demande des milliers de simulations. Ce module refait, sur des
tableaux numpy, exactement ce que font l'orchestrateur et les quatre agents (§6.2) :
stratégie à seuils et filtre de confiance, agent de risque (stop-loss, coupe-circuit et
reprise), exécution à l'ouverture du lendemain avec les mêmes frais et le même slippage.
Un test vérifie qu'il donne la même courbe de valeur que `agents.orchestrator.run_agents`.

`simulate_positions` sert aux références sans prévision : Buy & Hold, stratégie
aléatoire et oracle. On lui donne, pour chaque jour, la position voulue le lendemain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gold_agents.simulation.costs import CostModel

TRADING_DAYS = 252


@dataclass(frozen=True)
class AgentParams:
    theta_buy: float = 0.0
    theta_sell: float = 0.0
    min_confidence: float | None = None  # k ; None : pas de filtre
    stop_loss: float = 0.05
    max_drawdown: float = 0.20
    max_position: float = 1.0
    cooldown_days: int | None = None


@dataclass(frozen=True)
class FastResult:
    values: np.ndarray  # valeur à la clôture de chaque jour
    invested: np.ndarray  # True si de l'or est détenu à la clôture
    trade_days: np.ndarray  # indice du jour d'exécution de chaque trade
    trade_sides: np.ndarray  # +1 achat, -1 vente

    @property
    def n_trades(self) -> int:
        return len(self.trade_days)

    def metrics(self, risk_free_rate: float = 0.0) -> dict[str, float]:
        return {
            "sharpe": sharpe(self.values, risk_free_rate),
            "total_return": float(self.values[-1] / self.values[0] - 1),
            "worst_drawdown": max_drawdown(self.values),
            "n_trades": float(self.n_trades),
            "exposure": float(self.invested.mean()),
        }


def sharpe(values: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Comme `evaluation.financial_metrics.sharpe`, sur un tableau."""
    r = values[1:] / values[:-1] - 1 - risk_free_rate / TRADING_DAYS
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if not std > 1e-12:
        return 0.0
    return float(r.mean() / std * math.sqrt(TRADING_DAYS))


def max_drawdown(values: np.ndarray) -> float:
    return float((values / np.maximum.accumulate(values) - 1).min())


class _Book:
    """Portefeuille et exécution à l'ouverture, comme `simulation.engine.execute_order`."""

    def __init__(self, capital: float, costs: CostModel) -> None:
        self.cash, self.qty, self.entry = capital, 0.0, 0.0
        self.fee, self.slip = costs.fee_rate, costs.slippage
        self.days: list[int] = []
        self.sides: list[int] = []

    def buy(self, day: int, market_open: float, fraction: float) -> None:
        if self.qty != 0:
            return
        price = market_open * (1 + self.slip)
        qty = self.cash * fraction / (price * (1 + self.fee))
        if qty <= 0:
            return
        fee = qty * price * self.fee
        self.cash = max(self.cash - (qty * price + fee), 0.0)
        self.qty = qty
        self.entry = (price * qty + fee) / qty
        self.days.append(day)
        self.sides.append(1)

    def sell(self, day: int, market_open: float) -> None:
        if self.qty <= 0:
            return
        price = market_open * (1 - self.slip)
        self.cash += self.qty * price - self.qty * price * self.fee
        self.qty, self.entry = 0.0, 0.0
        self.days.append(day)
        self.sides.append(-1)

    def result(self, values: list[float], invested: list[bool]) -> FastResult:
        return FastResult(
            np.asarray(values), np.asarray(invested), np.asarray(self.days), np.asarray(self.sides)
        )


def simulate_agents(
    opens: np.ndarray,
    closes: np.ndarray,
    r_hat: np.ndarray,
    sigma_hat: np.ndarray | None,
    params: AgentParams,
    costs: CostModel,
    capital: float,
) -> FastResult:
    """Agents sur une période : `r_hat[t]` (NaN : pas de prévision) décide pour t+1."""
    n = len(closes)
    book = _Book(capital, costs)
    k = params.min_confidence
    peak, halted, days_in_cash = 0.0, False, 0
    pending = 0  # +1 achat, -1 vente, 0 rien
    values: list[float] = []
    invested: list[bool] = []
    for t in range(n):
        if pending == 1:
            book.buy(t, float(opens[t]), params.max_position)
        elif pending == -1:
            book.sell(t, float(opens[t]))
        pending = 0
        close = float(closes[t])
        value = book.cash + book.qty * close
        values.append(value)
        holding = book.qty > 0
        invested.append(holding)
        if t == n - 1:
            break

        # Agent de stratégie.
        signal = 0
        r = float(r_hat[t])
        if not math.isnan(r):
            if r > params.theta_buy:
                if not holding:
                    s = float(sigma_hat[t]) if sigma_hat is not None else 0.0
                    z = r / s if s else None
                    if k is None or (z is not None and z > k):
                        signal = 1
            elif r < -params.theta_sell and holding:
                signal = -1

        # Agent de risque.
        if halted and not holding and params.cooldown_days is not None:
            days_in_cash += 1
            if days_in_cash > params.cooldown_days:
                halted, days_in_cash, peak = False, 0, value
        peak = max(peak, value)
        drawdown = value / peak - 1 if peak > 0 else 0.0
        if not halted and drawdown <= -params.max_drawdown:
            halted = True
        if halted:
            pending = -1 if holding else 0
            continue
        if holding and book.entry and close / book.entry - 1 <= -params.stop_loss:
            pending = -1
            continue
        if signal == 1 and book.cash > 0:
            pending = 1
        elif signal == -1:
            pending = -1
    return book.result(values, invested)


def simulate_positions(
    opens: np.ndarray,
    closes: np.ndarray,
    want: np.ndarray,
    costs: CostModel,
    capital: float,
    max_position: float = 1.0,
) -> FastResult:
    """`want[t]` : être investi le lendemain (True) ou en cash (False), décidé le jour t."""
    n = len(closes)
    book = _Book(capital, costs)
    values: list[float] = []
    invested: list[bool] = []
    for t in range(n):
        if t > 0:
            if want[t - 1] and book.qty == 0:
                book.buy(t, float(opens[t]), max_position)
            elif not want[t - 1] and book.qty > 0:
                book.sell(t, float(opens[t]))
        values.append(book.cash + book.qty * float(closes[t]))
        invested.append(book.qty > 0)
    return book.result(values, invested)


def random_positions(n_days: int, n_trades: int, rng: np.random.Generator) -> np.ndarray:
    """Positions voulues d'une stratégie au hasard qui fait exactement `n_trades` trades.

    On tire `n_trades` jours de décision distincts ; à chacun on change de position
    (achat, puis vente, puis achat…). Le dernier jour n'a pas de lendemain : exclu.
    """
    n_trades = min(n_trades, n_days - 1)
    want = np.zeros(n_days, dtype=bool)
    if n_trades <= 0:
        return want
    days = np.sort(rng.choice(n_days - 1, size=n_trades, replace=False))
    flips = np.zeros(n_days, dtype=int)
    flips[days] = 1
    want[:] = np.cumsum(flips) % 2 == 1
    return want


def oracle_positions(opens: np.ndarray) -> np.ndarray:
    """L'oracle connaît le futur : investi demain si l'ouverture d'après-demain est plus haute.

    Pour illustration seulement (§7.3) : c'est le plafond qu'aucune prévision n'atteint.
    """
    want = np.zeros(len(opens), dtype=bool)
    want[:-2] = opens[2:] > opens[1:-1]
    return want
