"""Agent 3, risque (§6.2) : valide, refuse ou remplace le signal de la stratégie.

Règles, dans l'ordre de priorité :

1. **Drawdown max** : si la valeur du portefeuille tombe à `-max_drawdown` sous son plus
   haut, on vend tout et on passe en cash (coupe-circuit). Sans `cooldown_days`, on y
   reste jusqu'à la fin de la période ; avec, on reprend après `cooldown_days` jours en
   cash, et le plus haut repart de la valeur du jour de la reprise.
2. **Stop-loss** : si le cours tombe à `-stop_loss` sous le prix de revient de la
   position, on vend, quel que soit l'avis de la stratégie.
3. **Signal** : un BUY est accepté s'il reste du cash (taille `max_position` du cash),
   un SELL s'il y a de l'or. Pas de vente à découvert en v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gold_agents.agents.messages import MarketState, Order, Signal
from gold_agents.simulation.strategies import Action


@dataclass
class RiskAgent:
    max_position: float = 1.0
    stop_loss: float = 0.05
    max_drawdown: float = 0.20
    allow_short: bool = False
    cooldown_days: int | None = None  # jours en cash avant la reprise ; None : jamais
    peak_value: float = field(default=0.0, init=False)
    halted: bool = field(default=False, init=False)
    days_in_cash: int = field(default=0, init=False)  # depuis le coupe-circuit

    name = "risque"

    def __post_init__(self) -> None:
        if self.allow_short:
            raise NotImplementedError("vente à découvert non gérée en v1")

    def act(self, signal: Signal, state: MarketState) -> Order:
        day = signal.date
        if self.halted and not state.invested and self.cooldown_days is not None:
            self.days_in_cash += 1
            if self.days_in_cash > self.cooldown_days:  # reprise : nouveau plus haut
                self.halted, self.days_in_cash, self.peak_value = False, 0, state.value
        self.peak_value = max(self.peak_value, state.value)
        drawdown = state.value / self.peak_value - 1 if self.peak_value > 0 else 0.0

        if not self.halted and drawdown <= -self.max_drawdown:
            self.halted = True
            if state.invested:
                return Order(
                    day,
                    Action.SELL,
                    0.0,
                    True,
                    "max_drawdown",
                    f"perte de {drawdown:.1%} depuis le plus haut : passage en cash",
                )
        if self.halted:
            if state.invested:  # un SELL du coupe-circuit n'a pas pu s'exécuter
                return Order(day, Action.SELL, 0.0, True, "max_drawdown", "coupe-circuit actif")
            if signal.action is Action.BUY:
                return Order(day, Action.HOLD, 0.0, False, "rejet", "coupe-circuit : plus d'achat")
            return Order(day, Action.HOLD, 0.0, True, "max_drawdown", "coupe-circuit : en cash")

        if state.invested and state.entry_price:
            loss = state.close / state.entry_price - 1
            if loss <= -self.stop_loss:
                return Order(
                    day, Action.SELL, 0.0, True, "stop_loss", f"{loss:.1%} sous le prix de revient"
                )

        if signal.action is Action.BUY:
            if state.invested or state.cash <= 0:
                return Order(day, Action.HOLD, 0.0, False, "rejet", "pas de cash disponible")
            return Order(day, Action.BUY, self.max_position, True, "signal", signal.reason)
        if signal.action is Action.SELL:
            if not state.invested:
                return Order(day, Action.HOLD, 0.0, False, "rejet", "rien à vendre")
            return Order(day, Action.SELL, 0.0, True, "signal", signal.reason)
        return Order(day, Action.HOLD, 0.0, True, "signal", signal.reason)
