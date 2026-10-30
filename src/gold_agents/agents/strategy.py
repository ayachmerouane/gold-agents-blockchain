"""Agent 2, stratégie (§6.2) : transforme une prévision en signal BUY / SELL / HOLD.

Règle à seuils : BUY si `r̂ > θ_buy`, SELL si `r̂ < -θ_sell`, sinon HOLD. En v1 on est soit
tout investi, soit tout en cash : un BUY quand on est déjà investi, ou un SELL quand on
ne l'est pas, devient HOLD.

Filtre de confiance (optionnel, `min_confidence = k`) : on n'ACHÈTE que si le modèle est
sûr de lui, c'est-à-dire si `z = r̂ / σ̂ > k` (σ̂ du MC Dropout). Sous une loi normale,
`Φ(z)` est la probabilité de hausse selon le modèle : k = 0,5 demande au moins 69 %.
Pour VENDRE, aucune confiance n'est exigée : dans le doute, on réduit le risque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gold_agents.agents.messages import Forecast, Signal
from gold_agents.simulation.strategies import Action


def prob_up(z: float) -> float:
    """Φ(z) : probabilité que le rendement soit positif si r ~ N(r̂, σ̂²)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@dataclass
class StrategyAgent:
    theta_buy: float = 0.0
    theta_sell: float = 0.0
    min_confidence: float | None = None  # k ; None : pas de filtre de confiance

    @property
    def name(self) -> str:
        return "strategie" if self.min_confidence is None else "strategie_confiance"

    def act(self, forecast: Forecast, invested: bool) -> Signal:
        day, r = forecast.date, forecast.r_hat
        if r is None:
            return Signal(day, Action.HOLD, 0.0, "pas de prévision")
        z = forecast.z
        p = prob_up(z) if z is not None else None
        if r > self.theta_buy:
            if invested:
                return Signal(day, Action.HOLD, p or 1.0, f"r̂ {r:+.5f} > θ, déjà investi")
            k = self.min_confidence
            if k is not None and (z is None or z <= k):
                shown = "sans σ̂" if z is None else f"z {z:.2f} ≤ {k:g}"
                return Signal(
                    day, Action.HOLD, p or 0.0, f"r̂ {r:+.5f} > θ mais incertain ({shown})"
                )
            return Signal(day, Action.BUY, p or 1.0, f"r̂ {r:+.5f} > θ_buy {self.theta_buy:g}")
        if r < -self.theta_sell:
            down = 1 - p if p is not None else 1.0
            if not invested:
                return Signal(day, Action.HOLD, down, f"r̂ {r:+.5f} < -θ, déjà en cash")
            return Signal(day, Action.SELL, down, f"r̂ {r:+.5f} < -θ_sell {self.theta_sell:g}")
        return Signal(day, Action.HOLD, 0.0, f"r̂ {r:+.5f} entre les seuils")
