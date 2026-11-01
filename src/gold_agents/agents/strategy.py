"""Agent 2, stratégie (§6.2) : transforme une prévision en signal BUY / SELL / HOLD.

Règle à seuils : BUY si `r̂ > θ_buy`, SELL si `r̂ < -θ_sell`, sinon HOLD. En v1 on est soit
tout investi, soit tout en cash : un BUY quand on est déjà investi, ou un SELL quand on
ne l'est pas, devient HOLD.

Filtre de confiance (optionnel, `min_confidence = k`) : on n'ACHÈTE que si le modèle est
sûr de lui, c'est-à-dire si `z = r̂ / σ̂ > k` (σ̂ du MC Dropout). Sous une loi normale,
`Φ(z)` est la probabilité de hausse selon le modèle : k = 0,5 demande au moins 69 %.
Pour VENDRE, aucune confiance n'est exigée : dans le doute, on réduit le risque.

Seuil d'achat relatif (`buy_rule = "quantile"`, tentative 2 du test) : θ_buy n'est plus un
rendement mais un niveau de quantile q. On achète si r̂ dépasse le quantile q des
prévisions des `quantile_window` jours PRÉCÉDENTS du même modèle (rien du jour ni du
futur). Un seuil fixe en % dépend de l'échelle des prévisions, qui change d'un
réentraînement à l'autre : au test, l'ensemble ne dépassait jamais 0,09 % alors que le
seuil figé était de 0,10 %, d'où zéro trade. Le quantile, lui, ne dépend pas de l'échelle.
Tant qu'il y a moins de `quantile_min_days` prévisions passées, pas d'achat ; et jamais
d'achat sur une baisse prévue (seuil au moins égal à 0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from gold_agents.agents.messages import Forecast, Signal
from gold_agents.simulation.strategies import Action


def prob_up(z: float) -> float:
    """Φ(z) : probabilité que le rendement soit positif si r ~ N(r̂, σ̂²)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


BuyRule = Literal["absolute", "quantile"]


def past_quantile(history: Sequence[float], q: float, window: int, min_days: int) -> float:
    """Quantile q des `window` dernières prévisions passées ; NaN s'il y en a trop peu.

    Partagé par l'agent et le simulateur rapide : ils doivent donner le même seuil.
    """
    recent = history[-window:]
    if len(recent) < min_days:
        return math.nan
    return float(np.quantile(np.asarray(recent, dtype="float64"), q))


def relative_threshold(history: Sequence[float], q: float, window: int, min_days: int) -> float:
    """Seuil d'achat relatif : le quantile passé, jamais sous 0 (pas d'achat sur une baisse
    prévue)."""
    t = past_quantile(history, q, window, min_days)
    return t if math.isnan(t) else max(t, 0.0)


@dataclass
class StrategyAgent:
    theta_buy: float = 0.0  # rendement (règle absolue) ou niveau de quantile (règle relative)
    theta_sell: float = 0.0
    min_confidence: float | None = None  # k ; None : pas de filtre de confiance
    buy_rule: BuyRule = "absolute"
    quantile_window: int = 250
    quantile_min_days: int = 60

    def __post_init__(self) -> None:
        # Prévisions déjà vues (hors champ : n'apparaît pas dans le bloc genèse).
        self._history: list[float] = []

    def buy_threshold(self) -> float:
        """Seuil d'achat du jour, calculé avant d'ajouter la prévision du jour."""
        if self.buy_rule == "absolute":
            return self.theta_buy
        return relative_threshold(
            self._history, self.theta_buy, self.quantile_window, self.quantile_min_days
        )

    def _threshold_text(self, threshold: float) -> str:
        if self.buy_rule == "absolute":
            return f"θ_buy {self.theta_buy:g}"
        return f"seuil {threshold:+.5f} (quantile {self.theta_buy:.0%} des prévisions passées)"

    @property
    def name(self) -> str:
        return "strategie" if self.min_confidence is None else "strategie_confiance"

    def act(self, forecast: Forecast, invested: bool) -> Signal:
        day, r = forecast.date, forecast.r_hat
        if r is None:
            return Signal(day, Action.HOLD, 0.0, "pas de prévision")
        threshold = self.buy_threshold()
        self._history.append(r)
        return self._decide(forecast, r, threshold, invested)

    def _decide(self, forecast: Forecast, r: float, threshold: float, invested: bool) -> Signal:
        day = forecast.date
        z = forecast.z
        p = prob_up(z) if z is not None else None
        # Seuil NaN (historique trop court) : `r > NaN` est faux, donc jamais d'achat.
        if r > threshold:
            if invested:
                return Signal(day, Action.HOLD, p or 1.0, f"r̂ {r:+.5f} > θ, déjà investi")
            k = self.min_confidence
            if k is not None and (z is None or z <= k):
                shown = "sans σ̂" if z is None else f"z {z:.2f} ≤ {k:g}"
                return Signal(
                    day, Action.HOLD, p or 0.0, f"r̂ {r:+.5f} > θ mais incertain ({shown})"
                )
            why = f"r̂ {r:+.5f} > {self._threshold_text(threshold)}"
            return Signal(day, Action.BUY, p or 1.0, why)
        if r < -self.theta_sell:
            down = 1 - p if p is not None else 1.0
            if not invested:
                return Signal(day, Action.HOLD, down, f"r̂ {r:+.5f} < -θ, déjà en cash")
            return Signal(day, Action.SELL, down, f"r̂ {r:+.5f} < -θ_sell {self.theta_sell:g}")
        if math.isnan(threshold):
            return Signal(day, Action.HOLD, 0.0, f"r̂ {r:+.5f}, historique trop court")
        return Signal(day, Action.HOLD, 0.0, f"r̂ {r:+.5f} entre les seuils")
