"""Messages typés échangés entre les agents (§6.2).

Chaque jour t, l'orchestrateur fait circuler, dans l'ordre :

    MarketState -> [Prévision] -> Forecast -> [Stratégie] -> Signal -> [Risque] -> Order
    -> [Exécution, à l'ouverture de t+1] -> Trade

Les messages sont figés (`frozen`) : un agent ne peut pas modifier ce qu'il a reçu. Chacun
sait se résumer en dictionnaire JSON (`event`) pour être écrit dans le registre.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from gold_agents.simulation.strategies import Action


def _event(kind: str, message: Any) -> dict[str, Any]:
    data = asdict(message)
    out: dict[str, Any] = {"type": kind}
    for key, value in data.items():
        if isinstance(value, date):
            value = value.isoformat()
        elif isinstance(value, Action):
            value = value.value
        out[key] = value
    return out


@dataclass(frozen=True)
class MarketState:
    """Ce que l'orchestrateur sait le jour t, après la clôture (rien au-delà de t)."""

    date: date
    close: float
    cash: float
    qty: float
    value: float
    entry_price: float | None  # prix de revient de la position (frais compris), ou None

    @property
    def invested(self) -> bool:
        return self.qty > 0


@dataclass(frozen=True)
class Forecast:
    """Prévision du rendement log de t à t+1, et son incertitude si le modèle en donne."""

    date: date
    r_hat: float | None  # None : pas de prévision ce jour-là
    sigma_hat: float | None
    model_id: str

    @property
    def z(self) -> float | None:
        """Confiance : prévision rapportée à son incertitude (r̂ / σ̂)."""
        if self.r_hat is None or not self.sigma_hat:
            return None
        return self.r_hat / self.sigma_hat

    def event(self) -> dict[str, Any]:
        return {**_event("FORECAST", self), "z": self.z}


@dataclass(frozen=True)
class Signal:
    date: date
    action: Action
    strength: float  # confiance de la stratégie, de 0 à 1 (informatif en v1)
    reason: str

    def event(self) -> dict[str, Any]:
        return _event("SIGNAL", self)


@dataclass(frozen=True)
class Order:
    """Ordre validé (ou refusé) par l'agent de risque, exécuté à l'ouverture de t+1."""

    date: date
    action: Action
    fraction: float  # part du cash investie par un achat
    approved: bool
    rule: str  # règle qui a décidé : "signal", "stop_loss", "max_drawdown", "rejet"…
    reason: str

    @property
    def executable(self) -> bool:
        return self.approved and self.action is not Action.HOLD

    def event(self) -> dict[str, Any]:
        return _event("ORDER", self)
