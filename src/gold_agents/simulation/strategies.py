"""Stratégies de référence (§7.3). Toutes passent par le même moteur.

Une stratégie reçoit l'historique jusqu'au jour t inclus (jamais au-delà) et la quantité
d'or détenue ; elle renvoie une décision BUY / SELL / HOLD. En v1, BUY investit tout le
cash disponible (dans la limite de la taille de position) et SELL vend toute la position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import pandas as pd


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


HOLD = Decision(Action.HOLD, "rien à faire")


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, history: pd.DataFrame, qty: float) -> Decision:
        """`history` : lignes jusqu'au jour t inclus ; `qty` : onces détenues."""
        ...


class BuyAndHold:
    """Achète au premier jour puis ne touche plus à rien (référence minimale exigée)."""

    name = "buy_and_hold"

    def decide(self, history: pd.DataFrame, qty: float) -> Decision:
        if qty == 0:
            return Decision(Action.BUY, "Buy & Hold : achat initial")
        return Decision(Action.HOLD, "Buy & Hold : on garde")


@dataclass
class SmaCrossover:
    """Croisement de moyennes mobiles : investi quand SMA courte > SMA longue."""

    fast: int = 20
    slow: int = 50

    @property
    def name(self) -> str:
        return f"sma_{self.fast}_{self.slow}"

    def decide(self, history: pd.DataFrame, qty: float) -> Decision:
        close = history["close"]
        if len(close) < self.slow:
            return HOLD
        fast = float(close.iloc[-self.fast :].mean())
        slow = float(close.iloc[-self.slow :].mean())
        if fast > slow and qty == 0:
            return Decision(Action.BUY, f"SMA{self.fast} {fast:.2f} > SMA{self.slow} {slow:.2f}")
        if fast < slow and qty > 0:
            return Decision(Action.SELL, f"SMA{self.fast} {fast:.2f} < SMA{self.slow} {slow:.2f}")
        return HOLD


STRATEGIES: dict[str, type[BuyAndHold] | type[SmaCrossover]] = {
    "buy_and_hold": BuyAndHold,
    "sma_20_50": SmaCrossover,
}
