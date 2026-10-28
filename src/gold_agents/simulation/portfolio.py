"""Portefeuille (§7.1) : argent liquide et quantité d'or en onces (fractionnable).

Pas d'effet de levier ni de vente à découvert en v1 : le cash et la quantité ne peuvent
jamais devenir négatifs.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tolérance des arrondis flottants quand on dépense tout le cash.
_EPS = 1e-9


class PortfolioError(ValueError):
    pass


@dataclass
class Portfolio:
    cash: float
    qty: float = 0.0

    def value(self, price: float) -> float:
        """Valeur totale au prix `price` : cash + quantité × prix."""
        return self.cash + self.qty * price

    def buy(self, qty: float, price: float, fee: float) -> None:
        cost = qty * price + fee
        if qty <= 0 or cost > self.cash * (1 + _EPS) + _EPS:
            raise PortfolioError(f"achat impossible : {qty:.6f} oz coûtent {cost:.2f}")
        self.cash = max(self.cash - cost, 0.0)
        self.qty += qty

    def sell(self, qty: float, price: float, fee: float) -> None:
        if qty <= 0 or qty > self.qty * (1 + _EPS) + _EPS:
            raise PortfolioError(f"vente impossible : {qty:.6f} oz pour {self.qty:.6f} détenues")
        self.cash += qty * price - fee
        self.qty = max(self.qty - qty, 0.0)
