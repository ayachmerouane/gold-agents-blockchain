"""Coûts de transaction (§7.1) : frais proportionnels et slippage.

Le slippage est l'écart entre le prix affiché et le prix réellement obtenu : on achète un
peu plus cher et on vend un peu moins cher que l'ouverture. Les frais sont un pourcentage
du montant échangé (au prix obtenu).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.0
    slippage: float = 0.0

    def buy_price(self, market_price: float) -> float:
        return market_price * (1 + self.slippage)

    def sell_price(self, market_price: float) -> float:
        return market_price * (1 - self.slippage)

    def fee(self, notional: float) -> float:
        return notional * self.fee_rate

    def scaled(self, factor: float) -> CostModel:
        """Mêmes coûts multipliés par `factor` (analyse de sensibilité : × 0, × 2, × 5)."""
        return CostModel(self.fee_rate * factor, self.slippage * factor)
