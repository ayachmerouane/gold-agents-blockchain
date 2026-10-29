"""Agent 4, exécution (§6.2) : passe l'ordre validé à l'ouverture de t+1.

Il réutilise l'exécution du moteur de la phase 3 (`simulation.engine.execute_order`) :
mêmes frais, même slippage, même portefeuille que Buy & Hold. Les comparaisons entre
stratégies sont donc équitables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from gold_agents.agents.messages import Order
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import Trade, execute_order
from gold_agents.simulation.portfolio import Portfolio
from gold_agents.simulation.strategies import Decision


@dataclass
class ExecutionAgent:
    costs: CostModel

    name = "execution"

    def act(self, order: Order, row: pd.Series, portfolio: Portfolio) -> Trade | None:
        """`row` : la ligne du jour d'exécution (on lit seulement son ouverture)."""
        if not order.executable:
            return None
        decided_on: date = order.date
        decision = Decision(order.action, f"{order.rule} : {order.reason}")
        return execute_order(decision, decided_on, row, portfolio, self.costs, order.fraction)
