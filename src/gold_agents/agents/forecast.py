"""Agent 1, prévision (§6.2) : donne `r̂(t+1)` et `σ̂(t+1)` pour le jour t.

Les modèles sont figés : ils ont été entraînés en phase 4 et ne sont jamais réentraînés
pendant la simulation. La prévision du jour t ne lit que la fenêtre des W jours qui
finissent à t (voir `data.windows`), donc jamais de données au-delà de t. Pour aller vite,
toutes les fenêtres d'une période sont prédites d'un coup (`ForecastTable`) ; c'est
équivalent, puisque chaque fenêtre ne contient que son propre passé.

- `r̂` : moyenne des réseaux (toutes les graines) en mode évaluation, la même prévision
  que dans les résultats de la phase 4 ;
- `σ̂` : MC Dropout (§5.3), écart-type des prévisions tirées avec le dropout actif, sur
  toutes les graines et tous les réseaux de l'ensemble.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.agents.messages import Forecast, MarketState


@dataclass(frozen=True)
class ForecastTable:
    """Prévisions d'un modèle pour chaque jour d'une période."""

    model_id: str
    dates: list[date]
    r_hat: np.ndarray
    sigma_hat: np.ndarray | None = None  # None : le modèle ne donne pas d'incertitude

    def __post_init__(self) -> None:
        n = len(self.dates)
        if len(self.r_hat) != n or (self.sigma_hat is not None and len(self.sigma_hat) != n):
            raise ValueError("autant de prévisions que de dates attendues")

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame({"date": self.dates, "r_hat": self.r_hat})
        out["sigma_hat"] = np.nan if self.sigma_hat is None else self.sigma_hat
        return out


class ForecastAgent:
    def __init__(self, table: ForecastTable) -> None:
        self.table = table
        self._index = {d: i for i, d in enumerate(table.dates)}

    @property
    def name(self) -> str:
        return f"prevision_{self.table.model_id}"

    def act(self, state: MarketState) -> Forecast:
        i = self._index.get(state.date)
        if i is None:  # jour sans fenêtre complète : pas d'avis
            return Forecast(state.date, None, None, self.table.model_id)
        sigma = None if self.table.sigma_hat is None else float(self.table.sigma_hat[i])
        return Forecast(state.date, float(self.table.r_hat[i]), sigma, self.table.model_id)


def _days(dates: pd.DatetimeIndex) -> list[date]:
    return [pd.Timestamp(d).date() for d in dates]


def network_table(
    model_id: str,
    networks: Sequence[Any],
    x: np.ndarray,
    dates: pd.DatetimeIndex,
    mc_samples: int = 50,
    seed: int = 0,
) -> ForecastTable:
    """Ensemble de réseaux (`TrainedNetwork`, une ou plusieurs graines chacun).

    `mc_samples` passages MC Dropout par réseau et par graine ; 0 : pas de `σ̂`.
    """
    from gold_agents.models.uncertainty import mc_draws

    r_hat = np.mean([net.predict(x) for net in networks], axis=0)
    sigma = None
    if mc_samples > 0:
        draws = [
            mc_draws(model, x, mc_samples, seed + k) * net.y_scale
            for net in networks
            for k, model in enumerate(net.models)
        ]
        sigma = np.concatenate(draws).std(axis=0).astype("float64")
    return ForecastTable(model_id, _days(dates), np.asarray(r_hat, dtype="float64"), sigma)


def lightgbm_table(
    models: Sequence[Any], x_tabular: np.ndarray, dates: pd.DatetimeIndex
) -> ForecastTable:
    """LightGBM (B4), le concurrent sans réseau : moyenne des graines, sans `σ̂`."""
    r_hat = np.mean([m.predict(x_tabular) for m in models], axis=0).astype("float64")
    return ForecastTable("lightgbm", _days(dates), r_hat, None)
