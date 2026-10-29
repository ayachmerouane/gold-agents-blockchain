"""M2 : GRU (§5.2). Deux portes au lieu de trois : environ 25 % de paramètres en moins
que le LSTM de même taille, moins de risque de surapprentissage avec peu de données."""

from __future__ import annotations

from torch import nn

from gold_agents.models.recurrent import RecurrentRegressor


class GRURegressor(RecurrentRegressor):
    cell = nn.GRU
