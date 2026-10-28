"""M1 : LSTM (§5.2). Trois portes (oubli, entrée, sortie) et une mémoire à long terme."""

from __future__ import annotations

from torch import nn

from gold_agents.models.recurrent import RecurrentRegressor


class LSTMRegressor(RecurrentRegressor):
    cell = nn.LSTM
