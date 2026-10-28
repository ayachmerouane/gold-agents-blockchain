"""Réseau récurrent commun au LSTM (M1) et au GRU (M2) (§5.2).

Entrée `(N, W, F)` : W jours de F features. La couche récurrente lit les jours dans
l'ordre ; son état après le jour t passe par un dropout puis une couche linéaire qui
donne le rendement prévu de t+1 (une valeur par exemple, sortie `(N,)`).
Le dropout avant la tête sert aussi au MC Dropout (`uncertainty.py`), même avec une
seule couche récurrente (le dropout interne de PyTorch n'agit qu'entre deux couches).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetConfig:
    n_features: int
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.2

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class RecurrentRegressor(nn.Module):
    cell: type[nn.LSTM] | type[nn.GRU] = nn.LSTM

    def __init__(self, net: NetConfig) -> None:
        super().__init__()
        self.net = net
        self.rnn = self.cell(
            input_size=net.n_features,
            hidden_size=net.hidden,
            num_layers=net.layers,
            dropout=net.dropout if net.layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(net.dropout)
        self.head = nn.Linear(net.hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # état après le dernier jour de la fenêtre (le jour t)
        pred: torch.Tensor = self.head(self.dropout(last)).squeeze(-1)
        return pred
