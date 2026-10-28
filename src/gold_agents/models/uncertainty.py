"""Incertitude par MC Dropout (§5.3) : on garde le dropout actif à l'inférence et on
répète la prévision ; l'écart-type des passages donne `σ̂`, utile aux agents pour
réduire la taille d'une position quand le modèle hésite."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mc_dropout(
    model: nn.Module, x: np.ndarray, n_samples: int = 50, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne et écart-type de `n_samples` prévisions avec dropout actif."""
    torch.manual_seed(seed)
    model.train()  # active le dropout ; aucun poids n'est modifié (pas d'optimiseur)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        draws = np.stack([model(xt).numpy() for _ in range(n_samples)])
    model.eval()
    return draws.mean(axis=0), draws.std(axis=0)
